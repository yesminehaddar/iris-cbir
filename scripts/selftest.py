"""Correctness checks on a built index. Run it after any change to the search path.

    python scripts/selftest.py flickr1m

Checks that an image queried against itself scores exactly 0, that descriptors
recomputed from the file on disk reproduce the indexed ones, that the VP-tree
agrees with the exhaustive scan, that gathering the keyword candidates and masking
them out of a full scan give the same answer, and that a filtered search returns
only images that really carry the keywords. Exits non-zero on any failure.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app import datasets, db, search
from app.descriptors import IMPORTED_DESCRIPTORS, compute, load_image
from app.search import FeatureStore

FILTER_WORDS = "dark"
NARROW_WORDS = "dark vivid blue"
REPRODUCE_TOLERANCE = 0.05


class Checker:
    """Prints PASS/FAIL per assertion and counts the failures, so one run reports
    every problem rather than stopping at the first."""

    def __init__(self):
        self.failed = 0

    def __call__(self, label, passed, detail=""):
        if not passed:
            self.failed += 1
        print(f"{'PASS' if passed else 'FAIL'}  {label}{'  ' + detail if detail else ''}")


def self_query_scores_zero(store, check, rng, samples=20):
    """An image is its own nearest neighbour, at distance exactly 0.

    Not necessarily rank 1: identical images tie at 0, and any order is correct
    there. What must hold is that the query's own score is 0 — if the gemv
    identity were used unrefined it would come out at ~5e-4 instead.
    """
    for name in store.descriptor_names:
        nearest, missing = 0.0, 0
        for image_id in rng.choice(store.ids, min(samples, store.size), replace=False):
            hits, _ = store.search(store.vectors_of(int(image_id)), [name], 40)
            nearest = max(nearest, hits[0][1])
            mine = [s for i, s in hits if i == int(image_id)]
            if not mine or mine[0] != 0.0:
                missing += 1
        check(
            f"self-query scores exactly 0 ({name})",
            nearest == 0.0 and missing == 0,
            f"nearest={nearest}, not scoring 0: {missing}",
        )


def upload_reproduces_index(store, dataset, check, rng, samples=5):
    """Recomputing a descriptor from the file must give back the indexed vector.

    This is the invariant behind querying by an uploaded image: the same picture
    uploaded has to land on top of its own indexed row. It is also the only thing
    that catches a mismatch in the column scaling applied to CLD and HTD, since
    the stored matrix is scaled and a freshly computed vector is not.
    """
    with db.connect() as conn:
        rows = db.get_images(
            conn, [int(i) for i in rng.choice(store.ids, min(samples, store.size), False)]
        )
    # The provided descriptors came out of the distribution's archives, computed
    # on originals this base does not ship, so there is nothing to recompute.
    computed = [n for n in store.descriptor_names if n not in IMPORTED_DESCRIPTORS]
    if not computed:
        print("SKIP  no computed descriptors to reproduce")
        return
    worst = 0.0
    for image_id, row in rows.items():
        path = dataset.images_dir / row["url"].removeprefix(dataset.url_prefix + "/")
        if not path.exists():
            check("upload reproduces the index", False, f"missing file {path}")
            return
        fresh = store.apply_scale(compute(load_image(path), computed))
        stored = store.vectors_of(image_id)
        for name in computed:
            worst = max(worst, float(np.abs(fresh[name] - stored[name]).max()))
    # Not exactly 0: on another machine's numpy/scikit-image, float rounding can
    # move one of the Edge Histogram's 1024 blocks across the edge threshold, a
    # difference of 1/64. A scaling mismatch, which is what this guards against,
    # shows up as differences of 1 and more.
    check(
        f"upload reproduces the index ({len(computed)} computed descriptors)",
        worst <= REPRODUCE_TOLERANCE,
        f"largest difference {worst:g}",
    )


def _ground_truth(matrix, query, chunk=100_000):
    """float64 distances to every row, in chunks so a 1.5 GB matrix still fits."""
    out = np.empty(matrix.shape[0], dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    for start in range(0, matrix.shape[0], chunk):
        block = np.asarray(matrix[start : start + chunk], dtype=np.float64)
        out[start : start + chunk] = np.linalg.norm(block - query, axis=1)
    return out


def matches_ground_truth(store, check, rng, k=24, samples=3):
    """Both search paths against distances computed honestly in float64.

    This is the check that caught the one real ranking bug: the scan's
    `||a||^2 - 2a.b + ||b||^2` identity is ill-conditioned for a column-scaled
    descriptor, where it subtracts squared norms of ~19 000 to produce a squared
    distance of ~4. On `mp7_htd_ref` that left more float32 slack than the gap
    between the 24th and 25th neighbour, and the top k came back in the wrong
    order until the scan started over-selecting and ranking the pool exactly.
    """
    for name in store.descriptor_names:
        index = store.indexes[name]
        rows = rng.choice(store.size, size=min(samples, store.size), replace=False)
        scan_ok = tree_ok = 0
        for row in rows:
            query = np.asarray(index.matrix[row])
            truth = _ground_truth(index.matrix, query)
            want = np.sort(truth[np.argsort(truth, kind="stable")[:k]])
            got, _ = index.knn(query, k)
            scan_ok += int(np.allclose(want, np.sort(truth[got]), atol=1e-9))
            if store.has_tree(name):
                tree_rows, _ = store.trees[name].knn(query, k)
                tree_ok += int(np.allclose(want, np.sort(truth[tree_rows]), atol=1e-9))
        check(
            f"scan matches float64 ground truth ({name})",
            scan_ok == len(rows),
            f"{scan_ok}/{len(rows)}",
        )
        if store.has_tree(name):
            check(
                f"tree matches float64 ground truth ({name})",
                tree_ok == len(rows),
                f"{tree_ok}/{len(rows)}",
            )


def tree_agrees_with_scan(store, check, rng, samples=10):
    """The tree prunes; it must not change the answer."""
    trees = [n for n in store.descriptor_names if store.has_tree(n)]
    if not trees:
        print("SKIP  no trees saved; run build_index.py --trees-only")
        return
    for name in trees:
        same = True
        for image_id in rng.choice(store.ids, min(samples, store.size), replace=False):
            vectors = store.vectors_of(int(image_id))
            scan, _ = store.search(vectors, [name], 20, index="brute")
            tree, _ = store.search(vectors, [name], 20, index="vptree")
            same &= np.allclose([s for _, s in scan], [s for _, s in tree], atol=1e-6)
        check(f"vptree == scan ({name})", same)


def gather_matches_mask(store, dataset, check):
    """The two ways of applying a keyword filter must agree exactly."""
    rows = store.rows_for_words(FILTER_WORDS)
    if rows is None or not rows.size:
        print(f"SKIP  no tag index, or nothing matches {FILTER_WORDS!r}")
        return
    saved = search.FULL_SCAN_FRACTION
    vectors = store.vectors_of(int(store.ids[0]))
    try:
        for names in ([store.descriptor_names[0]], list(dataset.descriptors)):
            search.FULL_SCAN_FRACTION = 1.1  # force the gather
            gathered, _ = store.search(vectors, names, 20, candidate_rows=rows)
            search.FULL_SCAN_FRACTION = 0.0  # force the masked full scan
            masked, _ = store.search(vectors, names, 20, candidate_rows=rows)
            agree = [i for i, _ in gathered] == [i for i, _ in masked] and np.allclose(
                [s for _, s in gathered], [s for _, s in masked], atol=1e-9
            )
            check(f"gather == scan+mask ({len(names)} descriptor(s))", agree)
    finally:
        search.FULL_SCAN_FRACTION = saved


def filter_is_respected(store, check):
    """Nothing may come back that does not carry every keyword."""
    rows = store.rows_for_words(NARROW_WORDS)
    if rows is None or not rows.size:
        print(f"SKIP  nothing matches {NARROW_WORDS!r}")
        return
    allowed = set(rows.tolist())
    hits, _ = store.search(
        store.vectors_of(int(store.ids[0])), [store.descriptor_names[0]], 15,
        candidate_rows=rows,
    )
    inside = all(int(store.row_of_id[i]) in allowed for i, _ in hits)
    check("filtered hits all satisfy the keywords", inside, f"{len(hits)} hits")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset", nargs="?", default="flickr1m", choices=sorted(datasets.ALL_DATASETS)
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="skip the float64 ground-truth comparison, which reads every matrix",
    )
    args = parser.parse_args()
    dataset = datasets.ALL_DATASETS[args.dataset]
    store = FeatureStore(dataset).load()
    if not store.size:
        print(f"{dataset.name}: no features; run scripts/build_index.py {dataset.name}")
        return 1

    print(f"{dataset.name}: {store.size} images, {len(store.descriptor_names)} descriptors\n")
    check = Checker()
    rng = np.random.default_rng(1)
    self_query_scores_zero(store, check, rng)
    upload_reproduces_index(store, dataset, check, rng)
    if not args.quick:
        matches_ground_truth(store, check, rng)
    tree_agrees_with_scan(store, check, rng)
    gather_matches_mask(store, dataset, check)
    filter_is_respected(store, check)

    print("\nALL PASS" if not check.failed else f"\n{check.failed} FAILURE(S)")
    return 1 if check.failed else 0


if __name__ == "__main__":
    sys.exit(main())
