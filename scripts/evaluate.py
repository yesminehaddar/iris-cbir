"""Measure precision and recall at k for a labelled dataset."""

import argparse
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app import datasets, db
from app.search import FeatureStore


def labels_of(dataset, ids):
    """{image id: class label} for the indexed images — the ground truth."""
    with db.connect() as conn:
        rows = db.get_images(conn, ids)
    return {i: rows[i]["class_label"] for i in ids if i in rows}


def score(store, labels, names, k):
    """Mean precision@k, recall@k and query time over every image as a query."""
    ids = store.ids.tolist()
    precision, recall, elapsed = [], [], []
    class_sizes = {}
    for label in labels.values():
        class_sizes[label] = class_sizes.get(label, 0) + 1

    for image_id in ids:
        truth = labels.get(image_id)
        if truth is None:
            continue
        vectors = store.vectors_of(image_id)
        started = time.perf_counter()
        # k + 1, because the query image is its own nearest neighbour.
        hits, _ = store.search(vectors, names, k + 1, index="brute")
        elapsed.append((time.perf_counter() - started) * 1000.0)

        retrieved = [i for i, _ in hits if i != image_id][:k]
        relevant = sum(1 for i in retrieved if labels.get(i) == truth)
        precision.append(relevant / max(len(retrieved), 1))
        recall.append(relevant / max(class_sizes[truth] - 1, 1))

    return (
        float(np.mean(precision)) if precision else 0.0,
        float(np.mean(recall)) if recall else 0.0,
        float(np.mean(elapsed)) if elapsed else 0.0,
    )


def evaluate(dataset, k, max_combination):
    """Score every descriptor and every combination of them, best precision last."""
    if not dataset.labelled:
        print(f"{dataset.name} has no class labels; nothing to score against.")
        return
    store = FeatureStore(dataset).load()
    if not store.size:
        print(f"{dataset.name}: no features; run scripts/build_index.py {dataset.name}")
        return

    labels = labels_of(dataset, store.ids.tolist())
    classes = sorted(set(labels.values()))
    print(f"{dataset.name}: {store.size} images, {len(classes)} classes, k={k}")
    print(f"classes: {', '.join(classes)}\n")

    names = store.descriptor_names
    combinations = [
        combo
        for size in range(1, min(max_combination, len(names)) + 1)
        for combo in itertools.combinations(names, size)
    ]

    header = f"{'descriptors':44s} {'P@k':>7s} {'R@k':>7s} {'ms':>7s}"
    print(header)
    print("-" * len(header))
    rows = []
    for combo in combinations:
        p, r, ms = score(store, labels, list(combo), k)
        rows.append((p, r, ms, combo))
        print(f"{' + '.join(combo):44s} {p:7.3f} {r:7.3f} {ms:7.2f}")

    best = max(rows, key=lambda row: row[0])
    print(f"\nbest precision@{k}: {' + '.join(best[3])} at {best[0]:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset", nargs="?", default="objects", choices=sorted(datasets.ALL_DATASETS)
    )
    parser.add_argument("--k", type=int, default=9, help="results scored per query")
    parser.add_argument(
        "--max-combination", type=int, default=3, help="largest descriptor combination"
    )
    args = parser.parse_args()
    evaluate(datasets.ALL_DATASETS[args.dataset], args.k, args.max_combination)


if __name__ == "__main__":
    main()
