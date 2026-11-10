"""Offline indexer: compute descriptors, fill the images table, save the index.

    python scripts/build_index.py flickr1m
    python scripts/build_index.py flickr1m --limit 200000 --workers 10
    python scripts/build_index.py flickr1m --no-vptree
    python scripts/build_index.py flickr1m --trees-only --leaf-size 512
    python scripts/build_index.py flickr1m --restore-rows   # new machine, empty DB
    python scripts/build_index.py objects          # the unserved labelled base

Re-runnable per dataset: only the rows of the dataset named are replaced, so an
unserved base built for evaluation survives a Flickr rebuild untouched.

The descriptor pass runs first and the database rows are written afterwards, in a
single COPY that already carries the derived tags. Doing it the other way round
would mean a 800k-row UPDATE through the full-text index.

What it writes into features/<dataset>/:

    ids.npy                     database id of each row
    tags.npy                    uint32 bitmask of each row's derived tags
    tagvocab.json               which bit is which word
    <descriptor>.npy            the N x D feature matrix
    <descriptor>.sqnorm.npy     row norms, so the server's scan is one gemv
    <descriptor>.scale.npy      column divisors, for descriptors that need them
    <descriptor>.vptree.npz     the metric tree, when it was built
"""

import argparse
import itertools
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from tqdm import tqdm

from app import datasets, db
from app.descriptors import (
    DESCRIPTORS,
    IMPORTED_DESCRIPTORS,
    SCALED_DESCRIPTORS,
    ImageContext,
    load_image,
)
from app.visualtags import (
    BIT,
    DENSITY,
    EDGE_WORDS,
    HUE_NAMES,
    LIGHTNESS,
    SATURATION,
    VOCABULARY,
)
from app.search import FeatureStore, save_matrix
from app.vptree import LEAF_SIZE, VPTree

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"}
CHUNK = 2000  # images per task handed to a worker
_NUMBER = re.compile(r"(\d+)")


def _natural_key(path):
    """Sort im2.jpg before im10.jpg, and shard 2 before shard 10."""
    return [int(p) if p.isdigit() else p for p in _NUMBER.split(path.as_posix())]


def list_images(images_dir):
    """(relative path, class label) for every image under images_dir/<class>/<file>."""
    paths = [
        p
        for p in images_dir.rglob("*")
        if p.suffix.lower() in EXTENSIONS and p.is_file()
    ]
    paths.sort(key=_natural_key)
    items = []
    for path in paths:
        rel = path.relative_to(images_dir)
        items.append((rel, rel.parts[0] if len(rel.parts) > 1 else "unclassified"))
    return items


# --- visual tags -------------------------------------------------------------

# Columns of the per-image measurement row.
VAL, SAT, HUE, DENS, DIR = range(5)
#: Scales cut at their terciles, so each of their three words selects a third of
#: the base. The categorical ones (hue, edge direction) are left as they fall.
TERCILED = ((VAL, LIGHTNESS), (SAT, SATURATION), (DENS, DENSITY))


def visual_measures(ctx, vectors):
    """Scalars describing an image, for the parent process to turn into words.

    The thumbnail base ships without the Flickr tag files, so there is no human
    text to index and the keyword half of the engine would have nothing to work
    on. Each image is therefore described by what the descriptors already looked
    at: mean lightness and saturation, dominant hue, edge density and dominant
    edge direction.

    These come back as numbers rather than words so that the thresholds can be
    picked from the whole base at once. A fixed cut at V < 0.3 labelled 79% of
    this base "midtone", which is no use as a search term; terciles give three
    words that each select a third of it.
    """
    hue, sat, val = ctx.hsv[..., 0], ctx.hsv[..., 1], ctx.hsv[..., 2]

    # Dominant hue over the pixels colourful enough to have one; -1 when almost
    # none are, which is how a greyscale image ends up with no hue word at all.
    hue_bin = -1
    coloured = (sat > 0.2) & (val > 0.15)
    if float(coloured.mean()) > 0.08:
        bins = np.minimum((hue[coloured] * len(HUE_NAMES)).astype(np.intp), len(HUE_NAMES) - 1)
        hue_bin = int(np.bincount(bins, minlength=len(HUE_NAMES)).argmax())

    density, direction = 0.0, -1
    ehd = vectors.get("mp7_ehd")
    if ehd is not None and ehd.size == 80:
        by_type = ehd.reshape(16, 5).sum(axis=0) / 16.0
        density = float(by_type.sum())
        direction = int(by_type.argmax())

    return (float(val.mean()), float(sat.mean()), float(hue_bin), density, float(direction))


def tags_from_measures(measures):
    """Turn the measurement rows into (tag strings, uint32 bitmasks).

    The bitmask is the index the combined keyword + visual search actually uses:
    25 words in one uint32 per image means filtering the whole base is a masked
    comparison over 3 MB, instead of asking Postgres for several hundred thousand
    ids and capping them.
    """
    measures = np.asarray(measures, dtype=np.float64)
    count = measures.shape[0]
    masks = np.zeros(count, dtype=np.uint32)
    buckets = {}

    for column, names in TERCILED:
        values = measures[:, column]
        edges = np.quantile(values, [1 / 3, 2 / 3])
        bucket = np.searchsorted(edges, values, side="right")
        buckets[column] = bucket
        for index, word in enumerate(names):
            masks[bucket == index] |= np.uint32(BIT[word])

    hue = measures[:, HUE].astype(np.intp)
    for index, word in enumerate(HUE_NAMES):
        masks[hue == index] |= np.uint32(BIT[word])
    direction = measures[:, DIR].astype(np.intp)
    for index, word in enumerate(EDGE_WORDS):
        masks[direction == index] |= np.uint32(BIT[word])

    # The text form is what goes into Postgres, for the keyword-only search and
    # so the UI can show why an image came back.
    light, saturation, dense = (buckets[column] for column, _ in TERCILED)
    texts = []
    for i in range(count):
        words = [LIGHTNESS[light[i]], SATURATION[saturation[i]], DENSITY[dense[i]]]
        if hue[i] >= 0:
            words.append(HUE_NAMES[hue[i]])
        if direction[i] >= 0:
            words.append(EDGE_WORDS[direction[i]])
        texts.append(" ".join(dict.fromkeys(words)))
    return texts, masks


# --- per-worker descriptor computation ---------------------------------------

_WORKER = {}


def _init_worker(images_dir, names):
    _WORKER["dir"] = Path(images_dir)
    _WORKER["names"] = list(names)


def _describe_chunk(chunk):
    """(position, {descriptor: vector} | None, measurements or error) per image."""
    base, names = _WORKER["dir"], _WORKER["names"]
    out = []
    for position, rel in chunk:
        try:
            ctx = ImageContext(load_image(base / rel))
            vectors = {name: DESCRIPTORS[name](ctx) for name in names}
            out.append((position, vectors, visual_measures(ctx, vectors)))
        except OSError as exc:  # unreadable or truncated file: skip just this one
            out.append((position, None, str(exc)))
    return out


def computed_descriptors(dataset):
    """The dataset's descriptors this script can actually compute from pixels.

    A dataset also lists the ones loaded from the distribution's archives by
    import_mirflickr.py (`mp7_ehd_ref`, `mp7_htd_ref`, `gist`). Those have no
    function in DESCRIPTORS -- they were computed on full-resolution originals
    this base does not ship -- so asking for them here would raise KeyError on
    every single image.
    """
    names = [n for n in dataset.descriptors if n not in IMPORTED_DESCRIPTORS]
    unknown = [n for n in names if n not in DESCRIPTORS]
    if unknown:
        raise SystemExit(
            f"{dataset.name} lists descriptor(s) with no implementation: "
            f"{', '.join(unknown)}. Register them in app/descriptors.py, or mark "
            "them as imported."
        )
    return names


def describe_all(dataset, items, workers, names):
    """Run the descriptor pass over every image; returns matrices, measures, mask."""
    columns = {}
    measures = np.zeros((len(items), 5), dtype=np.float64)
    keep = np.zeros(len(items), dtype=bool)
    chunks = [
        [(i, items[i][0]) for i in range(start, min(start + CHUNK, len(items)))]
        for start in range(0, len(items), CHUNK)
    ]

    skipped = 0
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(dataset.images_dir), names),
    ) as pool:
        progress = tqdm(total=len(items), desc="descriptors", unit="img", smoothing=0.05)
        for results in pool.map(_describe_chunk, chunks):
            for position, vectors, note in results:
                if vectors is None:
                    skipped += 1
                    if skipped <= 10:
                        tqdm.write(f"skipped {items[position][0]}: {note}")
                    continue
                for name, vector in vectors.items():
                    if name not in columns:
                        columns[name] = np.zeros((len(items), vector.size), np.float32)
                    columns[name][position] = vector
                measures[position] = note
                keep[position] = True
            progress.update(len(results))
        progress.close()

    elapsed = time.perf_counter() - started
    kept = int(keep.sum())
    print(f"described {kept} images in {elapsed:.0f}s ({kept / max(elapsed, 1e-9):.0f}/s)")
    if skipped:
        print(f"{skipped} unreadable file(s) skipped")
    return columns, measures, keep


# --- database ----------------------------------------------------------------

def replace_rows(conn, dataset, records, ids=None):
    """Replace this dataset's rows in one COPY; returns the new ids, in order.

    `ids`, when given, are written as they are instead of being drawn from the
    sequence -- that is how `restore_rows` puts back rows an index is keyed to.

    The full-text index is dropped around the load: rebuilding a GIN index once
    is far cheaper than maintaining it across 800k inserts.
    """
    with conn.cursor() as cur:
        cur.execute("DROP INDEX IF EXISTS images_keywords_idx")
        cur.execute("DELETE FROM images WHERE dataset = %s", (dataset.name,))
        columns = "dataset, url, filename, class_label, tags"
        if ids is None:
            # Keep ids dense across rebuilds: the feature store's id -> row map is
            # an array indexed by id, so letting the sequence drift by 800k every
            # run would cost memory for nothing.
            cur.execute("SELECT coalesce(max(id), 0) + 1 AS next FROM images")
            prefixes = itertools.repeat(())
        else:
            cur.execute(
                "SELECT greatest(coalesce(max(id), 0), %s) + 1 AS next FROM images",
                (int(max(ids, default=0)),),
            )
            columns = "id, " + columns
            prefixes = ((int(i),) for i in ids)
        cur.execute(
            "SELECT setval(pg_get_serial_sequence('images', 'id'), %s, false)",
            (cur.fetchone()["next"],),
        )
        with cur.copy(f"COPY images ({columns}) FROM STDIN") as copy:
            for prefix, (rel, class_label, tags) in zip(prefixes, records):
                copy.write_row(
                    prefix
                    + (
                        dataset.name,
                        f"{dataset.url_prefix}/{rel.as_posix()}",
                        rel.name,
                        class_label,
                        tags,
                    )
                )
        cur.execute("CREATE INDEX images_keywords_idx ON images USING GIN (keywords)")
        cur.execute("ANALYZE images")
        cur.execute(
            "SELECT id FROM images WHERE dataset = %s ORDER BY id", (dataset.name,)
        )
        ids = [row["id"] for row in cur.fetchall()]
    conn.commit()
    return ids


# --- main --------------------------------------------------------------------

def save_tree(dataset, name, matrix, leaf_size=None):
    """Build and write one descriptor's VP-tree, replacing any tree already there."""
    leaf_size = LEAF_SIZE if leaf_size is None else leaf_size
    path = dataset.features_dir / f"{name}.vptree.npz"
    path.unlink(missing_ok=True)
    started = time.perf_counter()
    VPTree.build(matrix, leaf_size).save(path)
    print(f"  {name}.vptree.npz  leaf={leaf_size} in {time.perf_counter() - started:.1f}s")


def rebuild_trees(dataset, leaf_size):
    """Rebuild only the metric trees, reusing the feature matrices on disk.

    Tuning the leaf size is a question of measurement, and re-running the
    descriptor pass over 800k images to answer it would be absurd.
    """
    features_dir = dataset.features_dir
    if not (features_dir / "ids.npy").exists():
        print(f"no index in {features_dir}; run the full build first")
        return
    for name in dataset.descriptors:
        path = features_dir / f"{name}.npy"
        if path.exists():
            save_tree(dataset, name, np.load(path), leaf_size)
    print(f"Done: trees rebuilt in {features_dir}")


def tags_from_masks(masks, vocabulary):
    """The tag strings `tags_from_measures` wrote, recovered from the saved bitmasks.

    Each facet has at most one word set, and they are joined in the same order.
    There are only a few hundred distinct masks, so each is spelled out once.
    """
    bit = {word: 1 << i for i, word in enumerate(vocabulary)}
    order = list(dict.fromkeys(LIGHTNESS + SATURATION + DENSITY + HUE_NAMES + EDGE_WORDS))
    distinct, inverse = np.unique(masks, return_inverse=True)
    spelled = [
        " ".join(w for w in order if w in bit and int(mask) & bit[w])
        for mask in distinct
    ]
    return [spelled[i] for i in inverse.ravel()]


def _check_alignment(store, items, samples=20):
    """Recompute a descriptor for a spread of rows; True if every one matches.

    The rows are matched to files by position in the sorted listing, so a base
    whose files changed since the index was built would attach every row to the
    wrong image. This is what proves that it did not.
    """
    name = next(
        (n for n in store.descriptor_names if n in DESCRIPTORS), None
    )
    if name is None:
        return True
    index = store.indexes[name]
    for row in np.linspace(0, store.size - 1, samples).astype(int):
        ctx = ImageContext(load_image(store.dataset.images_dir / items[row][0]))
        vector = DESCRIPTORS[name](ctx)
        if name in store.scales:
            vector = vector / store.scales[name]
        if not np.allclose(vector, index.matrix[row], atol=1e-4):
            print(f"row {row} ({items[row][0]}) does not match its indexed {name}")
            return False
    return True


def restore_rows(dataset):
    """Recreate this base's database rows from the index already on disk.

    The rows live in Postgres, in a Docker volume, and the index in
    data/features/, so a project copied to another machine arrives with the
    index and without the rows: every image search then 404s and every text
    search is empty. Rebuilding would redo ten minutes of descriptors to get
    back what is already known -- `ids.npy` fixes each row's id, the sorted file
    listing its file, and `tags.npy` its visual words. The ids are kept, so the
    imported descriptors stay valid; only the tags have to be imported again.
    """
    store = FeatureStore(dataset).load()
    if not store.size:
        raise SystemExit(f"no index in {dataset.features_dir}; run the full build")
    print(f"scanning {dataset.images_dir}")
    items = list_images(dataset.images_dir)
    if len(items) != store.size:
        raise SystemExit(
            f"{len(items)} images on disk but {store.size} rows in the index, so "
            "rows cannot be matched to files; rebuild with: "
            f"python scripts/build_index.py {dataset.name}"
        )
    if not _check_alignment(store, items):
        raise SystemExit(
            "the images on disk are not the ones this index was built from; "
            f"rebuild with: python scripts/build_index.py {dataset.name}"
        )

    vocab_path = dataset.features_dir / "tagvocab.json"
    if store.tag_masks is not None:
        vocabulary = json.loads(vocab_path.read_text("utf-8"))
        tags = tags_from_masks(store.tag_masks, vocabulary)
    else:
        tags = [""] * store.size

    db.init_schema()
    print(f"writing {store.size} rows")
    with db.connect() as conn:
        ids = replace_rows(
            conn,
            dataset,
            [(rel, label, tag) for (rel, label), tag in zip(items, tags)],
            ids=store.ids.tolist(),
        )
        words = db.build_tag_vocab(conn, dataset.name)
        if words:
            print(f"fuzzy vocabulary: {words} words")
    if not np.array_equal(ids, store.ids):
        raise RuntimeError("restored ids do not match ids.npy")
    print(f"Done: {len(ids)} rows restored for {dataset.name}")
    if dataset is datasets.FLICKR:
        print(
            "The real Flickr tags are not in the index; load them again with:\n"
            "    python scripts/import_mirflickr.py --tags ../tags.zip"
        )


def _drop_imported(dataset):
    """Delete the imported descriptors' files after a rebuild; returns their names.

    A rebuild replaces the rows and renumbers the ids the feature matrices are
    keyed to, which leaves any previously imported matrix describing the wrong
    images. The row *count* usually still matches, so the feature store's shape
    check would not catch it and the base would answer confidently and wrongly.
    Deleting them turns a silent error into an obvious absence.
    """
    removed = []
    for name in dataset.descriptors:
        if name not in IMPORTED_DESCRIPTORS:
            continue
        found = False
        for suffix in (".npy", ".sqnorm.npy", ".scale.npy", ".vptree.npz"):
            path = dataset.features_dir / f"{name}{suffix}"
            if path.exists():
                path.unlink()
                found = True
        if found:
            removed.append(name)
    return removed


def build(dataset, limit=None, workers=None, build_trees=True, leaf_size=None):
    """Index one base from scratch: describe, write rows, save matrices and trees.

    The order matters. Descriptors are computed first and the database rows are
    written afterwards in a single COPY that already carries the derived tags,
    because the other way round would mean an 800k-row UPDATE through the
    full-text index. The row ids that COPY returns are what every saved matrix is
    keyed to, which is why anything imported earlier is dropped at the end.
    """
    features_dir = dataset.features_dir
    features_dir.mkdir(parents=True, exist_ok=True)

    print(f"scanning {dataset.images_dir}")
    items = list_images(dataset.images_dir)
    if not items:
        print(f"No images found under {dataset.images_dir}")
        if dataset.name == "objects":
            print("Tip: python scripts/make_dataset.py generates it.")
        return
    if limit:
        items = items[:limit]
    names = computed_descriptors(dataset)
    print(f"{len(items)} images, descriptors: {', '.join(names)}")

    columns, measures, keep = describe_all(dataset, items, workers, names)
    kept = int(keep.sum())
    if not kept:
        print("nothing to index")
        return

    if dataset.derived_tags:
        tags, masks = tags_from_measures(measures[keep])
        np.save(features_dir / "tags.npy", masks)
        (features_dir / "tagvocab.json").write_text(json.dumps(list(VOCABULARY)), "utf-8")
    else:
        # This base has words of its own in its filenames; see Dataset.derived_tags.
        tags = [""] * kept
        (features_dir / "tags.npy").unlink(missing_ok=True)
        (features_dir / "tagvocab.json").unlink(missing_ok=True)

    db.init_schema()
    kept_items = [item for item, wanted in zip(items, keep) if wanted]
    records = [
        (rel, class_label, tag)
        for (rel, class_label), tag in zip(kept_items, tags)
    ]
    print(f"writing {len(records)} rows")
    with db.connect() as conn:
        ids = replace_rows(conn, dataset, records)
        # The vocabulary describes whichever words are in the index, so it has to
        # be rebuilt whenever the rows change or fuzzy search would expand a
        # typed word to one that is no longer there.
        words = db.build_tag_vocab(conn, dataset.name)
        if words:
            print(f"fuzzy vocabulary: {words} words")
    if len(ids) != kept:
        raise RuntimeError(f"inserted {len(ids)} rows for {kept} described images")
    np.save(features_dir / "ids.npy", np.asarray(ids, dtype=np.int64))

    for name in dataset.descriptors:
        if name not in columns:
            continue
        matrix = np.ascontiguousarray(columns[name][keep])
        save_matrix(features_dir, name, matrix, scaled=name in SCALED_DESCRIPTORS)
        print(f"  {name}.npy  {matrix.shape}")

        if build_trees:
            save_tree(dataset, name, matrix, leaf_size)
        else:
            (features_dir / f"{name}.vptree.npz").unlink(missing_ok=True)

    stale = _drop_imported(dataset)
    print(f"Done: {kept} images indexed in {features_dir}")
    if stale:
        print(
            f"\nRemoved {', '.join(stale)}: those come from the distribution's\n"
            "archives and were keyed to the row order this rebuild just replaced.\n"
            "Load them again with:\n"
            "    python scripts/import_mirflickr.py --tags ../tags.zip\n"
            "    python scripts/import_mirflickr.py --features ../features_edgehistogram.zip --name mp7_ehd_ref\n"
            "    python scripts/import_mirflickr.py --features ../features_homogeneoustexture.zip --name mp7_htd_ref\n"
            "    python scripts/import_mirflickr.py --features ../features_gist.zip --name gist --binary\n"
            "(or just run: python setup.py)"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset", nargs="?", default="flickr1m", choices=sorted(datasets.ALL_DATASETS)
    )
    parser.add_argument("--limit", type=int, help="index only the first N images")
    parser.add_argument("--workers", type=int, help="worker processes (default: all cores)")
    parser.add_argument(
        "--no-vptree", action="store_true", help="skip building the metric trees"
    )
    parser.add_argument(
        "--trees-only",
        action="store_true",
        help="rebuild just the metric trees from the matrices already on disk",
    )
    parser.add_argument(
        "--leaf-size", type=int, help=f"vp-tree leaf size (default {LEAF_SIZE})"
    )
    parser.add_argument(
        "--restore-rows",
        action="store_true",
        help="recreate the database rows from the index on disk, without "
        "recomputing descriptors (after moving the project to another machine)",
    )
    args = parser.parse_args()
    dataset = datasets.ALL_DATASETS[args.dataset]
    if args.restore_rows:
        restore_rows(dataset)
        return
    if args.trees_only:
        rebuild_trees(dataset, args.leaf_size)
        return
    build(
        dataset,
        limit=args.limit,
        workers=args.workers,
        build_trees=not args.no_vptree,
        leaf_size=args.leaf_size,
    )


if __name__ == "__main__":
    main()
