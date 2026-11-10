"""Load the MIRFLICKR distribution's own tag and feature archives into the index.

    python scripts/import_mirflickr.py --tags ../tags.zip
    python scripts/import_mirflickr.py --features ../features_edgehistogram.zip --name mp7_ehd_ref
    python scripts/import_mirflickr.py --features ../features_homogeneoustexture.zip --name mp7_htd_ref
    python scripts/import_mirflickr.py --features ../features_gist.zip --name gist --binary

The archives are read in place. Between them they hold four million files of a few
hundred bytes each; unpacking that onto a filesystem costs several GB and far more
time than streaming the zip entries once, and nothing else needs them on disk.

Index mapping: the thumbnails are 1-based (`im1.jpg` … `im1000000.jpg`) and the
archives are 0-based (`0.txt` … `999999.txt`), so image `im<n>` is archive entry
`<n-1>`, in subdirectory `(n-1) // 10000`. That is verified rather than assumed —
our own edge histogram correlates 0.94 (cosine) with the provided one at this
offset and 0.67 at the neighbouring ones, which is the level of a random pair.

Run it after `build_index.py`, which is what defines the row order these matrices
have to line up with. Re-running `build_index.py` replaces the rows and therefore
clears the imported tags, so import them again afterwards; the feature matrices
are keyed to `ids.npy` and are rebuilt by this script too.
"""

import argparse
import re
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from tqdm import tqdm

from app import datasets, db
from app.descriptors import SCALED_DESCRIPTORS
from app.search import save_matrix
from app.vptree import VPTree

SHARD = 10_000  # archive entries per subdirectory
_IMAGE_NUMBER = re.compile(r"im(\d+)\.", re.IGNORECASE)
_ENTRY_NUMBER = re.compile(r"/(\d+)\.(txt|dat)$", re.IGNORECASE)


def image_number(filename):
    """The n of im<n>.jpg, or None."""
    found = _IMAGE_NUMBER.search(filename)
    return int(found.group(1)) if found else None


def archive_index(name):
    """The 0-based image index an archive entry belongs to, or None."""
    found = _ENTRY_NUMBER.search(name)
    return int(found.group(1)) if found else None


def row_lookup(dataset):
    """{archive index: feature-matrix row} for every indexed image.

    `ids.npy` fixes the row order, so the lookup goes through the database to get
    each row's filename and from there the image number.
    """
    ids_path = dataset.features_dir / "ids.npy"
    if not ids_path.exists():
        raise SystemExit(
            f"no index in {dataset.features_dir}; run build_index.py {dataset.name} first"
        )
    ids = np.load(ids_path)
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, filename FROM images WHERE dataset = %s", (dataset.name,)
        )
        names = {row["id"]: row["filename"] for row in cur.fetchall()}

    lookup, missing = {}, 0
    for row, image_id in enumerate(ids.tolist()):
        filename = names.get(image_id)
        number = image_number(filename) if filename else None
        if number is None:
            missing += 1
            continue
        lookup[number - 1] = row
    if missing:
        print(f"warning: {missing} indexed rows have no im<n> filename and are skipped")
    return ids, lookup


# --- tags --------------------------------------------------------------------

def import_tags(dataset, zip_path):
    """Fill the user_tags column from tags/<shard>/<index>.txt."""
    ids, lookup = row_lookup(dataset)
    row_to_id = {row: int(image_id) for row, image_id in enumerate(ids.tolist())}

    collected = {}
    empty = 0
    started = time.perf_counter()
    with zipfile.ZipFile(zip_path) as archive:
        entries = [i for i in archive.infolist() if not i.is_dir()]
        for info in tqdm(entries, desc="tags", unit="file", smoothing=0.05):
            index = archive_index(info.filename)
            if index is None:
                continue
            row = lookup.get(index)
            if row is None:
                continue
            text = archive.read(info).decode("utf-8", "replace")
            tags = [line.strip() for line in text.splitlines() if line.strip()]
            if not tags:
                empty += 1
                continue
            # Deduplicated and lower-cased; to_tsvector('simple') does not stem,
            # so what goes in is what has to be typed to match it.
            collected[row_to_id[row]] = " ".join(
                dict.fromkeys(t.lower() for t in tags)
            )

    elapsed = time.perf_counter() - started
    print(
        f"read {len(collected) + empty} tag files in {elapsed:.0f}s: "
        f"{len(collected)} with tags, {empty} empty"
    )
    if not collected:
        print("nothing to write")
        return

    print("writing to Postgres")
    with db.connect() as conn, conn.cursor() as cur:
        # Same trick as the indexer: maintaining the GIN index across 800k
        # updates costs far more than rebuilding it once.
        cur.execute("DROP INDEX IF EXISTS images_keywords_idx")
        cur.execute(
            "CREATE TEMP TABLE incoming_tags (id bigint PRIMARY KEY, tags text) "
            "ON COMMIT DROP"
        )
        with cur.copy("COPY incoming_tags (id, tags) FROM STDIN") as copy:
            for image_id, text in collected.items():
                copy.write_row((image_id, text))
        cur.execute("ANALYZE incoming_tags")
        cur.execute(
            "UPDATE images SET user_tags = t.tags FROM incoming_tags t "
            "WHERE images.id = t.id AND images.user_tags <> t.tags"
        )
        written = cur.rowcount
        cur.execute("CREATE INDEX images_keywords_idx ON images USING GIN (keywords)")
        cur.execute("ANALYZE images")
        conn.commit()

    total = len(ids)
    print(f"updated {written} rows; {len(collected)}/{total} of the base is tagged "
          f"({len(collected) / total:.1%})")

    # The real tags are the bulk of what fuzzy search matches against, so the
    # vocabulary is only meaningful once they are loaded.
    print("building the fuzzy-matching vocabulary")
    with db.connect() as conn:
        words = db.build_tag_vocab(conn, dataset.name)
    if words:
        print(f"  {words:,} words (in {db.VOCAB_MIN_NDOC}+ images)")


# --- feature matrices --------------------------------------------------------

def read_vector(data, binary):
    """One feature vector from an archive entry."""
    if binary:
        return np.frombuffer(data, dtype="<f4")
    return np.array(data.decode("ascii", "replace").split(), dtype=np.float32)


def import_features(dataset, zip_path, name, binary, build_tree=True):
    """Build features/<dataset>/<name>.npy from a provided feature archive."""
    if name not in dataset.descriptors:
        raise SystemExit(
            f"{name!r} is not listed for {dataset.name}; add it to app/datasets.py"
        )
    ids, lookup = row_lookup(dataset)
    matrix = None
    filled = np.zeros(len(ids), dtype=bool)

    started = time.perf_counter()
    with zipfile.ZipFile(zip_path) as archive:
        entries = [i for i in archive.infolist() if not i.is_dir()]
        for info in tqdm(entries, desc=name, unit="file", smoothing=0.05):
            index = archive_index(info.filename)
            if index is None:
                continue
            row = lookup.get(index)
            if row is None:
                continue
            vector = read_vector(archive.read(info), binary)
            if matrix is None:
                matrix = np.zeros((len(ids), vector.size), dtype=np.float32)
                print(f"\n{name}: {vector.size} dimensions")
            if vector.size != matrix.shape[1]:
                raise SystemExit(
                    f"{info.filename} has {vector.size} values, expected {matrix.shape[1]}"
                )
            matrix[row] = vector
            filled[row] = True

    if matrix is None:
        raise SystemExit("no usable entries found in the archive")
    elapsed = time.perf_counter() - started
    print(f"read {int(filled.sum())}/{len(ids)} vectors in {elapsed:.0f}s")

    short = int(len(ids) - filled.sum())
    if short:
        # Leaving a zero row in would make it a spurious nearest neighbour for
        # anything near the origin, so fill the gaps with the column means.
        print(f"warning: {short} rows had no entry; filling them with the column mean")
        matrix[~filled] = matrix[filled].mean(axis=0)

    features_dir = dataset.features_dir
    save_matrix(features_dir, name, matrix, scaled=name in SCALED_DESCRIPTORS)
    print(f"  {name}.npy  {matrix.shape}")

    tree_path = features_dir / f"{name}.vptree.npz"
    tree_path.unlink(missing_ok=True)
    if build_tree:
        started = time.perf_counter()
        VPTree.build(matrix).save(tree_path)
        print(f"  {name}.vptree.npz  built in {time.perf_counter() - started:.1f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="flickr1m", choices=sorted(datasets.DATASETS))
    parser.add_argument("--tags", metavar="ZIP", help="tags.zip")
    parser.add_argument("--features", metavar="ZIP", help="a features_*.zip")
    parser.add_argument("--name", help="descriptor name to store the features under")
    parser.add_argument(
        "--binary", action="store_true",
        help="entries are raw little-endian float32 rather than one value per line",
    )
    parser.add_argument("--no-vptree", action="store_true")
    args = parser.parse_args()

    dataset = datasets.DATASETS[args.dataset]
    db.init_schema()
    if not args.tags and not args.features:
        parser.error("give --tags or --features")
    if args.features and not args.name:
        parser.error("--features needs --name")

    if args.tags:
        import_tags(dataset, args.tags)
    if args.features:
        import_features(
            dataset, args.features, args.name, args.binary,
            build_tree=not args.no_vptree,
        )


if __name__ == "__main__":
    main()
