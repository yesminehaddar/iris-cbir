"""Memory-mapped feature stores and descriptor similarity search."""

import json
from pathlib import Path

import numpy as np

from app.vptree import VPTree


#: How many extra candidates the scan pre-selects before recomputing distances in
#: float64. The gemv identity is fast but ill-conditioned: on `mp7_htd_ref` it
#: subtracts squared norms of ~19 000 to produce a squared distance of ~4, a
#: cancellation of 4500x that leaves ~1e-3 of float32 slack on the answer — more
#: than the 5.7e-4 that separated the 24th neighbour from the 25th in a measured
#: case. Selecting a wider pool on the approximate distances and then ranking that
#: pool exactly costs a few dozen rows and makes the returned k right.
#:
#: (Centring each matrix at index time would cure the conditioning at the source,
#: since L2 is translation-invariant. This fixes it without rewriting 3 GB.)
REFINE_FACTOR = 4
REFINE_MARGIN = 64


def refine_pool(k, available):
    """Size of the pre-selection pool for a top-k query."""
    return int(min(max(k * REFINE_FACTOR, k + REFINE_MARGIN), available))


#: A column whose deviation over the base is below this is constant, and is left
#: unscaled. Dividing by it would turn float noise into the largest values in the
#: vector: mp7_htd has four such columns (deviations of frequency bands holding
#: one symmetric pair of coefficients, std ~1e-7), and a 1e-9 guard let them
#: through, scaled up to values of ~50 that decided most of every HTD ranking.
MIN_COLUMN_SCALE = 1e-4


def column_scale(matrix):
    """Per-column divisors for a descriptor whose components live on different scales."""
    scale = matrix.std(axis=0)
    scale[scale < MIN_COLUMN_SCALE] = 1.0
    return scale.astype(np.float32)


def save_matrix(features_dir, name, matrix, scaled):
    """Write <name>.npy with its row norms, scaling the columns first if asked.

    Shared by the indexer and the archive importer, so the files the FeatureStore
    reads are always produced the same way. Scales `matrix` in place and returns it.
    """
    features_dir = Path(features_dir)
    if scaled:
        scale = column_scale(matrix)
        matrix /= scale
        np.save(features_dir / f"{name}.scale.npy", scale)
    else:
        (features_dir / f"{name}.scale.npy").unlink(missing_ok=True)
    np.save(features_dir / f"{name}.npy", matrix)
    np.save(
        features_dir / f"{name}.sqnorm.npy",
        np.einsum("ij,ij->i", matrix, matrix).astype(np.float64),
    )
    return matrix


class BruteForceIndex:
    """Exhaustive Euclidean distances over one descriptor matrix."""

    def __init__(self, matrix, sq_norms=None):
        self.matrix = matrix
        # Row norms, so a full scan is a matrix-vector product instead of a
        # 800k x D temporary array of differences. The indexer saves them, since
        # computing them here would mean reading the whole memory map at startup.
        if sq_norms is None:
            sq_norms = np.einsum("ij,ij->i", matrix, matrix)
        self.sq_norms = np.asarray(sq_norms, dtype=np.float64)

    def distances(self, query_vec, rows=None):
        """Distances from query_vec to every row (or just `rows`), in that order."""
        query = np.asarray(query_vec, dtype=self.matrix.dtype)
        if rows is None:
            matrix, sq_norms = self.matrix, self.sq_norms
        else:
            matrix, sq_norms = self.matrix[rows], self.sq_norms[rows]
        sq = sq_norms - 2.0 * (matrix @ query) + float(query @ query)
        return np.sqrt(np.maximum(sq, 0.0))

    def exact_distances(self, query_vec, rows):
        """Honest float64 distances for a handful of rows, straight from the matrix."""
        query = np.asarray(query_vec, dtype=np.float64)
        block = np.asarray(self.matrix[rows], dtype=np.float64)
        return np.linalg.norm(block - query, axis=1)

    def knn(self, query_vec, k, rows=None):
        """Top-k (rows, distances) by increasing distance, exactly."""
        dists = self.distances(query_vec, rows)
        if dists.size == 0:
            return np.zeros(0, dtype=np.int64), dists
        k = min(k, dists.size)
        pool = refine_pool(k, dists.size)
        pre = np.argpartition(dists, pool - 1)[:pool]
        candidates = pre if rows is None else np.asarray(rows)[pre]
        exact = self.exact_distances(query_vec, candidates)
        order = np.argsort(exact, kind="stable")[:k]
        return candidates[order].astype(np.int64), exact[order]


#: Above this share of the base, gathering the candidate rows out of the memory
#: map costs more than scanning the whole matrix and ignoring the rest — a 1-in-3
#: keyword filter touches nearly every page either way.
FULL_SCAN_FRACTION = 0.10


def _minmax(values, where=None):
    """Scale distances to [0, 1]; a flat array becomes all zeros.

    `where` restricts the range to the candidates, so a query scores the same
    whether the candidates were gathered or masked out of a full scan.
    """
    considered = values if where is None else values[where]
    if considered.size == 0:
        return np.zeros_like(values)
    low, high = float(considered.min()), float(considered.max())
    if high - low < 1e-12:
        return np.zeros_like(values)
    return (values - low) / (high - low)


class FeatureStore:
    """Loads features/<descriptor>.npy for one dataset, plus any saved VP-trees."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.features_dir = Path(dataset.features_dir)
        self.ids = np.zeros(0, dtype=np.int64)
        self.row_of_id = np.zeros(0, dtype=np.int64)
        self.indexes = {}
        self.scales = {}
        self.trees = {}
        self.tag_masks = None  # uint32 per row, when the base has derived tags
        self.tag_bits = {}

    # --- loading -------------------------------------------------------------

    def load(self):
        """Open everything this base has on disk; returns self, so it can be chained.

        Nothing is read in full: the matrices are memory-mapped and the trees are
        flat arrays, so a 3 GB index opens instantly. Absence is not an error --
        a base with no `ids.npy` loads as empty and the API reports it as not
        indexed, which is what lets the server start before anything is built.
        """
        ids_path = self.features_dir / "ids.npy"
        if not ids_path.exists():
            return self

        self.ids = np.load(ids_path)
        # Database ids are dense within a dataset, so an array beats a dict here:
        # 800k entries cost a few MB instead of a hundred.
        span = int(self.ids.max()) + 1 if self.ids.size else 1
        self.row_of_id = np.full(span, -1, dtype=np.int64)
        self.row_of_id[self.ids] = np.arange(self.ids.size, dtype=np.int64)

        masks_path = self.features_dir / "tags.npy"
        vocab_path = self.features_dir / "tagvocab.json"
        if masks_path.exists() and vocab_path.exists():
            masks = np.load(masks_path)
            if masks.shape[0] == self.ids.shape[0]:
                self.tag_masks = masks
                vocabulary = json.loads(vocab_path.read_text("utf-8"))
                self.tag_bits = {w: 1 << i for i, w in enumerate(vocabulary)}

        for name in self.dataset.descriptors:
            path = self.features_dir / f"{name}.npy"
            if not path.exists():
                continue
            matrix = np.load(path, mmap_mode="r")
            if matrix.shape[0] != self.ids.shape[0]:
                raise ValueError(
                    f"{path.name} has {matrix.shape[0]} rows but ids.npy has "
                    f"{self.ids.shape[0]}; rebuild the index"
                )
            norms_path = self.features_dir / f"{name}.sqnorm.npy"
            norms = np.load(norms_path) if norms_path.exists() else None
            self.indexes[name] = BruteForceIndex(matrix, norms)

            scale_path = self.features_dir / f"{name}.scale.npy"
            if scale_path.exists():
                self.scales[name] = np.load(scale_path)

            tree_path = self.features_dir / f"{name}.vptree.npz"
            if tree_path.exists():
                self.trees[name] = VPTree.load(tree_path, matrix)
        return self

    # --- shape ---------------------------------------------------------------

    @property
    def descriptor_names(self):
        """The base's descriptors that are actually on disk, in registry order."""
        return [n for n in self.dataset.descriptors if n in self.indexes]

    @property
    def size(self):
        """How many images are indexed; 0 when nothing has been built."""
        return int(self.ids.size)

    def dimension(self, name):
        """One descriptor's vector length, or 0 if it is not loaded."""
        index = self.indexes.get(name)
        return int(index.matrix.shape[1]) if index is not None else 0

    def has_tree(self, name):
        """Whether a VP-tree was saved for this descriptor."""
        return name in self.trees

    # --- queries -------------------------------------------------------------

    def apply_scale(self, vectors):
        """Scale a freshly computed query vector the way the stored matrix was."""
        return {
            name: (vec / self.scales[name] if name in self.scales else vec)
            for name, vec in vectors.items()
        }

    def vectors_of(self, image_id):
        """Stored vectors of an indexed image, usable as a query."""
        row = self._row(image_id)
        if row is None:
            return None
        return {name: np.asarray(idx.matrix[row]) for name, idx in self.indexes.items()}

    def _row(self, image_id):
        image_id = int(image_id)
        if image_id < 0 or image_id >= self.row_of_id.size:
            return None
        row = int(self.row_of_id[image_id])
        return None if row < 0 else row

    def rows_for_words(self, text):
        """Rows whose derived tags contain every word, or None if not answerable here.

        This is the keyword filter for a base with a closed tag vocabulary. Twenty-
        five words fit in one uint32 per image, so narrowing the whole base is a
        masked comparison over 3 MB — no ceiling, and no asking the database for
        several hundred thousand ids to carry back across the wire. A word outside
        the vocabulary returns None, and the caller falls back to full-text search.
        """
        if self.tag_masks is None:
            return None
        wanted = 0
        for word in text.lower().split():
            bit = self.tag_bits.get(word)
            if bit is None:
                return None
            wanted |= bit
        if not wanted:
            return None
        wanted = np.uint32(wanted)
        return np.flatnonzero((self.tag_masks & wanted) == wanted)

    def rows_of(self, image_ids):
        """Rows for an array of ids, dropping any that are not in this dataset."""
        wanted = np.asarray(image_ids, dtype=np.int64)
        wanted = wanted[(wanted >= 0) & (wanted < self.row_of_id.size)]
        rows = self.row_of_id[wanted]
        return rows[rows >= 0]

    def search(
        self,
        query_vecs,
        descriptor_names,
        k,
        index="auto",
        candidate_rows=None,
    ):
        """Top-k (id, score) by increasing combined distance.

        Each requested descriptor gives a distance array, min-max normalised, then
        averaged across descriptors. A keyword or visual-property filter arrives
        as `candidate_rows`, already resolved to matrix rows by `rows_for_words`
        or `rows_of`. Returns the hits and a dict describing how it was done.

        `index` is "vptree" to ask for the metric tree and anything else -- "auto",
        "brute" -- for the scan. They are the same request: the scan is what wins
        on this index, so it is the default rather than a fallback. Asking for the
        tree where it cannot answer (several descriptors, or a filtered candidate
        set) falls back to the scan and says so in the returned note.
        """
        names = [n for n in descriptor_names if n in self.indexes]
        if not names or self.ids.size == 0 or k <= 0:
            return [], {"method": "none", "scanned": 0, "candidates": 0}

        rows = candidate_rows
        if rows is not None and rows.size == 0:
            return [], {"method": "none", "scanned": 0, "candidates": 0}

        single = len(names) == 1
        # "auto" is the scan, deliberately. The tree returns the same neighbours
        # and prunes well in low dimension, but on this index it never wins: the
        # scan is one BLAS pass running at memory bandwidth, and at 60-80
        # dimensions the distances concentrate enough that the tree ends up
        # touching most of the base anyway. See the measurements in docs/DESIGN.md.
        use_tree = (
            index == "vptree" and single and rows is None and self.has_tree(names[0])
        )
        note = None
        if index == "vptree" and not use_tree:
            note = (
                "the tree answers one metric over the whole base, so this query "
                "fell back to the scan"
            )

        if use_tree:
            tree = self.trees[names[0]]
            picked, dists = tree.knn(query_vecs[names[0]], k)
            hits = [
                (int(self.ids[row]), float(dist))
                for row, dist in zip(picked.tolist(), dists.tolist())
            ]
            return hits, {
                "method": "vptree",
                "scanned": int(tree.last_visited),
                "candidates": self.size,
            }

        # Either gather the candidates or scan everything and mask, whichever
        # reads less: `keep` is set when we chose to mask.
        keep, candidates = None, self.size if rows is None else int(rows.size)
        if rows is not None and rows.size > FULL_SCAN_FRACTION * self.size:
            keep = np.zeros(self.size, dtype=bool)
            keep[rows] = True
            rows = None

        if single:
            dists = self.indexes[names[0]].distances(query_vecs[names[0]], rows)
        else:
            dists = np.zeros(self.size if rows is None else rows.size, dtype=np.float64)
            for name in names:
                dists += _minmax(
                    self.indexes[name].distances(query_vecs[name], rows), keep
                )
            dists /= len(names)
        if keep is not None:
            dists = np.where(keep, dists, np.inf)

        take = min(k, candidates)
        if single:
            # Pre-select a wider pool on the approximate distances, then rank that
            # pool in float64. See refine_pool: the gemv identity is too
            # ill-conditioned to settle the boundary of the top k on its own.
            pool = refine_pool(take, candidates)
            pre = np.argpartition(dists, pool - 1)[:pool]
            refined = pre if rows is None else rows[pre]
            scores = self.indexes[names[0]].exact_distances(
                query_vecs[names[0]], refined
            )
            order = np.argsort(scores, kind="stable")[:take]
            picked, scores = refined[order], scores[order]
        else:
            best = np.argpartition(dists, take - 1)[:take]
            picked, scores = (best if rows is None else rows[best]), dists[best]
            order = np.argsort(scores, kind="stable")
            picked, scores = picked[order], scores[order]

        hits = [
            (int(self.ids[row]), float(score))
            for row, score in zip(picked.tolist(), scores.tolist())
        ]
        stats = {
            "method": "brute-force",
            # What it actually compared: the whole base when it scanned, the
            # candidate set when it gathered.
            "scanned": self.size if rows is None else candidates,
            "candidates": candidates,
        }
        if note:
            stats["note"] = note
        return hits, stats


class StoreSet:
    """The loaded feature stores, one per dataset."""

    def __init__(self, datasets):
        self.stores = {ds.name: FeatureStore(ds) for ds in datasets}

    def load(self):
        """Load every base's store. Called once, at server startup."""
        for store in self.stores.values():
            store.load()
        return self

    def __getitem__(self, name):
        return self.stores[name]
