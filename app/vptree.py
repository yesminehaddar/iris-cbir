"""Flat-array vantage-point tree for exact Euclidean k-nearest-neighbour search."""

import heapq
import math

import numpy as np

LEAF_SIZE = 512


class VPTree:
    """Exact k-nearest-neighbour index under the Euclidean distance."""

    def __init__(self, matrix, vantage, radius, left, right, start, end, order):
        self.matrix = matrix
        self.vantage = vantage  # row used as vantage point, -1 for a leaf
        self.radius = radius
        self.left = left
        self.right = right
        self.start = start  # leaves only: slice into `order`
        self.end = end
        self.order = order
        self.last_visited = 0  # points actually compared by the last knn(), for the UI

    # --- construction --------------------------------------------------------

    @classmethod
    def build(cls, matrix, leaf_size=None):
        """Build the tree over `matrix`, splitting until segments fit a leaf.

        Iterative and vectorised on purpose: one numpy distance pass per node, and
        an explicit `todo` stack rather than recursion. Doing it per point in
        Python would not finish on 800k rows, and the tree is deep enough that
        recursion would be uncomfortable.
        """
        leaf_size = LEAF_SIZE if leaf_size is None else leaf_size
        n = matrix.shape[0]
        order = np.arange(n, dtype=np.int64)
        vantage, radius = [], []
        left, right, start, end = [], [], [], []

        def add_node():
            vantage.append(-1)
            radius.append(0.0)
            left.append(-1)
            right.append(-1)
            start.append(0)
            end.append(0)
            return len(vantage) - 1

        todo = [(add_node(), 0, n)]
        while todo:
            node, low, high = todo.pop()
            if high - low <= leaf_size:
                start[node], end[node] = low, high
                continue

            # The first point of the segment is the vantage point. The base is
            # already in an arbitrary order, so this is as good as a random pick
            # and costs nothing.
            segment = order[low:high]
            pivot = segment[0]
            rest = segment[1:]
            dists = np.linalg.norm(matrix[rest] - matrix[pivot], axis=1)

            middle = dists.size // 2
            split = np.argpartition(dists, middle)
            order[low + 1 : high] = rest[split]
            vantage[node] = int(pivot)
            radius[node] = float(dists[split[middle]])

            boundary = low + 1 + middle + 1  # inner side keeps the median point
            left[node] = add_node()
            right[node] = add_node()
            # Push the far side first so the near side is expanded first.
            todo.append((right[node], boundary, high))
            todo.append((left[node], low + 1, boundary))

        return cls(
            matrix,
            np.asarray(vantage, dtype=np.int64),
            np.asarray(radius, dtype=np.float64),
            np.asarray(left, dtype=np.int64),
            np.asarray(right, dtype=np.int64),
            np.asarray(start, dtype=np.int64),
            np.asarray(end, dtype=np.int64),
            order,
        )

    # --- persistence ---------------------------------------------------------

    def save(self, path):
        """Write the tree to one .npz. It is flat arrays, so this is the whole of it."""
        np.savez(
            path,
            vantage=self.vantage,
            radius=self.radius,
            left=self.left,
            right=self.right,
            start=self.start,
            end=self.end,
            order=self.order,
        )

    @classmethod
    def load(cls, path, matrix):
        """Read a saved tree back, bound to `matrix`.

        The tree stores row numbers, not coordinates, so it is only meaningful
        against the matrix it was built from -- hence the row-count check, which
        catches a tree left behind by an earlier build of a different size.
        """
        with np.load(path) as data:
            tree = cls(
                matrix,
                data["vantage"],
                data["radius"],
                data["left"],
                data["right"],
                data["start"],
                data["end"],
                data["order"],
            )
        if tree.order.shape[0] != matrix.shape[0]:
            raise ValueError("vp-tree was built for a different number of rows")
        return tree

    # --- query ---------------------------------------------------------------

    def knn(self, query, k):
        """Exact k nearest rows as (rows, distances), nearest first."""
        if k <= 0 or self.matrix.shape[0] == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)

        query = np.asarray(query, dtype=self.matrix.dtype)
        heap = []  # max-heap on distance, holding at most k entries
        tau = math.inf
        visited = 0
        todo = [0]
        while todo:
            node = todo.pop()
            vantage = int(self.vantage[node])

            if vantage < 0:  # leaf: scan what is left of it
                low, high = int(self.start[node]), int(self.end[node])
                if high <= low:
                    continue
                rows = self.order[low:high]
                dists = np.linalg.norm(self.matrix[rows] - query, axis=1)
                visited += rows.size
                if len(heap) >= k:
                    keep = dists < tau
                    if not keep.any():
                        continue
                    rows, dists = rows[keep], dists[keep]
                # Only this leaf's own k best can ever enter the answer, so the
                # rest never reach the heap: pushing a whole leaf one point at a
                # time in Python costs more than the distances did.
                if dists.size > k:
                    pick = np.argpartition(dists, k - 1)[:k]
                    rows, dists = rows[pick], dists[pick]
                for dist, row in zip(dists.tolist(), rows.tolist()):
                    if len(heap) < k:
                        heapq.heappush(heap, (-dist, row))
                    elif dist < -heap[0][0]:
                        heapq.heapreplace(heap, (-dist, row))
                if len(heap) >= k:
                    tau = -heap[0][0]
                continue

            # The vantage point is a data point too.
            dist = float(np.linalg.norm(self.matrix[vantage] - query))
            visited += 1
            if len(heap) < k:
                heapq.heappush(heap, (-dist, vantage))
                if len(heap) >= k:
                    tau = -heap[0][0]
            elif dist < tau:
                heapq.heapreplace(heap, (-dist, vantage))
                tau = -heap[0][0]

            # Prune with the triangle inequality: nothing inside the ball can be
            # closer than dist - radius, nothing outside it closer than radius - dist.
            radius = self.radius[node]
            inner, outer = int(self.left[node]), int(self.right[node])
            worth_inner = dist - tau <= radius
            worth_outer = dist + tau >= radius
            # Push the far side first, so the near side is expanded first and tau
            # tightens before the far side is even considered.
            if dist < radius:
                if worth_outer:
                    todo.append(outer)
                if worth_inner:
                    todo.append(inner)
            else:
                if worth_inner:
                    todo.append(inner)
                if worth_outer:
                    todo.append(outer)

        self.last_visited = visited
        ordered = sorted((-d, r) for d, r in heap)
        if not ordered:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float64)
        dists, rows = zip(*ordered)
        return np.asarray(rows, dtype=np.int64), np.asarray(dists, dtype=np.float64)
