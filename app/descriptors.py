"""Descriptor implementations and registry used by indexing and search."""

import numpy as np
from skimage.feature import hog as sk_hog
from skimage.feature import local_binary_pattern

from app.imagebase import ImageContext, load_image, load_image_bytes  # noqa: F401
from app.mpeg7 import MPEG7, MPEG7_META, MPEG7_SCALED

def color_hist(ctx, bins=8):
    """HSV histogram, 3 channels x `bins`, each channel normalised to sum 1."""
    parts = []
    for channel in range(3):
        counts, _ = np.histogram(ctx.hsv[:, :, channel], bins=bins, range=(0.0, 1.0))
        counts = counts.astype(np.float32)
        total = counts.sum()
        parts.append(counts / total if total else counts)
    return np.concatenate(parts).astype(np.float32)


def lbp(ctx, points=8, radius=1):
    """Histogram of uniform local binary patterns (points + 2 bins)."""
    gray8 = (ctx.gray_work * 255).astype(np.uint8)  # LBP wants integer pixels
    codes = local_binary_pattern(gray8, points, radius, method="uniform")
    n_bins = points + 2
    counts, _ = np.histogram(codes, bins=n_bins, range=(0, n_bins))
    counts = counts.astype(np.float32)
    total = counts.sum()
    return (counts / total if total else counts).astype(np.float32)


def hog(ctx):
    """HOG on a 128x128 grayscale resize."""
    vector = sk_hog(
        ctx.gray_work,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return vector.astype(np.float32)


DESCRIPTORS = {
    "color_hist": color_hist,
    "lbp": lbp,
    "hog": hog,
    **MPEG7,
}

IMPORTED = {
    "mp7_ehd_ref": {
        "label": "Edge Histogram (provided)",
        "family": "texture",
        "standard": "MPEG-7, provided",
        "note": "80 local + 5 global + 65 semi-global bins",
    },
    "mp7_htd_ref": {
        "label": "Homogeneous Texture (provided)",
        "family": "texture",
        "standard": "MPEG-7, provided",
        "note": "DC, SD, 30 scale x orientation energies, 11 marginals",
    },
    "gist": {
        "label": "GIST",
        "family": "scene",
        "standard": "provided",
        "note": "global scene descriptor, 480 dimensions",
    },
}
IMPORTED_DESCRIPTORS = frozenset(IMPORTED)

DESCRIPTOR_META = {
    "color_hist": {"label": "Colour histogram", "family": "colour", "standard": "course"},
    "lbp": {"label": "Local Binary Patterns", "family": "texture", "standard": "course"},
    "hog": {"label": "Oriented gradients", "family": "shape", "standard": "course"},
    **MPEG7_META,
    **IMPORTED,
}

SCALED_DESCRIPTORS = frozenset(MPEG7_SCALED) | {"mp7_htd_ref"}


def compute(img, names=None):
    """Return {descriptor_name: vector} for one RGB image in [0, 1]."""
    wanted = list(DESCRIPTORS) if names is None else [n for n in names if n in DESCRIPTORS]
    ctx = ImageContext(img)
    return {name: DESCRIPTORS[name](ctx) for name in wanted}
