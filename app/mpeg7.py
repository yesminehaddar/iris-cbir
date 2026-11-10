"""MPEG-7 visual descriptors computed from image pixels."""

import math
from functools import lru_cache

import numpy as np
from scipy.fft import dctn

SQRT2 = math.sqrt(2.0)

# --- Scalable Colour (SCD) ---------------------------------------------------

SCD_BINS = (16, 4, 4)  # H, S, V -> 256 histogram bins
SCD_COEFFS = 64        # Haar coefficients kept: the "scalable" part of the name


def _haar(values):
    """MPEG-7 Haar transform: recurse on the sums, emit the differences.

    Coefficients come back coarsest-first, so truncating the tail is exactly the
    scalability the descriptor is named for.
    """
    bands = []
    current = values.astype(np.float64)
    while current.size > 1:
        bands.append(current[0::2] - current[1::2])
        current = current[0::2] + current[1::2]
    bands.append(current)
    return np.concatenate(bands[::-1]).astype(np.float32)


def scalable_color(ctx):
    """SCD: 256-bin HSV histogram, Haar-transformed, first 64 coefficients.

    Bins run V-major, then S, with H varying fastest. The order is what decides
    which information truncation throws away: keeping 64 of the 256 coefficients
    resolves the histogram down to blocks of four adjacent bins, so whichever axis
    varies fastest is the one that gets summed over and lost.

    With H fastest, a block is four neighbouring hues inside one (V, S) cell, and
    brightness survives. With V fastest it does not — and that is not academic:
    every greyscale image has all its pixels at S=0, H=0, so its whole histogram
    lives in four bins that differ only in V. Ordered V-fastest those four bins
    fall in a single block, and all 7.4% of this base that is black and white
    collapses onto one identical vector regardless of how light or dark it is.
    """
    n_h, n_s, n_v = SCD_BINS
    hsv = ctx.hsv
    q_h = np.minimum((hsv[..., 0] * n_h).astype(np.intp), n_h - 1)
    q_s = np.minimum((hsv[..., 1] * n_s).astype(np.intp), n_s - 1)
    q_v = np.minimum((hsv[..., 2] * n_v).astype(np.intp), n_v - 1)
    cells = ((q_v * n_s + q_s) * n_h + q_h).ravel()
    hist = np.bincount(cells, minlength=n_h * n_s * n_v).astype(np.float64)
    total = hist.sum()
    if total:
        hist /= total
    return _haar(hist)[:SCD_COEFFS]


# --- Colour Layout (CLD) -----------------------------------------------------

CLD_COEFFS = (6, 3, 3)  # zigzag coefficients kept for Y, Cb, Cr

# Standard JPEG/MPEG-7 luminance-chrominance matrix, for RGB in [0, 1].
_RGB_TO_YCBCR = np.array(
    [
        [0.299000, 0.587000, 0.114000],
        [-0.168736, -0.331264, 0.500000],
        [0.500000, -0.418688, -0.081312],
    ],
    dtype=np.float64,
)


@lru_cache(maxsize=1)
def _zigzag():
    """Row and column indices of an 8x8 block in zigzag order."""
    cells = [(r, c) for r in range(8) for c in range(8)]
    # Walk the anti-diagonals, alternating direction on each one.
    cells.sort(key=lambda rc: (rc[0] + rc[1], rc[0] if (rc[0] + rc[1]) % 2 else -rc[0]))
    rows, cols = zip(*cells)
    return np.asarray(rows), np.asarray(cols)


def _block_average(img, rows, cols):
    """Average `img` down to rows x cols by partitioning it into equal blocks."""
    h, w = img.shape[:2]
    if h % rows == 0 and w % cols == 0:
        return img.reshape(rows, h // rows, cols, w // cols, -1).mean(axis=(1, 3))
    r_edges = (np.arange(rows + 1) * h) // rows
    c_edges = (np.arange(cols + 1) * w) // cols
    out = np.empty((rows, cols, img.shape[2]), dtype=np.float64)
    for i in range(rows):
        band = img[r_edges[i] : max(r_edges[i + 1], r_edges[i] + 1)]
        for j in range(cols):
            cell = band[:, c_edges[j] : max(c_edges[j + 1], c_edges[j] + 1)]
            out[i, j] = cell.mean(axis=(0, 1))
    return out


def color_layout(ctx):
    """CLD: 8x8 block averages -> YCbCr -> 8x8 DCT -> 6 Y + 3 Cb + 3 Cr coefficients."""
    small = _block_average(ctx.rgb.astype(np.float64), 8, 8)
    ycbcr = small @ _RGB_TO_YCBCR.T
    ycbcr[..., 1:] += 0.5  # centre the chrominance planes
    rows, cols = _zigzag()
    parts = []
    for channel, keep in enumerate(CLD_COEFFS):
        coeffs = dctn(ycbcr[..., channel], type=2, norm="ortho")
        parts.append(coeffs[rows[:keep], cols[:keep]])
    return np.concatenate(parts).astype(np.float32)


# --- Edge Histogram (EHD) ----------------------------------------------------

EHD_SUBIMAGES = 4        # 4 x 4 = 16 sub-images, each with its own 5-bin histogram
EHD_BLOCKS_PER_SIDE = 8  # blocks per sub-image side: 1024 blocks, the standard's target
EHD_THRESHOLD = 11.0     # on the 0-255 luminance scale, as in the standard

# The five edge filters, applied to the 2x2 sub-block means ordered
# (top-left, top-right, bottom-left, bottom-right).
_EHD_FILTERS = np.array(
    [
        [1.0, -1.0, 1.0, -1.0],      # vertical
        [1.0, 1.0, -1.0, -1.0],      # horizontal
        [SQRT2, 0.0, 0.0, -SQRT2],   # 45 degrees
        [0.0, SQRT2, -SQRT2, 0.0],   # 135 degrees
        [2.0, -2.0, -2.0, 2.0],      # non-directional
    ],
    dtype=np.float64,
)
EHD_EDGE_TYPES = ("vertical", "horizontal", "diagonal45", "diagonal135", "nondirectional")


def edge_histogram(ctx):
    """EHD: dominant edge type per image-block, histogrammed over 16 sub-images."""
    gray = ctx.gray_mpeg7
    blocks = EHD_SUBIMAGES * EHD_BLOCKS_PER_SIDE  # image-blocks per side
    h, w = gray.shape
    sub_h, sub_w = max(h // (2 * blocks), 1), max(w // (2 * blocks), 1)
    gray = gray[: 2 * blocks * sub_h, : 2 * blocks * sub_w]

    # Mean of each 2x2 sub-block, then the four means of every image-block.
    means = gray.reshape(2 * blocks, sub_h, 2 * blocks, sub_w).mean(axis=(1, 3))
    quad = np.stack(
        [means[0::2, 0::2], means[0::2, 1::2], means[1::2, 0::2], means[1::2, 1::2]],
        axis=-1,
    )
    responses = np.abs(quad @ _EHD_FILTERS.T)
    strongest = responses.argmax(axis=-1)
    is_edge = responses.max(axis=-1) >= EHD_THRESHOLD

    # Regroup (blocks, blocks) into (16 sub-images, blocks per sub-image).
    per_side, cells = EHD_SUBIMAGES, EHD_BLOCKS_PER_SIDE
    shape = (per_side, cells, per_side, cells)
    strongest = strongest.reshape(shape).transpose(0, 2, 1, 3).reshape(per_side**2, -1)
    is_edge = is_edge.reshape(shape).transpose(0, 2, 1, 3).reshape(per_side**2, -1)

    # Bin 5 is "no dominant edge"; it is counted then dropped, which is what makes
    # a flat sub-image show up as five near-zero bins instead of a uniform one.
    codes = np.where(is_edge, strongest, 5)
    offsets = np.arange(per_side**2)[:, None] * 6
    counts = np.bincount((offsets + codes).ravel(), minlength=per_side**2 * 6)
    counts = counts.reshape(-1, 6)[:, :5].astype(np.float64)
    return (counts / codes.shape[1]).ravel().astype(np.float32)


# --- Colour Structure (CSD) --------------------------------------------------

CSD_BINS = 64
CSD_WINDOW = 8  # side of the square structuring element

# HMMD quantisation: (diff_low, diff_high, hue levels, sum levels) per subspace.
# 1x8 + 4x4 + 4x4 + 4x4 + 4x2 = 64 cells.
_CSD_SUBSPACES = (
    (0.0, 6.0, 1, 8),
    (6.0, 20.0, 4, 4),
    (20.0, 60.0, 4, 4),
    (60.0, 110.0, 4, 4),
    (110.0, 256.0, 4, 2),
)


def _hmmd_cells(ctx):
    """Quantise every pixel into one of the 64 HMMD cells of the CSD."""
    hsv = ctx.hsv
    hue = hsv[..., 0] * 360.0
    maximum = hsv[..., 2] * 255.0           # HMMD "Max"
    diff = hsv[..., 1] * maximum            # HMMD "Diff" = Max - Min
    summed = maximum - diff / 2.0           # HMMD "Sum" = (Max + Min) / 2

    cells = np.zeros(hue.shape, dtype=np.intp)
    offset = 0
    for low, high, n_hue, n_sum in _CSD_SUBSPACES:
        band = (diff >= low) & (diff < high)
        if band.any():
            q_hue = np.minimum((hue[band] / 360.0 * n_hue).astype(np.intp), n_hue - 1)
            q_sum = np.minimum((summed[band] / 256.0 * n_sum).astype(np.intp), n_sum - 1)
            cells[band] = offset + q_hue * n_sum + q_sum
        offset += n_hue * n_sum
    return cells


def _csd_subsample(h, w):
    """Standard's rule: subsample so the 8x8 element always covers the same share."""
    return max(int(round(0.5 * math.log2(max(h * w, 1)) - 8.0)), 0)


# One bit per HMMD cell, so "which colours does this window hold" is a 64-bit OR.
_CSD_BIT = np.array([1 << i for i in range(CSD_BINS)], dtype="<u8")


def _sliding_or(bits, window, axis):
    """OR every `window` consecutive entries along `axis`, by repeated doubling."""
    out = bits
    span = 1
    while span < window:
        step = min(span, window - span)
        head, tail = [slice(None)] * out.ndim, [slice(None)] * out.ndim
        head[axis] = slice(None, out.shape[axis] - step)
        tail[axis] = slice(step, None)
        out = out[tuple(head)] | out[tuple(tail)]
        span += step
    return out


def color_structure(ctx):
    """CSD: share of 8x8 windows in which each of the 64 HMMD colours appears.

    The structuring element is evaluated as a bitwise OR rather than one integral
    image per colour: a window holds colour c iff bit c survives the OR of its
    pixels. Six array operations replace a loop over the colours present, which
    matters because this descriptor otherwise dominates indexing 800k images.
    """
    cells = _hmmd_cells(ctx)
    step = 2 ** _csd_subsample(*cells.shape)
    if step > 1:
        cells = cells[::step, ::step]

    h, w = cells.shape
    window = CSD_WINDOW
    if h < window or w < window:  # too small to slide: fall back to a plain histogram
        counts = np.bincount(cells.ravel(), minlength=CSD_BINS).astype(np.float64)
        return (counts / cells.size).astype(np.float32)

    windows = _sliding_or(_sliding_or(_CSD_BIT[cells], window, 1), window, 0)
    packed = np.ascontiguousarray(windows).view(np.uint8).reshape(-1, 8)
    present = np.unpackbits(packed, axis=1, bitorder="little").sum(axis=0, dtype=np.float64)
    positions = (h - window + 1) * (w - window + 1)
    return (present / positions).astype(np.float32)


# --- Homogeneous Texture (HTD) -----------------------------------------------

HTD_SCALES = 5         # radial frequency bands
HTD_ORIENTATIONS = 6   # 30-degree angular sectors; 5 x 6 = 30 channels


@lru_cache(maxsize=4)
def _htd_channels(shape):
    """Flat indices of the 30 frequency channels partitioning the spectrum."""
    h, w = shape
    f_y = np.fft.fftfreq(h)[:, None] * 2.0  # Nyquist at 1
    f_x = np.fft.fftfreq(w)[None, :] * 2.0
    radius = np.hypot(f_y, f_x)
    angle = np.mod(np.arctan2(f_y, f_x), np.pi)  # the power spectrum is symmetric

    channels = []
    for scale in range(HTD_SCALES):
        low, high = 2.0 ** -(scale + 1), 2.0**-scale
        band = (radius > low) & (radius <= high)
        for sector in range(HTD_ORIENTATIONS):
            start = sector * np.pi / HTD_ORIENTATIONS
            stop = (sector + 1) * np.pi / HTD_ORIENTATIONS
            mask = band & (angle >= start) & (angle < stop)
            channels.append(np.flatnonzero(mask.ravel()))
    return tuple(channels)


def homogeneous_texture(ctx):
    """HTD: log energy mean and deviation in 30 channels, plus the DC pair."""
    gray = ctx.gray_mpeg7
    power = (np.abs(np.fft.fft2(gray)) ** 2).ravel()
    energies, deviations = [], []
    for indices in _htd_channels(gray.shape):
        if indices.size == 0:
            energies.append(0.0)
            deviations.append(0.0)
            continue
        band = power[indices]
        energies.append(math.log10(1.0 + band.mean()))
        deviations.append(math.log10(1.0 + band.std()))
    head = [float(gray.mean()), float(gray.std())]
    return np.asarray(head + energies + deviations, dtype=np.float32)


# --- registry ----------------------------------------------------------------

MPEG7 = {
    "mp7_scd": scalable_color,
    "mp7_cld": color_layout,
    "mp7_csd": color_structure,
    "mp7_ehd": edge_histogram,
    "mp7_htd": homogeneous_texture,
}

MPEG7_META = {
    "mp7_scd": {"label": "Scalable Colour", "family": "colour", "standard": "MPEG-7"},
    "mp7_cld": {"label": "Colour Layout", "family": "colour", "standard": "MPEG-7"},
    "mp7_csd": {"label": "Colour Structure", "family": "colour", "standard": "MPEG-7"},
    "mp7_ehd": {"label": "Edge Histogram", "family": "texture", "standard": "MPEG-7"},
    "mp7_htd": {"label": "Homogeneous Texture", "family": "texture", "standard": "MPEG-7"},
}

# Descriptors whose components do not share a scale, so a raw L2 would be decided
# by one or two of them: the DCT coefficient against the AC ones for CLD, the
# 0-255 mean against the log energies for HTD. The indexer divides each column by
# its deviation over the base and stores the divisor next to the matrix.
MPEG7_SCALED = frozenset({"mp7_cld", "mp7_htd"})
