"""Image loading and lazily shared descriptor inputs."""

import io
from functools import cached_property

import numpy as np
from PIL import Image
from skimage.color import rgb2gray, rgb2hsv
from skimage.transform import resize

WORK_SIZE = (128, 128)   # grayscale working size for LBP / HOG
MPEG7_SIZE = (64, 64)    # grayscale working size for the MPEG-7 texture descriptors


def load_image(path):
    """Read an image file as an RGB float array in [0, 1]."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


def load_image_bytes(data):
    """Same, from raw bytes (an uploaded query image)."""
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0


class ImageContext:
    """One RGB image plus the lazily-computed views descriptors ask for."""

    def __init__(self, rgb):
        self.rgb = rgb

    @cached_property
    def hsv(self):
        """HSV in [0, 1]^3 — feeds the colour histogram, SCD and (as HMMD) the CSD."""
        return rgb2hsv(self.rgb)

    @cached_property
    def gray(self):
        """Luminance in [0, 1] at the image's own size; the two resizes build on it."""
        return rgb2gray(self.rgb)

    @cached_property
    def gray_work(self):
        """128x128 luminance in [0, 1], for LBP and HOG."""
        return resize(self.gray, WORK_SIZE, anti_aliasing=True)

    @cached_property
    def gray_mpeg7(self):
        """64x64 luminance on the 0-255 scale the MPEG-7 thresholds are defined on."""
        g = self.gray
        if g.shape != MPEG7_SIZE:
            g = resize(g, MPEG7_SIZE, anti_aliasing=True)
        return (g * 255.0).astype(np.float64)
