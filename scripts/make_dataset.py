"""Generate the synthetic labelled base (3 classes x 10 images) in OBJECTS_DIR.

    python scripts/make_dataset.py
    python scripts/build_index.py objects
    python scripts/evaluate.py objects --k 9

This base is *not served* -- see app/datasets.py. It exists only because
precision and recall at k need class labels, and the Flickr base has none: its
folders are 10 000-image shards. The shape is what the class name says and the
colour is picked at random, which is exactly why the evaluation is meaningful --
HOG, a shape descriptor, must score high and the colour histogram must sit at
chance. docs/DESIGN.md (Evaluation) has the numbers.
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw

from app import datasets

SIZE = 160
PER_CLASS = 10
SHAPES = ["circle", "square", "triangle"]
COLORS = {
    "red": (214, 54, 46),
    "green": (48, 158, 76),
    "blue": (47, 111, 219),
    "yellow": (229, 182, 38),
    "purple": (132, 68, 190),
}


def draw(shape, color, rng):
    """One image: the named shape in the given colour, on a near-white ground.

    Position and padding jitter so the base is not 10 identical pictures per
    class, which would make any descriptor look perfect.
    """
    bg = tuple(rng.randint(225, 250) for _ in range(3))
    img = Image.new("RGB", (SIZE, SIZE), bg)
    d = ImageDraw.Draw(img)
    pad = rng.randint(20, 40)
    box = (pad, pad, SIZE - pad, SIZE - pad)
    if shape == "circle":
        d.ellipse(box, fill=color)
    elif shape == "square":
        d.rectangle(box, fill=color)
    else:
        d.polygon(
            [((box[0] + box[2]) // 2, box[1]), (box[0], box[3]), (box[2], box[3])],
            fill=color,
        )
    return img


def main():
    # Parsed even though there are no options: without it, `--help` silently
    # wrote 30 files instead of printing anything.
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()

    rng = random.Random(0)
    root = datasets.OBJECTS.images_dir
    count = 0
    for shape in SHAPES:
        folder = root / shape
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(PER_CLASS):
            name = rng.choice(list(COLORS))
            img = draw(shape, COLORS[name], rng)
            img.save(folder / f"{shape}_{name}_{i:02d}.png")
            count += 1
    print(f"{count} images written to {root}")


if __name__ == "__main__":
    main()
