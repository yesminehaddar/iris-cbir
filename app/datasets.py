"""Dataset configuration shared by the API, indexer, and feature store."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


def _path(env_var, default):
    """Resolve a configured path, relative to the project root when not absolute."""
    value = Path(os.environ.get(env_var, default))
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


@dataclass(frozen=True)
class Dataset:
    """One image base and its indexing/search configuration."""

    name: str
    label: str
    images_dir: Path
    features_dir: Path
    descriptors: tuple[str, ...]
    labelled: bool
    keyword_source: str
    rank_keywords: bool
    derived_tags: bool

    @property
    def url_prefix(self):
        """Where this base's images are mounted, and the prefix stored in its rows.

        The indexer writes `<url_prefix>/<relative path>` into `images.url`, and
        the app mounts `images_dir` at the same place, so a stored URL resolves
        without the two ends having to agree on anything else.
        """
        return f"/images/{self.name}"


OBJECTS = Dataset(
    name="objects",
    label="Objects (synthetic, not served)",
    images_dir=_path("OBJECTS_DIR", "data/objects"),
    features_dir=_path("OBJECTS_FEATURES_DIR", "data/features/objects"),
    descriptors=("color_hist", "lbp", "hog"),
    labelled=True,
    keyword_source="class name and filename",
    rank_keywords=True,
    derived_tags=False,
)

FLICKR = Dataset(
    name="flickr1m",
    label="Flickr 1M thumbnails",
    images_dir=_path("FLICKR_DIR", "../thumbnails"),
    features_dir=_path("FLICKR_FEATURES_DIR", "data/features/flickr1m"),
    descriptors=(
        "mp7_scd", "mp7_cld", "mp7_csd", "mp7_ehd", "mp7_htd",
        "mp7_ehd_ref", "mp7_htd_ref", "gist",
    ),
    labelled=False,
    keyword_source="tags people typed on Flickr",
    rank_keywords=True,
    derived_tags=True,
)

DATASETS = {ds.name: ds for ds in (FLICKR,)}

ALL_DATASETS = {ds.name: ds for ds in (FLICKR, OBJECTS)}

DEFAULT_DATASET = FLICKR.name


def get(name):
    """Look up a served dataset by name."""
    return DATASETS.get(name)
