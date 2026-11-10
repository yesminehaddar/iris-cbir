"""Closed visual-property vocabulary shared by indexing and the API."""

LIGHTNESS = ("dark", "midtone", "bright")
SATURATION = ("greyscale", "muted", "vivid")
DENSITY = ("smooth", "detailed", "textured")
#: Indexed by the MPEG-7 edge histogram's five edge types, in the standard's own
#: order: vertical, horizontal, 45 degrees, 135 degrees, non-directional. Both
#: diagonals deliberately map to the same word — "diagonal" twice is what makes
#: `EDGE_WORDS[direction]` work for a direction of 2 or 3 — so this tuple has
#: five entries but only four distinct words. Do not deduplicate it: that would
#: shift `busy` onto the 135-degree type and leave non-directional edges unnamed.
EDGE_WORDS = ("vertical", "horizontal", "diagonal", "diagonal", "busy")
HUE_NAMES = (
    "red", "orange", "yellow", "lime", "green", "teal",
    "cyan", "sky", "blue", "purple", "magenta", "pink",
)

#: Bit order of the mask: 25 distinct words, one bit each, which is why they fit
#: in a uint32. `dict.fromkeys` collapses the repeated "diagonal" while keeping
#: first-seen order. Written to tagvocab.json at index time, which stays the
#: authoritative order in case this tuple is ever reordered without a rebuild.
VOCABULARY = tuple(
    dict.fromkeys(LIGHTNESS + SATURATION + DENSITY + EDGE_WORDS + HUE_NAMES)
)
BIT = {word: 1 << index for index, word in enumerate(VOCABULARY)}

#: How the form groups them: (key, label, words). One choice per facet.
FACETS = (
    ("lightness", "Lightness", LIGHTNESS),
    ("saturation", "Saturation", SATURATION),
    ("texture", "Texture", DENSITY),
    ("hue", "Dominant hue", HUE_NAMES),
    ("edges", "Dominant edges", tuple(dict.fromkeys(EDGE_WORDS))),
)


def facets_for(words):
    """The facets restricted to `words`, for a base that only knows some of them."""
    known = set(words)
    out = []
    for key, label, choices in FACETS:
        available = [w for w in choices if w in known]
        if available:
            out.append({"key": key, "label": label, "words": available})
    return out
