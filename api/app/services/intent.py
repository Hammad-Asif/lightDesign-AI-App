import re
from dataclasses import dataclass, field

KIND_TERMS: dict[str, list[str]] = {
    "roof": ["roofline", "roof line", "roof", "eave", "fascia", "gutter", "gable", "rake", "ridge", "peak"],
    "hedge": ["hedge"],
    "shrub": ["shrub", "bush"],
    "topiary": ["topiary", "topiaries"],
    "tree_canopy": ["tree canopy", "canopy", "tree"],
}

TRUNK_TERMS = ["trunk", "wrap"]
FROND_TERMS = ["frond", "palm leaves", "palm leaf"]

BOUNDARY_PHRASES = [
    r"up to the (?:bottom|base) of the fronds?",
    r"to the (?:bottom|base) of the fronds?",
    r"up to the (?:bottom|base) of the husk",
    r"to the (?:bottom|base) of the husk",
    r"below the fronds?",
    r"under the fronds?",
]

SIDES = ["front", "left", "right"]

# What a customer means by "the roofline" or an unqualified "C9 on the roof": the whole
# outline, eaves plus every sloped edge up to the peak, with the ridge left dark.
OUTLINE_EDGES = ["eave", "rake", "hip"]
OUTLINE_PATTERNS = [r"\broof ?lines?\b", r"\broof edges?\b", r"\boutline\b"]

# Narrow phrasings keep their own meaning and are never widened to the outline. Whole words
# only, so "leave" is not read as "eave". A gable front is lit along its rakes, the sloping
# edges, not along a ridge: "gable" used to be a ridge term, which only worked while rakes
# were mislabelled "ridge".
EDGE_PATTERNS = {
    "eave": [r"\beaves?\b", r"\bfascia\b", r"\bgutters?\b", r"\balong the bottom\b", r"\bbottom edges?\b"],
    "rake": [r"\brakes?\b", r"\bgables?\b"],
    "hip": [r"\bhips?\b"],
    "ridge": [r"\bridges?\b", r"\bpeaks?\b", r"\balong the top\b", r"\btop edges?\b"],
}
EXCLUSIVE = r"\b(?:only|just)\b"


def _edges(text: str, roof_requested: bool) -> list[str]:
    """The roof edges an instruction asks for.

    A narrow phrase on its own wins ("eaves only", "gutter line", "the ridge"). The outline
    phrasing widens to eaves, rakes and hips and adds any other edge also named ("roofline
    and the ridge"), unless "only" or "just" makes the narrow phrase exclusive. A roof
    request that names no edge at all means the outline.
    """
    narrow = [edge for edge, patterns in EDGE_PATTERNS.items()
              if any(re.search(pattern, text) for pattern in patterns)]
    outline = any(re.search(pattern, text) for pattern in OUTLINE_PATTERNS)
    if narrow and (not outline or re.search(EXCLUSIVE, text)):
        return narrow
    if outline:
        return OUTLINE_EDGES + [edge for edge in narrow if edge not in OUTLINE_EDGES]
    return list(OUTLINE_EDGES) if roof_requested else []


@dataclass
class Intent:
    kinds: list[str] = field(default_factory=list)
    sides: list[str] = field(default_factory=list)
    edges: list[str] = field(default_factory=list)
    matched: bool = False


def parse(prompt: str) -> Intent:
    text = " " + re.sub(r"\s+", " ", prompt.lower().strip()) + " "
    if not text.strip():
        return Intent()

    scoped = text
    for phrase in BOUNDARY_PHRASES:
        scoped = re.sub(phrase, " ", scoped)

    kinds: list[str] = []

    if "palm" in text:
        wants_trunk = any(term in scoped for term in TRUNK_TERMS)
        wants_frond = any(term in scoped for term in FROND_TERMS)
        if not wants_trunk and not wants_frond:
            kinds += ["palm_trunk", "palm_frond"]
        else:
            if wants_trunk:
                kinds.append("palm_trunk")
            if wants_frond:
                kinds.append("palm_frond")

    without_palms = re.sub(r"palm (?:tree|trees)?", " ", scoped)
    for kind, terms in KIND_TERMS.items():
        source = without_palms if kind == "tree_canopy" else scoped
        if any(term in source for term in terms):
            kinds.append(kind)

    sides = [side for side in SIDES if re.search(rf"\b{side}\b", text)]
    ordered = [k for k in
               ["roof", "hedge", "shrub", "topiary", "palm_trunk", "palm_frond", "tree_canopy"]
               if k in kinds]
    # Edges read from the scoped text, so a boundary like "the bottom of the fronds" is not
    # taken for "along the bottom".
    return Intent(kinds=ordered, sides=sides, edges=_edges(scoped, "roof" in ordered),
                  matched=bool(ordered))
