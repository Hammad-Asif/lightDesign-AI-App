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
EAVE_TERMS = ["eave", "fascia", "gutter", "roofline", "roof line", "roof edge"]
# A gable front is lit along its rakes, the sloping edges, not along a ridge. "gable" used to
# be a ridge term, which only worked while rakes were mislabelled "ridge"; once rakes carried
# their own label it filtered every gable line out of a "light the gable" instruction.
RAKE_TERMS = ["rake", "gable"]
RIDGE_TERMS = ["ridge", "peak"]


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
    edges = []
    if any(term in text for term in EAVE_TERMS):
        edges.append("eave")
    if any(term in text for term in RAKE_TERMS):
        edges.append("rake")
    if any(term in text for term in RIDGE_TERMS):
        edges.append("ridge")

    ordered = [k for k in
               ["roof", "hedge", "shrub", "topiary", "palm_trunk", "palm_frond", "tree_canopy"]
               if k in kinds]
    return Intent(kinds=ordered, sides=sides, edges=edges, matched=bool(ordered))
