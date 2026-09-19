"""What a plain-English instruction asks for.

Two readings of the same text. `parse()` is the union: which region kinds to detect, which
sides and roof edges to keep — what `/detect` needs. `specification()` keeps the clauses
apart: "C9 warm white on the front roofline at 12 inch spacing, mini lights in the hedges" is
two requests, each with its own place to light and its own lighting, so the design can be
confirmed request by request before anything is generated.

Rules, not a language model: every reading here is deterministic and runs offline, and what
the rules could not read is reported (`unread`) rather than guessed at.
"""

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
COLLAR_TERMS = ["collar"]
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
SLOPED_EDGES = ["rake", "hip"]

# Places that are lit but never detected: the salesperson draws them. A wreath is its own
# region — the thing that hangs on the door — so a wreath product asks for a wreath region.
MANUAL_PATTERNS = {
    "door": [r"\b(?:front |entry )?doors?\b(?! ?frame)", r"\bentry(?:way)?\b"],
    "garage_door": [r"\bgarage(?: doors?)?\b"],
    "window": [r"\bwindows?\b"],
    "column": [r"\bcolumns?\b", r"\bpillars?\b", r"\bposts?\b"],
    "fence": [r"\bfences?\b", r"\brailings?\b", r"\bbanisters?\b"],
    "driveway_edge": [r"\bdriveway\b"],
    "walkway_edge": [r"\bwalkway\b", r"\bpathway\b", r"\bsidewalk\b", r"\bpath\b"],
    "wreath": [r"\bwreaths?\b"],
}

# What lights a place when the instruction does not say: rooflines take a socket line, planting
# and palms take strands. Reported as an assumption, never silently.
DEFAULT_CATEGORY = {
    "roof": "c9", "hedge": "mini", "shrub": "mini", "topiary": "mini", "tree_canopy": "mini",
    "palm_trunk": "mini", "palm_collar": "mini", "palm_frond": "mini", "wreath": "wreath",
}

# -- lighting vocabulary ----------------------------------------------------------------------

CATEGORY_PATTERNS = {
    "c9": [r"\bc-?9s?\b"],
    "c7": [r"\bc-?7s?\b"],
    "mini": [r"\bminis?\b", r"\bmini[- ]?lights?\b", r"\bfairy lights?\b", r"\bstring lights?\b"],
    "icicle": [r"\bicicles?\b"],
    "wreath": [r"\bwreaths?\b"],
    "garland": [r"\bgarlands?\b"],
}

# Longest phrases first, so "warm white" is read before "white" and "cool white" is never
# split into a stray "white". A bare "white" is taken literally as pure white and reported as
# an assumption, because installers often mean warm white by it.
COLOUR_PHRASES: list[tuple[str, str]] = [
    (r"warm[- ]white", "warm_white"), (r"soft[- ]white", "warm_white"),
    (r"cool[- ]white", "cool_white"), (r"cold[- ]white", "cool_white"),
    (r"bright[- ]white", "cool_white"), (r"ice[- ]white", "cool_white"),
    (r"pure[- ]white", "pure_white"), (r"white", "pure_white"),
    (r"red", "red"), (r"green", "green"), (r"blue", "blue"),
    (r"amber", "amber"), (r"gold(?:en)?", "amber"), (r"yellow", "amber"),
    (r"orange", "orange"), (r"purple", "purple"), (r"violet", "purple"),
    (r"pink", "pink"), (r"magenta", "pink"),
]
# The sequence a "multicolour" strand carries in the catalogue.
MULTICOLOUR = ["red", "green", "blue", "amber"]
MULTICOLOUR_PATTERN = r"\bmulti(?:-?colou?r(?:ed)?)?\b"
CANDY_CANE_PATTERN = r"\bcandy[- ]cane\b"
# Colour words we know are colours but the palette does not carry: reported, never mapped.
UNKNOWN_COLOURS = [
    "teal", "turquoise", "cyan", "aqua", "silver", "black", "brown", "lime", "peach", "coral",
    "lavender", "champagne", "rose",
]
COUNT_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "single": 1, "double": 2}

INCHES = r"(\d+(?:\.\d+)?)\s*(?:\"|”|in\b|inch(?:es)?\b)"
# A number in inches just before a wreath or garland word is its size, not a spacing.
SIZE_FOLLOWS = r"\s*(?:wreaths?|garlands?|wide|diameter|thick|round|across)\b"
FEET_APART = r"(\d+(?:\.\d+)?)\s*(?:ft|foot|feet)\s*(?:apart|spacing|on cent(?:er|re)s?|centers|centres)"

TWINKLE_OFF = r"\b(?:steady|no twinkl\w*|non-?twinkl\w*|not twinkl\w*|without twinkl\w*)\b"
TWINKLE_ON = r"\btwinkl\w*\b"


def _edges(text: str, roof_requested: bool) -> list[str]:
    """The roof edges an instruction asks for.

    A narrow phrase on its own wins ("eaves only", "gutter line", "the ridge"). The outline
    phrasing widens to eaves, rakes and hips and adds any other edge also named ("roofline
    and the ridge"), unless "only" or "just" makes the narrow phrase exclusive. A roof
    request that names no edge at all means the outline.
    """
    narrow = [edge for edge, patterns in EDGE_PATTERNS.items()
              if any(re.search(pattern, text) for pattern in patterns)]
    # A rake and a hip are the same thing to a customer: the sloping edge of a roof plane. The
    # reference role table splits them by plane shape (a plane that narrows to its top is hip
    # roofed, so its slopes are hips; a triangle counts as narrowing), which the wording of an
    # instruction does not track. So asking for either lights both.
    if any(edge in narrow for edge in SLOPED_EDGES):
        narrow += [edge for edge in SLOPED_EDGES if edge not in narrow]
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


KIND_ORDER = ["roof", "hedge", "shrub", "topiary", "palm_trunk", "palm_collar", "palm_frond", "tree_canopy"]


def _normalise(prompt: str) -> str:
    return " " + re.sub(r"\s+", " ", prompt.lower().strip()) + " "


def _unscoped(text: str) -> str:
    scoped = text
    for phrase in BOUNDARY_PHRASES:
        scoped = re.sub(phrase, " ", scoped)
    return scoped


def _targets(text: str) -> Intent:
    """The detectable kinds, sides and roof edges one stretch of normalised text names."""
    if not text.strip():
        return Intent()
    scoped = _unscoped(text)
    kinds: list[str] = []

    if "palm" in text:
        parts = (("palm_trunk", TRUNK_TERMS), ("palm_collar", COLLAR_TERMS), ("palm_frond", FROND_TERMS))
        wanted = [kind for kind, terms in parts if any(term in scoped for term in terms)]
        # A palm named without a part means the whole palm, all three zones.
        kinds += wanted or ["palm_trunk", "palm_collar", "palm_frond"]

    without_palms = re.sub(r"palm (?:tree|trees)?", " ", scoped)
    for kind, terms in KIND_TERMS.items():
        source = without_palms if kind == "tree_canopy" else scoped
        if any(term in source for term in terms):
            kinds.append(kind)

    sides = [side for side in SIDES if re.search(rf"\b{side}\b", text)]
    ordered = [k for k in KIND_ORDER if k in kinds]
    # Edges read from the scoped text, so a boundary like "the bottom of the fronds" is not
    # taken for "along the bottom".
    return Intent(kinds=ordered, sides=sides, edges=_edges(scoped, "roof" in ordered), matched=bool(ordered))


def parse(prompt: str) -> Intent:
    return _targets(_normalise(prompt))


# -- the specification -------------------------------------------------------------------------


@dataclass
class Lighting:
    """The lighting one clause describes, as spoken; resolved against the catalogue later."""

    category: str | None = None
    # Palette keys in the order spoken, with how many of each: "two red, one white".
    colours: list[str] = field(default_factory=list)
    counts: list[int] = field(default_factory=list)
    spacing_in: float | None = None
    # Wreath or garland size, when one is given.
    size_in: float | None = None
    twinkle: bool | None = None
    # Candy cane: the colour changes each turn of a trunk wrap rather than each bulb.
    per_turn: bool = False
    # Readings that had to be chosen rather than read, in the user's words.
    assumed: list[str] = field(default_factory=list)
    # Words that looked like lighting but the rules could not place.
    unread: list[str] = field(default_factory=list)

    def named(self) -> bool:
        return bool(self.category or self.colours or self.spacing_in or self.twinkle is not None
                    or self.size_in or self.per_turn)


@dataclass
class Request:
    """One clause: where the lights go and what lights they are."""

    text: str
    kinds: list[str] = field(default_factory=list)
    sides: list[str] = field(default_factory=list)
    edges: list[str] = field(default_factory=list)
    # Kinds in `kinds` that detection never finds and the salesperson draws by hand.
    manual: list[str] = field(default_factory=list)
    lighting: Lighting = field(default_factory=Lighting)

    def places(self) -> bool:
        return bool(self.kinds)


@dataclass
class Specification:
    requests: list[Request]
    intent: Intent


def _lighting(text: str) -> Lighting:
    found = Lighting()
    for category, patterns in CATEGORY_PATTERNS.items():
        if any(re.search(pattern, text) for pattern in patterns):
            found.category = category
            break

    # Sizes and spacings before colours, so "24 inch" is not left as an unread word.
    remaining = text
    for match in list(re.finditer(INCHES, remaining)):
        value = float(match.group(1))
        if re.match(SIZE_FOLLOWS, remaining[match.end():]):
            found.size_in = value
        else:
            found.spacing_in = value
    feet = re.search(FEET_APART, remaining)
    if feet:
        found.spacing_in = float(feet.group(1)) * 12
    remaining = re.sub(INCHES, " ", remaining)

    if re.search(TWINKLE_OFF, remaining):
        found.twinkle = False
    elif re.search(TWINKLE_ON, remaining):
        found.twinkle = True

    if re.search(CANDY_CANE_PATTERN, remaining):
        found.colours, found.counts, found.per_turn = ["red", "pure_white"], [1, 1], True
        remaining = re.sub(CANDY_CANE_PATTERN, " ", remaining)
    if re.search(MULTICOLOUR_PATTERN, remaining):
        found.colours, found.counts = list(MULTICOLOUR), [1] * len(MULTICOLOUR)
        remaining = re.sub(MULTICOLOUR_PATTERN, " ", remaining)

    # Colours in the order spoken, each with an optional count in front ("two red", "2x red")
    # or behind ("red x2"); a colour repeated straight after itself adds up.
    words = "|".join(pattern for pattern, _ in COLOUR_PHRASES)
    counted = re.compile(
        rf"(?:\b(?P<before>\d+|{'|'.join(COUNT_WORDS)})\s*x?\s*)?\b(?P<colour>{words})\b(?:\s*x\s*(?P<after>\d+))?"
    )
    steps: list[tuple[str, int]] = []
    for match in counted.finditer(remaining):
        key = next(k for pattern, k in COLOUR_PHRASES if re.fullmatch(pattern, match.group("colour")))
        count = match.group("before") or match.group("after") or "1"
        count = COUNT_WORDS.get(count, int(count) if count.isdigit() else 1)
        if match.group("colour") == "white":
            found.assumed.append('"white" read as pure white')
        if steps and steps[-1][0] == key and not match.group("before") and not match.group("after"):
            steps[-1] = (key, steps[-1][1] + 1)
        else:
            steps.append((key, count))
    if steps and not found.colours:
        found.colours = [key for key, _ in steps]
        found.counts = [count for _, count in steps]

    for word in UNKNOWN_COLOURS:
        if re.search(rf"\b{word}\b", remaining):
            found.unread.append(word)
    return found


def _split_on_and(piece: str) -> list[str]:
    """Split "X and Y" only where the second half is a request of its own.

    "C9 on the roof and mini lights in the hedges" is two requests; "red and green C9",
    "eaves and rakes" and "warm white on the roof and the hedges" are one each.
    """
    parts = [part.strip() for part in re.split(r"\b(?:and|plus|also)\b", piece)]
    if len(parts) == 1:
        return [piece]
    clauses = [parts[0]]
    for part in parts[1:]:
        left, right = _targets(" " + clauses[-1] + " "), _targets(" " + part + " ")
        # "the hedges and shrubs" shares one lighting; a second half that brings its own
        # lighting ("... and mini lights in the hedges") is a request of its own.
        if left.kinds and right.kinds and set(left.kinds) != set(right.kinds) and _lighting(" " + part + " ").named():
            clauses.append(part)
        else:
            clauses[-1] = clauses[-1] + " and " + part
    return clauses


def _clauses(text: str) -> list[str]:
    pieces = [p.strip() for p in re.split(r"[;,]|\.(?!\d)|\bthen\b", text) if p.strip()]
    return [clause.strip() for piece in pieces for clause in _split_on_and(piece) if clause.strip()]


def _manual(text: str, lighting: Lighting) -> list[str]:
    found = [kind for kind, patterns in MANUAL_PATTERNS.items()
             if any(re.search(pattern, text) for pattern in patterns)]
    # A wreath product hangs as a wreath region wherever it was said to go; the door or
    # window named is where to draw it, not a thing to light in its own right.
    if lighting.category == "wreath":
        return ["wreath"]
    return found


def _inherit(into: Lighting, source: Lighting) -> None:
    """Fill what a clause left unsaid from the lighting named next to it."""
    if into.category is None:
        into.category = source.category
    if not into.colours:
        into.colours, into.counts, into.per_turn = list(source.colours), list(source.counts), source.per_turn
    if into.spacing_in is None:
        into.spacing_in = source.spacing_in
    if into.twinkle is None:
        into.twinkle = source.twinkle
    if into.size_in is None:
        into.size_in = source.size_in


def _as_written(prompt: str, text: str) -> str:
    """A clause in the user's own casing, found piece by piece in the original instruction."""
    lowered = prompt.lower()
    pieces = []
    for piece in text.split(", "):
        at = lowered.find(piece)
        pieces.append(prompt[at:at + len(piece)] if at >= 0 else piece)
    return ", ".join(pieces)


def specification(prompt: str) -> Specification:
    """The instruction as separate requests, plus the union that detection works from.

    A clause that names lighting but no place ("at 12 inch spacing") refines the request
    before it — every request so far when it says "all", the one after it when it comes
    first. A clause that names a place but no lighting takes the lighting of the request
    before it.
    """
    text = _normalise(prompt)
    requests: list[Request] = []
    pending: Lighting | None = None
    for clause in _clauses(text):
        padded = " " + clause + " "
        targets = _targets(padded)
        lighting = _lighting(_unscoped(padded))
        manual = _manual(padded, lighting)
        kinds = targets.kinds + [kind for kind in manual if kind not in targets.kinds]
        if not kinds:
            if not lighting.named():
                continue
            if requests:
                # "all warm white" refines every request so far; anything else, the last one.
                spans_all = re.match(r"\s*(?:all|everything|both|the whole lot)\b", clause)
                for request in requests if spans_all else requests[-1:]:
                    _inherit(request.lighting, lighting)
                    request.lighting.assumed += lighting.assumed
                    request.lighting.unread += lighting.unread
                    request.text += ", " + clause
            else:
                pending = lighting if pending is None else pending
                if pending is not lighting:
                    _inherit(pending, lighting)
            continue
        request = Request(text=clause, kinds=kinds, sides=targets.sides, edges=targets.edges,
                          manual=manual, lighting=lighting)
        if pending is not None:
            _inherit(request.lighting, pending)
            pending = None
        elif requests and not lighting.named():
            _inherit(request.lighting, requests[-1].lighting)
        requests += _by_product(request)
    for request in requests:
        request.text = _as_written(prompt, request.text)
    return Specification(requests=requests, intent=parse(prompt))


def _by_product(request: Request) -> list[Request]:
    """"Warm white on the roof and the hedges" names no product, and the roof and the hedges
    take different ones by default, so it becomes one request per default product."""
    if request.lighting.category is not None:
        return [request]
    groups: dict[str | None, list[str]] = {}
    for kind in request.kinds:
        groups.setdefault(DEFAULT_CATEGORY.get(kind), []).append(kind)
    if len(groups) < 2:
        return [request]
    return [
        Request(text=request.text, kinds=kinds, sides=request.sides,
                edges=request.edges if "roof" in kinds else [],
                manual=[kind for kind in request.manual if kind in kinds],
                lighting=Lighting(**vars(request.lighting)))
        for kinds in groups.values()
    ]
