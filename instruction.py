"""
Reads a lighting instruction ("C9 warm white on the front roofline at 12 inch spacing, mini lights in
the hedges") into targets: where the lights go, which product, colours, spacing, twinkle, plus the
assumptions made on the way. Fixed rules, no model call, so reading it again is free and repeatable.

Rules (after the Light Design product's reader):
  * clauses split on , ; . and "then"; "and / plus / also" splits only when both halves name a place
  * a clause with lighting but no place refines the clause before it ("all ..." refines every one)
  * a clause with a place but no lighting takes the previous clause's lighting
  * with no product named, places whose default products differ become separate targets
  * "roofline", "roof edge", "outline" mean eave + rake + hip; the ridge stays dark unless named
"""

from __future__ import annotations

import re
import uuid

# --------------------------------------------------------------------------------------------
# Catalogue
# --------------------------------------------------------------------------------------------

COLOURS = {
    "warm_white": {"label": "Warm white", "hex": "#ffb86b"}, "pure_white": {"label": "Pure white", "hex": "#fff4e0"},
    "cool_white": {"label": "Cool white", "hex": "#e8f1ff"}, "red": {"label": "Red", "hex": "#ff2a2a"},
    "green": {"label": "Green", "hex": "#1fd65f"}, "blue": {"label": "Blue", "hex": "#2f6bff"},
    "amber": {"label": "Amber", "hex": "#ffb000"}, "orange": {"label": "Orange", "hex": "#ff7a1a"},
    "purple": {"label": "Purple", "hex": "#9b4dff"}, "pink": {"label": "Pink", "hex": "#ff5fb0"},
}

PRODUCTS = [
    {"id": "c9", "name": "C9 LED", "category": "socket", "bulb_in": 1.125, "spacing_in": 12, "spacing_options": [12, 15, 18, 24, 36], "twinkle": False},
    {"id": "c7", "name": "C7 LED", "category": "socket", "bulb_in": 0.875, "spacing_in": 12, "spacing_options": [12, 15, 18, 24, 36], "twinkle": False},
    {"id": "mini", "name": "Mini lights", "category": "strand", "spacing_in": 4, "spacing_options": [4], "twinkle": False, "strand_ft": 24},
    {"id": "icicle", "name": "Icicle lights", "category": "strand", "spacing_in": 4, "spacing_options": [4], "twinkle": False, "drops_in": [6, 9, 12]},
    {"id": "wreath", "name": "Wreath", "category": "accent", "spacing_in": None, "spacing_options": [], "sizes_in": [24, 36, 48]},
    {"id": "garland", "name": "Garland", "category": "accent", "spacing_in": None, "spacing_options": []},
]
PRODUCT_BY_ID = {p["id"]: p for p in PRODUCTS}

# kind -> label, geometry, whether automatic detection can find it (and with which SAM 3 prompt)
KINDS: dict[str, dict] = {
    "roof":          {"label": "Roofline", "geometry": "line", "detect": "roof"},
    "fence":         {"label": "Fence", "geometry": "line", "detect": None},
    "driveway_edge": {"label": "Driveway edge", "geometry": "line", "detect": None},
    "walkway_edge":  {"label": "Walkway edge", "geometry": "line", "detect": None},
    "dormer":        {"label": "Dormer", "geometry": "area", "detect": "dormer"},
    "chimney":       {"label": "Chimney", "geometry": "area", "detect": "chimney"},
    "soffit":        {"label": "Soffit", "geometry": "area", "detect": "soffit"},
    "hedge":         {"label": "Hedges", "geometry": "area", "detect": "hedge"},
    "shrub":         {"label": "Shrubs", "geometry": "area", "detect": "shrub"},
    "topiary":       {"label": "Topiary", "geometry": "area", "detect": "topiary"},
    "palm_trunk":    {"label": "Palm trunks", "geometry": "area", "detect": "palm tree trunk"},
    "palm_frond":    {"label": "Palm fronds", "geometry": "area", "detect": "palm frond"},
    "tree_canopy":   {"label": "Trees", "geometry": "area", "detect": "tree"},
    "window":        {"label": "Windows", "geometry": "area", "detect": None},
    "door":          {"label": "Door", "geometry": "area", "detect": None},
    "garage_door":   {"label": "Garage door", "geometry": "area", "detect": None},
    "column":        {"label": "Columns", "geometry": "area", "detect": None},
    "wreath":        {"label": "Wreath", "geometry": "area", "detect": None},
    "garland":       {"label": "Garland", "geometry": "area", "detect": None},
}
EDGES = ["eave", "rake", "hip", "ridge", "valley"]
DEFAULT_PRODUCT = {k: "c9" for k in KINDS} | {k: "mini" for k in ("hedge", "shrub", "topiary", "palm_trunk", "palm_frond", "tree_canopy")} | {"wreath": "wreath", "garland": "garland"}

# phrase -> kind, longest first so "garage door" wins over "door" and "palm trunk" over "palm"
PLACE_WORDS = sorted([
    ("roofline", "roof"), ("roof line", "roof"), ("roof edge", "roof"), ("roof edges", "roof"), ("roof", "roof"), ("outline", "roof"),
    ("eave", "roof"), ("eaves", "roof"), ("fascia", "roof"), ("gutter", "roof"), ("gutters", "roof"), ("rake", "roof"), ("rakes", "roof"),
    ("gable", "roof"), ("gables", "roof"), ("hip", "roof"), ("hips", "roof"), ("ridge", "roof"), ("ridges", "roof"), ("peak", "roof"),
    ("peaks", "roof"), ("valley", "roof"), ("valleys", "roof"),
    ("fence", "fence"), ("driveway", "driveway_edge"), ("walkway", "walkway_edge"), ("path", "walkway_edge"), ("sidewalk", "walkway_edge"),
    ("dormer", "dormer"), ("dormers", "dormer"), ("chimney", "chimney"), ("soffit", "soffit"), ("soffits", "soffit"),
    ("hedge", "hedge"), ("hedges", "hedge"), ("hedgerow", "hedge"), ("shrub", "shrub"), ("shrubs", "shrub"), ("bush", "shrub"), ("bushes", "shrub"),
    ("topiary", "topiary"), ("topiaries", "topiary"), ("palm trunk", "palm_trunk"), ("palm trunks", "palm_trunk"), ("palm trees", "palm_trunk"),
    ("palm tree", "palm_trunk"), ("palms", "palm_trunk"), ("palm", "palm_trunk"), ("palm frond", "palm_frond"), ("palm fronds", "palm_frond"),
    ("fronds", "palm_frond"), ("tree", "tree_canopy"), ("trees", "tree_canopy"), ("canopy", "tree_canopy"),
    ("window", "window"), ("windows", "window"), ("garage door", "garage_door"), ("garage doors", "garage_door"), ("garage", "garage_door"),
    ("front door", "door"), ("door", "door"), ("doors", "door"), ("column", "column"), ("columns", "column"), ("pillar", "column"),
    ("pillars", "column"), ("post", "column"), ("posts", "column"), ("wreath", "wreath"), ("wreaths", "wreath"), ("garland", "garland"),
], key=lambda pw: -len(pw[0]))

EDGE_WORDS = {"eave": "eave", "eaves": "eave", "fascia": "eave", "gutter": "eave", "gutters": "eave", "along the bottom": "eave",
              "rake": "rake", "rakes": "rake", "gable": "rake", "gables": "rake", "hip": "hip", "hips": "hip",
              "ridge": "ridge", "ridges": "ridge", "peak": "ridge", "peaks": "ridge", "along the top": "ridge", "valley": "valley", "valleys": "valley"}
WHOLE_ROOF = ("roofline", "roof line", "roof edge", "outline", "roof")

COLOUR_WORDS = sorted([
    ("warm white", ["warm_white"]), ("soft white", ["warm_white"]), ("pure white", ["pure_white"]), ("bright white", ["pure_white"]),
    ("cool white", ["cool_white"]), ("cold white", ["cool_white"]), ("white", ["pure_white"]), ("red", ["red"]), ("green", ["green"]),
    ("blue", ["blue"]), ("amber", ["amber"]), ("gold", ["amber"]), ("orange", ["orange"]), ("purple", ["purple"]), ("violet", ["purple"]),
    ("pink", ["pink"]), ("multicolour", ["red", "green", "blue", "amber"]), ("multicolor", ["red", "green", "blue", "amber"]),
    ("multi-colour", ["red", "green", "blue", "amber"]), ("multi-color", ["red", "green", "blue", "amber"]), ("multi colored", ["red", "green", "blue", "amber"]),
    ("rainbow", ["red", "orange", "amber", "green", "blue", "purple"]), ("candy cane", ["red", "pure_white"]), ("christmas colours", ["red", "green"]),
    ("christmas colors", ["red", "green"]),
], key=lambda cw: -len(cw[0]))
NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "a": 1, "an": 1}
PRODUCT_WORDS = [(r"\bc\s?9s?\b", "c9"), (r"\bc\s?7s?\b", "c7"), (r"\bmini(?:[\s-]?lights?)?\b|\bstring lights?\b|\bfairy lights?\b", "mini"),
                 (r"\bicicles?\b", "icicle"), (r"\bwreaths?\b", "wreath"), (r"\bgarlands?\b", "garland")]
SIDE_WORDS = {"front": "front", "left": "left", "right": "right", "back": "back", "rear": "back"}


def _has(text: str, phrase: str) -> bool:
    return re.search(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", text) is not None


def _places(text: str) -> tuple[list[str], list[str]]:
    kinds: list[str] = []
    taken = text
    for phrase, kind in PLACE_WORDS:
        if _has(taken, phrase):
            if kind not in kinds:
                kinds.append(kind)
            taken = re.sub(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", " ", taken)
    # product names are not places ("wreath" the product also names the place; keep it)
    edges: list[str] = []
    if "roof" in kinds:
        named = [EDGE_WORDS[w] for w in sorted(EDGE_WORDS, key=len, reverse=True) if _has(text, w)]
        whole = any(_has(text, w) for w in WHOLE_ROOF)
        only = bool(re.search(r"\b(only|just)\b", text))
        edges = [] if (named and (only or not whole)) else ["eave", "rake", "hip"]
        for e in named:
            if e not in edges:
                edges.append(e)
        if ("rake" in edges) != ("hip" in edges) and not only:     # a rake and a hip are the same kind of sloped edge seen differently
            edges += [e for e in ("rake", "hip") if e not in edges]
        edges = [e for e in EDGES if e in edges]
    return kinds, edges


def _lighting(text: str) -> dict:
    out: dict = {}
    for pattern, pid in PRODUCT_WORDS:
        if re.search(pattern, text):
            out["product_id"] = pid
            break
    steps, taken = [], text
    for m in re.finditer(r"\b(one|two|three|four|five|six|\d)\s+([a-z ]+?)(?=,|\bthen\b|\band\b|$| on | in | along | at )", text):
        word = m.group(1)
        count = NUMBERS[word] if word in NUMBERS else int(word)
        for phrase, cols in COLOUR_WORDS:
            if _has(m.group(2), phrase):
                steps.append({"colour": cols[0], "count": int(count)})
                break
    if not steps:
        found: list[tuple[int, list[str]]] = []
        for phrase, cols in COLOUR_WORDS:
            m = re.search(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", taken)
            if m:
                found.append((m.start(), cols))
                taken = taken[:m.start()] + " " * len(phrase) + taken[m.end():]
        for _, cols in sorted(found):
            steps += [{"colour": c, "count": 1} for c in cols]
    if steps:
        out["pattern"] = steps
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:\"|”|in\b|inch(?:es)?\b)", text)
    if m:
        out["spacing_in"] = float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:ft|feet|foot|')\s*(?:apart|spacing|spaced)", text)
    if m:
        out["spacing_in"] = float(m.group(1)) * 12
    if re.search(r"\btwinkl(e|ing)\b|\bsparkl", text):
        out["twinkle"] = True
    if re.search(r"\bsteady\b|\bno twinkle\b", text):
        out["twinkle"] = False
    return out


def _sides(text: str) -> list[str]:
    text = text.replace("front door", " ")
    return [side for word, side in SIDE_WORDS.items() if _has(text, word)]


def _clauses(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"[,;.\n]|\bthen\b", text) if p.strip()]
    out: list[str] = []
    for part in parts:
        pieces = re.split(r"\b(?:and|plus|also|&)\b", part)
        buf = pieces[0]
        for piece in pieces[1:]:
            if _places(piece.lower())[0] and _places(buf.lower())[0] and _lighting(piece.lower()):
                out.append(buf.strip())
                buf = piece
            elif _places(piece.lower())[0] and _places(buf.lower())[0] and set(_places(piece.lower())[0]) - set(_places(buf.lower())[0]) and \
                    DEFAULT_PRODUCT.get(_places(piece.lower())[0][0]) != DEFAULT_PRODUCT.get(_places(buf.lower())[0][0]):
                out.append(buf.strip())
                buf = piece
            else:
                buf = f"{buf} and {piece}"
        out.append(buf.strip())
    return [re.sub(r"\s+", " ", c) for c in out if c]


def parse(text: str) -> list[dict]:
    """The targets an instruction names, in the order it names them."""
    requests: list[dict] = []
    pending: dict = {}                                              # lighting said before any place ("two red, one white C9 on ...")
    pending_text: list[str] = []
    for clause in _clauses(text):
        low = clause.lower()
        kinds, edges = _places(low)
        light = _lighting(low)
        if pending and kinds:
            merged = {**pending, **{k: v for k, v in light.items() if k != "pattern"}}
            merged["pattern"] = pending.get("pattern", []) + light.get("pattern", [])
            light, clause = merged, ", ".join(pending_text + [clause])
            pending, pending_text = {}, []
        if not kinds:
            if light and not requests:
                pending.update({k: v for k, v in light.items() if k != "pattern"})
                pending["pattern"] = pending.get("pattern", []) + light.get("pattern", [])
                pending_text.append(clause)
            elif light and requests:                                # lighting with no place refines what came before
                scope = requests if re.match(r"\s*all\b", low) else requests[-1:]
                for r in scope:
                    r["light"].update(light)
                    r["text"] += f", {clause}"
            continue
        if not light and requests:
            light = dict(requests[-1]["light"])
        requests.append({"text": clause, "kinds": kinds, "edges": edges, "sides": _sides(low), "light": light})

    targets: list[dict] = []
    for r in requests:
        light = r["light"]
        groups: dict[str, list[str]] = {}
        if light.get("product_id"):
            groups[light["product_id"]] = r["kinds"]
        else:
            for k in r["kinds"]:
                groups.setdefault(DEFAULT_PRODUCT[k], []).append(k)
        for pid, kinds in groups.items():
            targets.append(_target(r, kinds, pid, light))
    return targets


def _target(r: dict, kinds: list[str], pid: str, light: dict) -> dict:
    product = PRODUCT_BY_ID[pid]
    assumed, problems = [], []
    if not light.get("product_id"):
        assumed.append(product["name"])
    pattern = light.get("pattern")
    if not pattern and product["category"] in ("socket", "strand"):
        pattern = [{"colour": "warm_white", "count": 1}]
        assumed.append("warm white")
    if light.get("pattern") and any(s["colour"] == "pure_white" for s in light["pattern"]) and re.search(r"(?<![a-z])white", r["text"].lower()) \
            and not re.search(r"pure white|bright white|warm white|cool white|soft white|cold white", r["text"].lower()):
        assumed.append('"white" read as pure white')
    spacing = light.get("spacing_in")
    if spacing is not None and product["spacing_options"] and spacing not in product["spacing_options"]:
        problems.append(f'{product["name"]} is sold at {", ".join(str(s) for s in product["spacing_options"])} in spacing, not {spacing:g} in.')
    manual = [k for k in kinds if not KINDS[k]["detect"]]
    return {
        "id": uuid.uuid4().hex[:8], "text": r["text"], "kinds": kinds, "edges": r["edges"] if "roof" in kinds else [],
        "sides": r["sides"], "manual": manual,
        "product_id": pid, "product_name": product["name"],
        "pattern": pattern or [], "spacing_in": spacing if spacing is not None else product["spacing_in"],
        "twinkle": bool(light.get("twinkle", product.get("twinkle", False))),
        "assumed": assumed, "problems": problems,
    }


def catalogue() -> dict:
    return {"products": PRODUCTS, "colours": COLOURS, "kinds": KINDS, "edges": EDGES, "default_product": DEFAULT_PRODUCT}
