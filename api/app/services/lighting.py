"""From what an instruction says to what the catalogue can supply.

`resolve()` turns each parsed request into the product it names — or the one the place takes
by default — with a colour pattern, spacing and twinkle as `LightingSettings`, and says
plainly what was assumed and what could not be matched. Nothing here is final: the result is
shown for confirmation, and the design is only made from what the user confirms.
"""

from app.schemas import LightingSettings, Pattern, PatternStep
from app.services import intent, patterns
from app.services.products import COLOURS, SOCKET_CATEGORIES

CATEGORY_NAMES = {
    "c9": "C9", "c7": "C7", "mini": "mini lights", "icicle": "icicle lights",
    "wreath": "a wreath", "garland": "garland",
}


def colour_names(keys: list[str]) -> str:
    return ", ".join(COLOURS[key][0].lower() for key in keys) or "no colour"


def _pattern(lighting: intent.Lighting) -> dict:
    colours = lighting.colours or ["warm_white"]
    counts = lighting.counts or [1] * len(colours)
    steps = [PatternStep(colour=colour, count=count) for colour, count in zip(colours, counts)]
    return Pattern(steps=steps, unit="turn" if lighting.per_turn else "bulb").model_dump()


def _strand(lighting: intent.Lighting, candidates: list) -> tuple[object | None, list[str], list[str]]:
    """The manufactured strand matching the colours asked for; None when the catalogue has none."""
    assumed: list[str] = []
    wanted = lighting.colours or ["warm_white"]
    if not lighting.colours:
        assumed.append("warm white")
    same = [p for p in candidates if list(p.colours) == wanted] or \
           [p for p in candidates if set(p.colours) == set(wanted)]
    if not same:
        return None, assumed, [f"No {CATEGORY_NAMES[lighting.category]} in {colour_names(wanted)} in the catalogue"]
    if lighting.size_in is not None and any(p.diameter_in for p in same):
        nearest = min(same, key=lambda p: abs((p.diameter_in or 0) - lighting.size_in))
        if nearest.diameter_in != lighting.size_in:
            assumed.append(f"nearest size is {nearest.diameter_in:g} in")
        same = [nearest] + [p for p in same if p is not nearest]
    if lighting.twinkle is not None:
        same = [p for p in same if p.twinkle == lighting.twinkle] or same
    return same[0], assumed, []


def resolve_request(request: intent.Request, products: list) -> dict:
    """One request against the catalogue: product, settings, and what was assumed or unmatched."""
    lighting = request.lighting
    assumed = list(lighting.assumed)
    problems = [f'"{word}" is not a colour we carry' for word in lighting.unread]
    category = lighting.category
    if category is None:
        defaults = {intent.DEFAULT_CATEGORY.get(kind) for kind in request.kinds} - {None}
        if len(defaults) == 1:
            category = defaults.pop()
            assumed.append(CATEGORY_NAMES[category])
        # A multi-colour sequence or a candy-cane wrap is a socket-line pattern: strands cannot do it.
        if category not in SOCKET_CATEGORIES and (lighting.per_turn or any(c > 1 for c in lighting.counts)):
            category = "c9"
            assumed.append("C9 for the colour sequence")
    lighting = intent.Lighting(**{**vars(lighting), "category": category})

    candidates = [p for p in products if p.category == category] if category else []
    product = None
    settings: LightingSettings | None = None
    if category is None:
        problems.append("Say which lights go here (C9, C7, mini lights, icicles, wreath or garland)")
    elif not candidates:
        problems.append(f"No {CATEGORY_NAMES[category]} in the catalogue")
    elif category in SOCKET_CATEGORIES:
        if lighting.twinkle is not None:
            candidates = [p for p in candidates if p.twinkle == lighting.twinkle] or candidates
        product = candidates[0]
        if not lighting.colours:
            assumed.append("warm white")
        spacing = None
        if lighting.spacing_in is not None:
            options = product.spacing_options_in or [product.spacing_in]
            if lighting.spacing_in in options:
                spacing = lighting.spacing_in
            else:
                sizes = ", ".join(f"{option:g}" for option in options)
                problems.append(f"{product.name} comes at {sizes} inch spacing, not {lighting.spacing_in:g}")
        twinkle = lighting.twinkle if lighting.twinkle is not None and lighting.twinkle != product.twinkle else None
        settings = LightingSettings(product_id=product.id, pattern=_pattern(lighting), spacing_in=spacing, twinkle=twinkle)
    else:
        product, more_assumed, unmatched = _strand(lighting, candidates)
        assumed += more_assumed
        problems += unmatched
        if product is not None:
            if lighting.spacing_in is not None and lighting.spacing_in != product.spacing_in:
                problems.append(f"{product.name} has a fixed {product.spacing_in:g} inch spacing")
            twinkle = lighting.twinkle if lighting.twinkle is not None and lighting.twinkle != product.twinkle else None
            settings = LightingSettings(product_id=product.id, twinkle=twinkle)

    if product is not None and settings is not None:
        for kind in request.kinds:
            for problem in patterns.problems(settings, product, kind):
                if problem not in problems:
                    problems.append(problem)

    return {
        "text": request.text,
        "kinds": request.kinds,
        "sides": request.sides,
        "edges": request.edges,
        "manual": request.manual,
        "lighting": {
            "category": category, "colours": lighting.colours, "counts": lighting.counts,
            "spacing_in": lighting.spacing_in, "size_in": lighting.size_in,
            "twinkle": lighting.twinkle, "per_turn": lighting.per_turn,
        },
        "product_id": str(product.id) if product is not None else None,
        "product_name": product.name if product is not None else None,
        "settings": settings.model_dump(mode="json") if settings is not None else None,
        "assumed": assumed,
        "problems": problems,
        "choices": [{"id": str(p.id), "name": p.name} for p in candidates[:12]],
    }


def resolve(spec: intent.Specification, products: list) -> list[dict]:
    return [resolve_request(request, products) for request in spec.requests]


def interpretation(prompt: str, products: list) -> dict:
    """Everything `/detect` and the confirmation step read from an instruction, JSON-ready."""
    spec = intent.specification(prompt)
    return {
        "kinds": spec.intent.kinds,
        "sides": spec.intent.sides,
        "edges": spec.intent.edges,
        "matched": spec.intent.matched,
        "requests": resolve(spec, products),
    }


def matches(request: dict, region) -> bool:
    """Whether a confirmed request lights this region: a visible target of one of its kinds, on
    a side it named, and for rooflines an edge it named. An edge nobody has identified yet is
    never lit on the strength of a guess."""
    if region.role != "target" or not region.visible or region.kind not in request["kinds"]:
        return False
    if request.get("sides") and region.side and region.side not in request["sides"]:
        return False
    if region.kind == "roof" and request.get("edges"):
        return region.edge in request["edges"]
    return region.kind != "roof" or region.edge not in ("junction", "unknown")
