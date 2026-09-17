"""The lighting product library: what a product is, and what each kind must describe.

A product is the physical thing installed — a C9 socket line, a mini-light strand, a wreath —
described with the attributes an estimate and the placement engine need. Colour, spacing and
twinkle are data here and render parameters later; nothing about a product is baked into pixels.

Sizes are in inches and lengths in feet, the units the client quotes in.
"""

# The colours a bulb or strand can be. Hex values are what the renderer will draw; the palette
# itself is still to be confirmed with the client.
COLOURS: dict[str, tuple[str, str]] = {
    "warm_white": ("Warm white", "#ffb86b"),
    "pure_white": ("Pure white", "#fff4e0"),
    "cool_white": ("Cool white", "#e8f1ff"),
    "red": ("Red", "#ff2a2a"),
    "green": ("Green", "#1fd65f"),
    "blue": ("Blue", "#2f6bff"),
    "amber": ("Amber", "#ffb000"),
    "orange": ("Orange", "#ff7a1a"),
    "purple": ("Purple", "#9b4dff"),
    "pink": ("Pink", "#ff5fb0"),
}

# C9 and C7 are socket lines: bulbs of any colour go into the sockets, so their colours come from
# the design's pattern and the product carries none. Every other kind is a manufactured item
# whose colours are fixed — a warm white/cool white mini strand is a product of its own, with that
# sequence as data, not an effect applied afterwards.
SOCKET_CATEGORIES = {"c9", "c7"}

REQUIRED: dict[str, set[str]] = {
    "c9": {"bulb_diameter_in", "spacing_in"},
    "c7": {"bulb_diameter_in", "spacing_in"},
    "mini": {"bulb_diameter_in", "spacing_in", "strand_length_ft", "colours"},
    # Icicle spacing is between drops along the top wire; drops_in repeats along the strand.
    "icicle": {"bulb_diameter_in", "spacing_in", "strand_length_ft", "drops_in", "colours"},
    "wreath": {"diameter_in", "colours"},
    "garland": {"diameter_in", "strand_length_ft", "colours"},
}


def problems(product: dict) -> list[str]:
    """Everything wrong with a product description, as sentences; empty when it is valid."""
    category = product["category"]
    found = [
        f"{category} needs {field.replace('_', ' ')}"
        for field in sorted(REQUIRED[category])
        if product.get(field) in (None, [], "")
    ]
    if category in SOCKET_CATEGORIES and product.get("colours"):
        found.append(f"{category} bulbs take their colours from the design, so leave colours empty")
    if product.get("drops_in") and category != "icicle":
        found.append("only icicle lights have drops")
    options = product.get("spacing_options_in") or []
    if options and category not in SOCKET_CATEGORIES:
        found.append(f"{category} has a fixed spacing; spacing options are for socket lines")
    if options and product.get("spacing_in") not in options:
        found.append("the default spacing must be one of the spacing options")
    unknown = [c for c in product.get("colours") or [] if c not in COLOURS]
    if unknown:
        found.append(f"unknown colours: {', '.join(unknown)}")
    return found
