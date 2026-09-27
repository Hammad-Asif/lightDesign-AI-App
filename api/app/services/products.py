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


# These sentences are shown to the salesperson as they are, on the product form and wherever a
# product is chosen, so they name the thing the way the form labels it. The phrasing is
# imperative to sidestep the agreement problem in "mini lights needs a length".
CATEGORY_NAMES: dict[str, str] = {
    "c9": "the C9", "c7": "the C7", "mini": "the mini lights",
    "icicle": "the icicle lights", "wreath": "the wreath", "garland": "the garland",
}

FIELD_NAMES: dict[str, str] = {
    "bulb_diameter_in": "a bulb size",
    "spacing_in": "a spacing",
    "strand_length_ft": "a length",
    "diameter_in": "a diameter",
    "drops_in": "its drop lengths",
    "colours": "at least one colour",
}


def problems(product: dict) -> list[str]:
    """Everything wrong with a product description, as sentences; empty when it is valid."""
    category = product["category"]
    named = CATEGORY_NAMES[category]
    found = []
    for field in sorted(REQUIRED[category]):
        if product.get(field) not in (None, [], ""):
            continue
        # A garland's diameter is how thick it is, not how wide across like a wreath.
        wanted = "a thickness" if field == "diameter_in" and category == "garland" \
            else FIELD_NAMES[field]
        found.append(f"Give {named} {wanted}")
    if category in SOCKET_CATEGORIES and product.get("colours"):
        found.append(f"Bulbs for {named} are chosen per design, so leave the colours empty")
    if product.get("drops_in") and category != "icicle":
        found.append("Only icicle lights have drops")
    options = product.get("spacing_options_in") or []
    if options and category not in SOCKET_CATEGORIES:
        found.append(f"Spacing options are for socket lines; {named} has one fixed spacing")
    if options and product.get("spacing_in") not in options:
        found.append("The default spacing has to be one of the spacings it is sold at")
    unknown = [c for c in product.get("colours") or [] if c not in COLOURS]
    if unknown:
        found.append(f"We do not carry {', '.join(unknown)}")
    return found
