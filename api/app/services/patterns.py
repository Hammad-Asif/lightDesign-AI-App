"""Colour patterns: which colour each position along a path gets.

A position is a bulb along a line, or one full turn of a wrap around a trunk. Placement (Week 2)
decides where the positions are; this decides their colours, and checks that a region's lighting
settings make sense for the product and the region they are applied to.
"""

from app.schemas import LightingSettings, Pattern, PatternStep, Wrap

# Zones a strand is wrapped around rather than laid along.
WRAP_KINDS = {"palm_trunk", "palm_collar"}


def cycle(pattern: Pattern) -> list[str]:
    """One repeat of the pattern, one colour per position: red x2, white -> red, red, white."""
    return [step.colour for step in pattern.steps for _ in range(step.count)]


def colours(pattern: Pattern, count: int) -> list[str]:
    """The colour of each of `count` positions, listed in the path's own order."""
    unit = cycle(pattern)
    from_start = [unit[(index + pattern.offset) % len(unit)] for index in range(count)]
    # Reversed, the sequence begins at the far end: position count-1 takes the first colour.
    return from_start[::-1] if pattern.direction == "reverse" else from_start


def resolve(
    settings: LightingSettings, product, region_kind: str
) -> tuple[Pattern, float | None, bool, Wrap | None]:
    """The pattern, spacing, twinkle and wrap a region is actually lit with, defaults filled in."""
    if product.colours:
        pattern = Pattern(steps=[PatternStep(colour=colour) for colour in product.colours])
    else:
        pattern = settings.pattern
    spacing = settings.spacing_in if settings.spacing_in is not None else product.spacing_in
    twinkle = settings.twinkle if settings.twinkle is not None else product.twinkle
    wrap = settings.wrap or (Wrap() if region_kind in WRAP_KINDS else None)
    return pattern, spacing, twinkle, wrap


def problems(settings: LightingSettings, product, region_kind: str) -> list[str]:
    """Everything wrong with applying `settings` and `product` to a region; empty when valid."""
    found = []
    if product.colours and settings.pattern is not None:
        found.append(f"{product.name} has its own colours; choose another product to change them")
    if not product.colours and settings.pattern is None:
        found.append(f"{product.name} needs a colour pattern")

    options = product.spacing_options_in or []
    if settings.spacing_in is not None:
        if options and settings.spacing_in not in options:
            sizes = ", ".join(f"{option:g}" for option in options)
            found.append(f"{product.name} comes at {sizes} inch spacing")
        elif not options and settings.spacing_in != product.spacing_in:
            found.append(f"{product.name} has a fixed spacing")

    wraps = region_kind in WRAP_KINDS
    if settings.wrap is not None and not wraps:
        found.append("only palm trunks and collars are wrapped")
    if settings.pattern is not None and settings.pattern.unit == "turn" and not wraps:
        found.append("a pattern that changes colour each turn needs a wrapped zone")
    if (product.category == "wreath") != (region_kind == "wreath"):
        found.append("wreath products go on wreath regions, and wreath regions take wreaths")
    return found
