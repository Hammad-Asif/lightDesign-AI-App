"""Score both instruction readers against the same table of instructions.

    python tests/parse_eval.py             # rules only, free
    python tests/parse_eval.py --llm       # both, and the LLM half costs a few cents

Each case lists only the fields it cares about; anything not listed is not scored, so a case can
say "this instruction must reach palm_trunk" without also pinning a spacing nobody mentioned.
Scoring is field by field rather than all-or-nothing: a reading that gets the place and the
colours right but misses the spacing should not score the same as one that got nothing.
"""
import argparse
import os
import sys
from pathlib import Path

API = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API))
os.environ.setdefault("SEGMENTATION_PROVIDER", "mock")

from app.config import get_settings          # noqa: E402
from app.services import intent, intent_llm  # noqa: E402

# kinds / sides / edges are sets: order does not matter. category, spacing, size, twinkle,
# per_turn and colours are exact. colours keeps its spoken order, so it is a list.
#
# The third element of a case is a tolerance: extra set members that are defensible rather than
# wrong, so a reader is not marked down for a reading the rules make on purpose. It is deliberately
# short -- every entry in it is a judgement the eval is declining to make.
CASES: list[tuple[str, dict, dict]] = [
    # The client's own example, from docs/M2_DAY5_UPDATE.md.
    ("red and green C9 on the front roofline at 15 inch spacing, warm white minis on the palm "
     "trunks, 24 inch wreath on the front door",
     {"kinds": {"roof", "palm_trunk", "wreath"}, "sides": {"front"},
      "edges": {"eave", "rake", "hip"}, "spacing_in": 15.0}, {}),
    ("C7 cool white along the gutters only",
     {"kinds": {"roof"}, "edges": {"eave"}, "category": "c7", "colours": ["cool_white"]}, {}),
    ("two red one white C9s on the peaks",
     {"kinds": {"roof"}, "edges": {"ridge"}, "category": "c9", "counts": [2, 1]}, {}),
    ("wrap the two tall palms by the driveway in warm white",
     {"kinds": {"palm_trunk"}, "colours": ["warm_white"]}, {}),
    ("red C9 on the garage roofline",
     {"kinds": {"roof"}, "category": "c9", "colours": ["red"]}, {}),
    ("the fronds of the palms in green mini lights",
     {"kinds": {"palm_frond"}, "category": "mini", "colours": ["green"]}, {}),
    ("C9 on the left side roofline",
     {"kinds": {"roof"}, "sides": {"left"}, "category": "c9"}, {}),
    ("multicolor C9 on the roofline, twinkling",
     {"kinds": {"roof"}, "category": "c9", "twinkle": True}, {}),
    # Naming either sloped edge yields both on purpose (intent.SLOPED_EDGES), so rake is allowed.
    ("warm white on the ridge and hips",
     {"kinds": {"roof"}, "edges": {"ridge", "hip"}, "colours": ["warm_white"]}, {"edges": {"rake"}}),
    ("make the house look festive", {"kinds": set()}, {}),
    ("C9 on the roofline",
     {"kinds": {"roof"}, "edges": {"eave", "rake", "hip"}, "category": "c9"}, {}),
    ("mini lights in the hedges", {"kinds": {"hedge"}, "category": "mini"}, {}),
    ("warm white minis on the shrubs and topiaries",
     {"kinds": {"shrub", "topiary"}, "category": "mini", "colours": ["warm_white"]}, {}),
    ("candy cane wrap on the palm trunks", {"kinds": {"palm_trunk"}, "per_turn": True}, {}),
    ("C9 on the front and right roofline", {"kinds": {"roof"}, "sides": {"front", "right"}}, {}),
    ("24 inch wreath on the front door", {"kinds": {"wreath"}, "size_in": 24.0}, {}),
    ("icicle lights along the eaves",
     {"kinds": {"roof"}, "edges": {"eave"}, "category": "icicle"}, {}),
    # A palm named with no part lights all three zones on purpose, for the salesperson to trim.
    ("light the palms", {"kinds": {"palm_trunk", "palm_collar", "palm_frond"}}, {}),
    ("blue mini lights in the hedges",
     {"kinds": {"hedge"}, "category": "mini", "colours": ["blue"]}, {}),
    ("C9 at 12 inch spacing on the roofline", {"kinds": {"roof"}, "spacing_in": 12.0}, {}),
    ("garland on the columns", {"kinds": {"column"}, "category": "garland"}, {}),
    ("red and green alternating C9 on the roofline",
     {"kinds": {"roof"}, "category": "c9", "colours": ["red", "green"]}, {}),
    ("tree canopy in warm white minis",
     {"kinds": {"tree_canopy"}, "category": "mini", "colours": ["warm_white"]}, {}),
    ("C9 on the roofline and mini lights on the palm trunks",
     {"kinds": {"roof", "palm_trunk"}}, {}),
    # The rest are the cases the two readers should part company on: a thing named as a landmark
    # rather than a place to light, and an instruction that says what NOT to light.
    ("nothing on the garage", {"kinds": set()}, {}),
    ("C9 on the roofline but skip the garage", {"kinds": {"roof"}}, {}),
    ("warm white minis on the bushes in front of the windows",
     {"kinds": {"shrub"}, "category": "mini"}, {}),
    ("leave the ridge dark, C9 on the rest of the roofline",
     {"kinds": {"roof"}, "edges": {"eave", "rake", "hip"}}, {}),
    ("a wreath on the front door and nothing else", {"kinds": {"wreath"}}, {}),
    ("same as last year: red and green C9 on the roof",
     {"kinds": {"roof"}, "category": "c9", "colours": ["red", "green"]}, {}),
]


def reading(spec) -> dict:
    """The union across requests, in the shape the cases are written in."""
    out = {
        "kinds": {k for r in spec.requests for k in r.kinds},
        "sides": {s for r in spec.requests for s in r.sides},
        "edges": {e for r in spec.requests for e in r.edges},
    }
    for name in ("category", "spacing_in", "size_in", "twinkle"):
        found = [getattr(r.lighting, name) for r in spec.requests if getattr(r.lighting, name) is not None]
        out[name] = found[0] if found else None
    out["per_turn"] = any(r.lighting.per_turn for r in spec.requests)
    out["colours"] = next((r.lighting.colours for r in spec.requests if r.lighting.colours), [])
    out["counts"] = next((r.lighting.counts for r in spec.requests if r.lighting.counts), [])
    return out


def score(expected: dict, got: dict, allowed: dict) -> tuple[int, int, list[str]]:
    right, total, missed = 0, 0, []
    for field, want in expected.items():
        total += 1
        have = got.get(field)
        if isinstance(want, set):
            # Tolerated extras count as neither right nor wrong: they are removed before comparing.
            ok = set(have or set()) - allowed.get(field, set()) == want
        elif isinstance(want, list):
            ok = list(have or []) == want
        else:
            ok = have == want
        if ok:
            right += 1
        else:
            missed.append(f"{field}: wanted {want!r}, got {have!r}")
    return right, total, missed


def builtin_products(settings) -> list:
    """The shipped library, so the model is shown the same products the app would show it."""
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import Session

    from app.models import Product

    engine = create_engine(settings.database_url)
    with Session(engine) as db:
        return list(db.scalars(select(Product).where(Product.owner_id.is_(None))
                               .order_by(Product.category, Product.name)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="also score the model reader (costs money)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    settings = get_settings()
    library = builtin_products(settings)
    # read() answers with the rules whenever the model is unreachable or unreadable, so an LLM
    # score could quietly be a rules score. Count the fallbacks and say so under the table.
    fell_back: list[str] = []
    spent = [0.0]

    def by_llm(text: str):
        spec, note = intent_llm.read(text, library, settings)
        spent[0] += note.get("cost_usd") or 0.0
        if note["reader"] != "llm":
            fell_back.append(f"{text[:40]}: {note['why']}")
        return spec

    readers = [("rules", intent.specification)]
    if args.llm:
        settings.instruction_parser = "llm"
        readers.append(("llm", by_llm))

    results: dict[str, tuple[int, int]] = {}
    for name, run in readers:
        right = total = 0
        print(f"\n=== {name} ===")
        for text, expected, allowed in CASES:
            got = reading(run(text))
            r, t, missed = score(expected, got, allowed)
            right += r
            total += t
            flag = " " if r == t else "x"
            print(f" {flag} {r}/{t}  {text[:58]}")
            if missed and (args.verbose or r != t):
                for line in missed:
                    print(f"        {line}")
        results[name] = (right, total)

    print(f"\n{'reader':8s} {'fields right':>13s} {'of':>4s} {'score':>7s}")
    for name, (right, total) in results.items():
        print(f"{name:8s} {right:13d} {total:4d} {100 * right / max(total, 1):6.1f}%")
    if args.llm:
        print(f"\nthe model's half of this run cost ${spent[0]:.4f}")
    if fell_back:
        print(f"\nthe model fell back to the rules on {len(fell_back)} case(s):")
        for line in fell_back:
            print(f"  {line}")


if __name__ == "__main__":
    main()
