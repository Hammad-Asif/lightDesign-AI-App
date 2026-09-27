"""Reading an instruction with a language model, behind the rules parser's own interface.

`intent.specification()` reads an instruction with hand-written patterns. It is exact on the
phrasings it knows and silent on everything else: a clause it has no pattern for simply does not
appear. This reads the same instruction with a model and returns the same `Specification`, so
`lighting.resolve_request`, the confirmation screen and `/detect` are unchanged and detection
receives exactly the Intent fields it does today.

Three things keep it honest:

  * The vocabulary is built from live data, not written into a prompt by hand -- the place kinds
    intent.py knows, the edges, the sides, and the customer's own product library and palette.
    A product that is not in the library cannot be named, because the model is never told it.
  * Everything the model returns is validated against that vocabulary before it is used. An
    unknown kind, product or colour is dropped and reported, never passed on.
  * If there is no key, the call fails, or the answer is unreadable while the rules do find
    places, the rules answer is used instead. The reply always says which reader was used and
    why, so a fallback is visible rather than silent.

Only the instruction text is sent. Never the photo, never the key in a log.
"""
from __future__ import annotations

import json
import time

import httpx
from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings
from app.services import intent as rules
from app.services import products as catalogue

API = "https://openrouter.ai/api/v1/chat/completions"

# (prompt, library fingerprint) -> (Specification, note). In memory only: a reading is cheap to
# repeat and must never outlive a change to the library it was resolved against.
_CACHE: dict[tuple[str, str], tuple[rules.Specification, dict]] = {}
CACHE_MAX = 256

# Sent to OpenRouter verbatim as the model's structured-output contract (strict json_schema).
# Kept as a plain dict, not generated from the models below, for the same reason as roofline_v3:
# the provider's strict-mode shape need not match how the answer is validated once it is back.
SCHEMA = {
    "type": "object",
    "properties": {
        "requests": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "kinds": {"type": "array", "items": {"type": "string"}},
                "sides": {"type": "array", "items": {"type": "string"}},
                "edges": {"type": "array", "items": {"type": "string"}},
                "category": {"type": ["string", "null"]},
                "colours": {"type": "array", "items": {"type": "string"}},
                "counts": {"type": "array", "items": {"type": "integer"}},
                "spacing_in": {"type": ["number", "null"]},
                "size_in": {"type": ["number", "null"]},
                "twinkle": {"type": ["boolean", "null"]},
                "per_turn": {"type": "boolean"},
            },
            "required": ["text", "kinds", "sides", "edges", "category", "colours",
                         "counts", "spacing_in", "size_in", "twinkle", "per_turn"],
            "additionalProperties": False,
        }},
        "unclear": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["requests", "unclear"],
    "additionalProperties": False,
}


class _RawRequest(BaseModel):
    """One item of the model's answer, before it is read against the vocabulary. Mirrors
    SCHEMA's item shape, so shape and type errors are caught here rather than by hand below."""

    model_config = ConfigDict(extra="forbid")

    text: str
    kinds: list[str] = Field(default_factory=list)
    sides: list[str] = Field(default_factory=list)
    edges: list[str] = Field(default_factory=list)
    category: str | None = None
    colours: list[str] = Field(default_factory=list)
    counts: list[int] = Field(default_factory=list)
    spacing_in: float | None = None
    size_in: float | None = None
    twinkle: bool | None = None
    per_turn: bool = False


class _RawAnswer(BaseModel):
    """The model's whole answer, before it is read against the vocabulary. Mirrors SCHEMA's
    top level."""

    model_config = ConfigDict(extra="forbid")

    requests: list[_RawRequest] = Field(default_factory=list)
    unclear: list[str] = Field(default_factory=list)


def library_fingerprint(products: list) -> str:
    """Changes whenever anything a reading depends on changes."""
    # str(p.id): a product id is a UUID, which json.dumps will not encode.
    return json.dumps([[str(p.id), p.name, p.category, p.spacing_in,
                        sorted(p.spacing_options_in or []), sorted(p.colours or [])]
                       for p in sorted(products, key=lambda p: str(p.id))], sort_keys=True)


def system_prompt(products: list) -> str:
    lines = []
    for p in products:
        spacings = p.spacing_options_in or ([p.spacing_in] if p.spacing_in else [])
        lines.append(f"  - {p.id} | {p.name} | category {p.category}"
                     f" | spacings sold: {', '.join(f'{s:g}' for s in spacings) or 'n/a'}"
                     f" | colours made: {', '.join(p.colours or []) or 'n/a'}")
    return f"""You read a holiday-lighting instruction into structured requests. Return JSON only.

PLACES (kind ids you may use; use nothing else):
  detected automatically: {', '.join(rules.KIND_ORDER)}
  drawn by hand:          {', '.join(rules.MANUAL_PATTERNS)}

SIDES: {', '.join(rules.SIDES)}   (omit when the instruction does not say)

EDGES (roof only): {', '.join(rules.EDGE_PATTERNS)}, valley
  eave = the bottom edge, the gutter line, the fascia
  rake = the sloping edge of a gable end
  hip  = the sloping edge where two roof faces meet and fall away
  ridge = the top edge, the peak, where two faces meet and rise together
  valley = the inward gutter where two roof faces meet
  "roofline" or "roof" with no edge named means exactly: {', '.join(rules.OUTLINE_EDGES)}.
  ridge and valley are included ONLY when the instruction names them.
  When the instruction picks edges ("gutters only", "on the peaks"), return just those.

CATEGORIES: {', '.join(rules.CATEGORY_PATTERNS)}
PALETTE (colour keys; use nothing else): {', '.join(catalogue.COLOURS)}

THE CUSTOMER'S PRODUCT LIBRARY:
{chr(10).join(lines) or '  (empty)'}

RULES
  1. Return only what the instruction asks for. Do not add places, colours or products.
  2. `text` is that request's own words, quoted from the instruction.
  3. Never invent a product or a colour. If what is asked for is not in the library or the
     palette above, still return what was said and add a note to `unclear`.
  4. A clause with lighting but no place refines the request before it. If it says "all" or
     "everything", it refines every request so far.
  5. "palms" with no part named means all three palm zones (palm_trunk, palm_collar,
     palm_frond), for the salesperson to trim. A named part means that part alone, and
     "wrap" names the trunk: you wrap a trunk, not a frond.
  6. `colours` are palette keys in the order spoken. `counts` is how many of each, same order
     ("two red one white" -> colours [red, white-ish key], counts [2, 1]). Leave counts empty
     when no number is spoken.
  7. `per_turn` is true only for a candy-cane wrap, where the colour changes each turn rather
     than each bulb.
  8. Some things are named to say WHERE something is, not to be lit: "the palms by the
     driveway", "bushes in front of the windows", "the garage roofline". Light the thing
     asked for, not the landmark.
  9. Honour what the instruction says NOT to light: "nothing on the garage", "skip the
     garage", "leave the ridge dark". Return no request for it, and no edge for it.
 10. A wreath is its own place: "wreath" is in the list above, so a wreath on the front door
     is the place `wreath`, not `door`. There is NO garland place, so a garland is lit on
     whatever it wraps -- "garland on the columns" is the place `column`, product garland.
 11. Anything you could not read confidently goes in `unclear`, one short sentence each.
"""


def _lighting(item: _RawRequest, note: dict) -> rules.Lighting:
    palette = set(catalogue.COLOURS)
    colours, counts = [], []
    for index, colour in enumerate(item.colours):
        key = colour.strip().lower().replace(" ", "_")
        if key not in palette:
            note["dropped"].append(f"colour {colour!r} is not in the palette")
            continue
        colours.append(key)
        counts.append(item.counts[index] if index < len(item.counts) else 1)
    category = item.category
    if category is not None:
        category = category.strip().lower()
        if category not in rules.CATEGORY_PATTERNS:
            note["dropped"].append(f"product type {category!r} is not one we sell")
            category = None
    return rules.Lighting(
        category=category,
        colours=colours,
        counts=counts if any(c != 1 for c in counts) else [],
        spacing_in=item.spacing_in,
        size_in=item.size_in,
        twinkle=item.twinkle,
        per_turn=item.per_turn,
    )


def to_specification(answer: _RawAnswer, note: dict) -> rules.Specification:
    """The model's answer as the same Specification the rules parser produces."""
    known = set(rules.KIND_ORDER) | set(rules.MANUAL_PATTERNS)
    manual_kinds = set(rules.MANUAL_PATTERNS)
    edges_known = set(rules.EDGE_PATTERNS) | {"valley"}
    requests: list[rules.Request] = []
    for item in answer.requests:
        kinds, manual = [], []
        for kind in item.kinds:
            key = kind.strip().lower()
            if key not in known:
                # A mailbox or a lamp post: real, but nothing detection can find and nothing
                # intent.py has a kind for. Reported, never sent to detection.
                note["by_hand"].append(kind)
                continue
            kinds.append(key)
            if key in manual_kinds:
                manual.append(key)
        sides = [s for s in item.sides if s in rules.SIDES]
        edges = [e for e in item.edges if e in edges_known]
        if not kinds:
            continue
        requests.append(rules.Request(
            text=item.text.strip(),
            kinds=kinds, sides=sides, edges=edges, manual=manual,
            lighting=_lighting(item, note),
        ))
    ordered = [k for k in rules.KIND_ORDER if any(k in r.kinds for r in requests)]
    ordered += [k for r in requests for k in r.kinds if k not in ordered]
    union = rules.Intent(
        kinds=ordered,
        sides=[s for s in rules.SIDES if any(s in r.sides for r in requests)],
        edges=sorted({e for r in requests for e in r.edges}),
        matched=bool(requests),
    )
    return rules.Specification(requests=requests, intent=union)


def read(prompt: str, products: list, settings: Settings) -> tuple[rules.Specification, dict]:
    """The instruction as a Specification, plus a note saying which reader produced it."""
    fallback = rules.specification(prompt)

    def with_rules(why: str) -> tuple[rules.Specification, dict]:
        return fallback, {"reader": "rules", "why": why, "unclear": [], "dropped": [], "by_hand": []}

    if settings.instruction_parser != "llm":
        return with_rules("the fixed rules are configured")
    if not settings.openrouter_api_key.strip():
        return with_rules("no OpenRouter key is configured")

    key = (prompt.strip(), library_fingerprint(products))
    if key in _CACHE:
        return _CACHE[key]

    note = {"reader": "llm", "model": settings.parser_model, "why": "",
            "unclear": [], "dropped": [], "by_hand": []}
    started = time.perf_counter()
    try:
        response = httpx.post(API, json={
            "model": settings.parser_model,
            "messages": [
                {"role": "system", "content": system_prompt(products)},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "lighting_requests", "strict": True, "schema": SCHEMA}},
            # No temperature: this model does not accept one, and require_parameters makes
            # OpenRouter refuse the call (404) rather than quietly drop the JSON schema.
            # A cap is required, not tidiness: with none, OpenRouter reserves the model's full
            # 65k default against the balance and refuses the call (402) when the remaining
            # credit will not cover it. A reading of a sentence needs a small fraction of this.
            "max_completion_tokens": 4000,
            "provider": {"require_parameters": True}, "stream": False,
        }, headers={"Authorization": f"Bearer {settings.openrouter_api_key}"},
            timeout=settings.parser_timeout)
        response.raise_for_status()
        raw = response.json()
        if raw.get("error"):
            raise ValueError("provider returned an error")
        answer = _RawAnswer.model_validate(json.loads(raw["choices"][0]["message"]["content"]))
        spec = to_specification(answer, note)
        note["unclear"] = list(answer.unclear)
    except httpx.TimeoutException:
        return with_rules("the language model timed out")
    except httpx.HTTPStatusError as exc:
        return with_rules(f"the language model refused the request (HTTP {exc.response.status_code})")
    except httpx.RequestError:
        return with_rules("the language model could not be reached")
    except (ValueError, KeyError, TypeError, IndexError):
        return with_rules("the language model's answer could not be read")

    # An empty reading can mean either "there is nothing here to light" or "I could not read
    # this". The model says which: an instruction it could not follow leaves a note in
    # `unclear`. A confident empty reading is kept -- "nothing on the garage" is correctly
    # empty, and the rules, which cannot hear a refusal, would answer "garage".
    if not spec.requests and fallback.requests and note["unclear"]:
        return with_rules("the language model could not read the instruction")

    usage = raw.get("usage") or {}
    note["cost_usd"] = usage.get("cost")
    note["why"] = f"read in {time.perf_counter() - started:.1f}s"
    if len(_CACHE) >= CACHE_MAX:
        _CACHE.clear()
    _CACHE[key] = (spec, note)
    return spec, note


def sentence(note: dict) -> str:
    """The one line the confirmation screen shows next to the reading."""
    if note["reader"] == "rules":
        return f"Read with the fixed rules: {note['why']}." if note["why"] else "Read with the fixed rules."
    model = note.get("model")
    return f"Read by {model}." if model else "Read with the language model."
