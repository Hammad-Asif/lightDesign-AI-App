from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator

from app.services import products

RegionKind = Literal[
    "roof",
    "driveway_edge",
    "walkway_edge",
    "fence",
    "column",
    "window",
    "door",
    "garage_door",
    "wreath",
    "hedge",
    "shrub",
    "topiary",
    "palm_trunk",
    "palm_collar",
    "palm_frond",
    "tree_canopy",
    "obstruction",
    "custom",
]
GeometryType = Literal["line", "area"]
# A blocker is detected to explain what hides a line, not to be lit.
RegionRole = Literal["target", "blocker"]
Side = Literal["front", "left", "right"]
# "rake" is the sloping edge of a gable end — lit like an eave, but not one, so it is not
# folded into "ridge". "unknown" is emitted when adjacency cannot decide, rather than
# fabricating a label the UI would present as fact.
# "hip" is a sloping line where two roof planes meet; it is lit as part of the outline.
# "junction" is where a lower roof runs into the wall above it: identified, and never lit.
Edge = Literal["eave", "rake", "hip", "ridge", "valley", "junction", "unknown"]
MaskingMode = Literal["manual", "automatic"]


class Region(BaseModel):
    id: str
    kind: RegionKind
    role: RegionRole = "target"
    label: str
    geometry: GeometryType
    # inferred: a partition-mode connector drawn across an occluded stretch from no pixels.
    source: Literal["auto", "manual", "inferred"]
    color: str
    score: float = 1.0
    visible: bool = True
    side: Side | None = None
    edge: Edge | None = None
    polyline: list[list[int]] | None = None
    mask_url: str | None = None
    # Lines only: [start, end] pixel distances along the polyline that pass behind an object.
    hidden: list[list[float]] | None = None


class Interpreted(BaseModel):
    kinds: list[RegionKind] = []
    sides: list[Side] = []
    edges: list[Edge] = []
    matched: bool = False
    # The instruction clause by clause, each resolved against the catalogue (services/lighting.py):
    # where the lights go, which product, pattern and spacing, what was assumed, what did not match.
    requests: list[dict] | None = None
    # Partition mode: per detected plane, pitch as seen, calibrated when a focal length is
    # known, and the nearest standard pitch with which of the two it came from.
    pitch: dict | None = None
    # Partition mode: the detected roof planes as closed polygons in photo pixels, each with the
    # colour it has in the partition, so the editor can fill them under the lines.
    planes: list[dict] | None = None
    # Direct roof detector provenance and measured billing/latency; absent on the SAM path.
    detection_run: dict | None = None
    # Which reader produced the requests above -- the fixed rules or the language model -- and
    # anything it could not read. A silent fallback would otherwise look like a clean reading.
    read_with: dict | None = None
    # TARGET_HOUSE_ONLY: every roof plane detection returned, as {"outline": polygons in photo
    # pixels, "keep": on the target house, "score"}, plus the instruction's side and edge filter
    # the lines were drawn with. The editor's House tool taps a plane in or out of the house.
    target: dict | None = None


class Photo(BaseModel):
    id: str
    prompt: str
    width: int
    height: int
    image_url: str
    mode: MaskingMode | None = None
    approved: bool = False
    interpreted: Interpreted | None = None


class PhotoDetail(Photo):
    regions: list[Region]


class DetectRequest(BaseModel):
    kinds: list[RegionKind] | None = None
    all_kinds: bool = False


class RefineRequest(BaseModel):
    region_id: str | None = None
    kind: RegionKind = "custom"
    box: list[int] | None = Field(default=None, min_length=4, max_length=4)
    point: list[int] | None = Field(default=None, min_length=2, max_length=2)

    @model_validator(mode="after")
    def one_prompt(self):
        if (self.box is None) == (self.point is None):
            raise ValueError("Provide exactly one of box or point")
        return self


class TargetTap(BaseModel):
    """A tap on a roof plane, in photo pixels: moves that plane into or out of the target house."""

    point: list[int] = Field(min_length=2, max_length=2)


class CreateRegionRequest(BaseModel):
    kind: RegionKind = "custom"
    label: str | None = None
    geometry: GeometryType = "area"
    side: Side | None = None
    color: str | None = Field(default=None, pattern=r"^#[0-9a-fA-F]{6}$")
    polyline: list[list[int]] | None = None


class RegionPatch(BaseModel):
    label: str | None = None
    kind: RegionKind | None = None
    side: Side | None = None
    edge: Edge | None = None
    color: str | None = Field(default=None, pattern=r"^#[0-9a-fA-F]{6}$")
    visible: bool | None = None
    polyline: list[list[int]] | None = None


class ModeRequest(BaseModel):
    mode: MaskingMode


class InterpretRequest(BaseModel):
    # None keeps the photo's instruction and re-reads it, e.g. after the catalogue changed.
    prompt: str | None = Field(default=None, max_length=2000)


class RegisterRequest(BaseModel):
    email: EmailStr
    name: str = Field(min_length=1, max_length=100, pattern=r"\S")
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    email: str
    name: str
    created_at: datetime


ProductCategory = Literal["c9", "c7", "mini", "icicle", "wreath", "garland"]
Inches = Annotated[float, Field(gt=0, le=120)]


class ProductBase(BaseModel):
    category: ProductCategory
    name: str = Field(min_length=1, max_length=120, pattern=r"\S")
    bulb_diameter_in: float | None = Field(default=None, gt=0, le=12)
    spacing_in: float | None = Field(default=None, gt=0, le=120)
    spacing_options_in: list[Inches] = Field(default=[], max_length=12)
    strand_length_ft: float | None = Field(default=None, gt=0, le=1000)
    diameter_in: float | None = Field(default=None, gt=0, le=240)
    bulb_count: int | None = Field(default=None, ge=0, le=100_000)
    drops_in: list[Inches] = Field(default=[], max_length=24)
    colours: list[str] = Field(default=[], max_length=12)
    twinkle: bool = False


class ProductCreate(ProductBase):
    @model_validator(mode="after")
    def describes_its_category(self):
        found = products.problems(self.model_dump())
        if found:
            raise ValueError("; ".join(found))
        return self


class ProductPatch(BaseModel):
    category: ProductCategory | None = None
    name: str | None = Field(default=None, min_length=1, max_length=120, pattern=r"\S")
    bulb_diameter_in: float | None = Field(default=None, gt=0, le=12)
    spacing_in: float | None = Field(default=None, gt=0, le=120)
    spacing_options_in: list[Inches] | None = Field(default=None, max_length=12)
    strand_length_ft: float | None = Field(default=None, gt=0, le=1000)
    diameter_in: float | None = Field(default=None, gt=0, le=240)
    bulb_count: int | None = Field(default=None, ge=0, le=100_000)
    drops_in: list[Inches] | None = Field(default=None, max_length=24)
    colours: list[str] | None = Field(default=None, max_length=12)
    twinkle: bool | None = None


class Product(ProductBase):
    id: str
    # Built-in catalogue entries are shared and read-only; everything else belongs to its user.
    builtin: bool
    # Retired: out of the library and out of the parser, still resolvable for saved designs.
    archived: bool = False
    # A picture of the product, or None when none was uploaded.
    image_url: str | None = None


class PaletteColour(BaseModel):
    key: str
    name: str
    hex: str


class PatternStep(BaseModel):
    colour: str
    count: int = Field(default=1, ge=1, le=50)

    @field_validator("colour")
    @classmethod
    def in_palette(cls, value: str) -> str:
        if value not in products.COLOURS:
            raise ValueError(f"unknown colour {value!r}")
        return value


class Pattern(BaseModel):
    """An ordered colour sequence repeated along a path: red, white; or red x2, green, white."""

    steps: list[PatternStep] = Field(min_length=1, max_length=8)
    # "reverse" starts the sequence from the far end of the path.
    direction: Literal["forward", "reverse"] = "forward"
    # Where in the sequence the first position falls.
    offset: int = Field(default=0, ge=0, le=1000)
    # What advances the sequence: each bulb, or each full turn of a wrap (candy-cane bands).
    unit: Literal["bulb", "turn"] = "bulb"


class Wrap(BaseModel):
    """How a strand spirals up a trunk or collar. start and stop are shares of the zone's height,
    measured from its base."""

    start: float = Field(default=0.0, ge=0, le=1)
    stop: float = Field(default=1.0, ge=0, le=1)
    turn_spacing_in: float = Field(default=6.0, gt=0, le=48)

    @model_validator(mode="after")
    def starts_below_stop(self):
        if self.stop <= self.start:
            raise ValueError("a wrap must stop above where it starts")
        return self


class LightingSettings(BaseModel):
    """What lights one region of a design. Stored in design_regions.settings; these are render
    parameters, never baked into pixels."""

    product_id: UUID
    # Required for socket lines (C9, C7); strands carry their own colours.
    pattern: Pattern | None = None
    # A spacing the product is sold at; None means the product's default.
    spacing_in: float | None = Field(default=None, gt=0, le=120)
    # None means the product's own setting.
    twinkle: bool | None = None
    # Palm trunks and collars only; None there means wrapped over the full height.
    wrap: Wrap | None = None


class DesignRequest(BaseModel):
    """One confirmed request: the places it lights and the lighting they get."""

    text: str = Field(default="", max_length=400)
    kinds: list[RegionKind] = Field(min_length=1)
    sides: list[Side] = []
    edges: list[Edge] = []
    settings: LightingSettings


class DesignPut(BaseModel):
    requests: list[DesignRequest] = Field(default=[], max_length=24)


class DesignRegionOut(BaseModel):
    region_id: str
    settings: LightingSettings


class DesignOut(BaseModel):
    id: str
    name: str
    # The confirmed requests, each with the regions it lit at the time of confirmation.
    requests: list[dict]
    regions: list[DesignRegionOut]
