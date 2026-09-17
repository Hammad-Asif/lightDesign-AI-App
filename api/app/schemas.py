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
Edge = Literal["eave", "rake", "hip", "ridge", "junction", "unknown"]
MaskingMode = Literal["manual", "automatic"]


class Region(BaseModel):
    id: str
    kind: RegionKind
    role: RegionRole = "target"
    label: str
    geometry: GeometryType
    source: Literal["auto", "manual"]
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
