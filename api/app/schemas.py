from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, EmailStr, Field, model_validator

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
    "palm_frond",
    "tree_canopy",
    "obstruction",
    "custom",
]
GeometryType = Literal["line", "area"]
Side = Literal["front", "left", "right"]
Edge = Literal["eave", "ridge"]
MaskingMode = Literal["manual", "automatic"]


class Region(BaseModel):
    id: str
    kind: RegionKind
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
