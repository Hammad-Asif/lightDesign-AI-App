from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

API_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = API_DIR.parent


class Settings(BaseSettings):
    # Both paths are absolute. They used to resolve against the working directory, which is
    # how two separate photo stores ended up on disk depending on where uvicorn was started.
    model_config = SettingsConfigDict(env_file=API_DIR / ".env", extra="ignore")

    storage_dir: Path = REPO_DIR / "storage"
    database_url: str = "postgresql+psycopg://localhost/lightdesign"

    # Sign-in lives in an HttpOnly cookie so the browser also sends it when the editor canvas
    # loads photo and mask images, which cannot carry an Authorization header.
    session_cookie: str = "ld_session"
    session_days: int = 14
    # True in production, behind HTTPS.
    cookie_secure: bool = False
    # JSON list in the environment, e.g. CORS_ORIGINS='["https://designs.example.com"]'.
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # fal | mock — mock is offline and is what the smoke tests assert against.
    segmentation_provider: str = "mock"

    # Where roof lines come from, in order: "v3" (the hosted model on the target-house crop) and/or
    # "partition" (the SAM plane geometry, EDGE_EXTRACTOR/PARTITION_ENGINE). The first source that
    # answers draws ALL the roof lines; the rest are fallbacks, never mixed in. SAM always runs for
    # roof planes, blockers, the target house and every other kind. "v3,partition": V3's lines,
    # partition's only if V3 fails. "partition" alone is the existing pipeline, plus confidence.
    roofline_sources: Annotated[list[str], NoDecode] = ["partition"]
    # Deprecated: the old either/or switch. v3_sol with no ROOFLINE_SOURCES means "v3,partition".
    roofline_detector: str | None = None

    # How a typed instruction is read into requests. "rules" is the hand-written patterns in
    # services/intent.py; "llm" asks a model and falls back to the rules whenever it cannot
    # answer. Both produce the same Specification, so detection and the confirmation screen
    # are unaffected either way. Reuses openrouter_api_key.
    instruction_parser: Literal["rules", "llm"] = "llm"
    parser_model: str = "openai/gpt-6-sol"
    parser_timeout: float = 20
    openrouter_api_key: str = Field(default="", repr=False)
    roofline_model: str = "openai/gpt-6-sol"
    roofline_reasoning: Literal["none", "low", "medium", "high", "xhigh", "max"] = "low"
    roofline_timeout: float = Field(default=360, gt=0, le=600)
    # Padding around the target house's roof bounding box for the V3 crop, as a share of the
    # box's own width/height on each side. Enough to keep corners, porch and garage roofs.
    v3_crop_padding: float = Field(default=0.10, ge=0.0, le=0.5)
    # The crop's top goes up to where the sky begins above the roof (one more SAM prompt, "sky"),
    # so an upper storey, chimney or gable top SAM's roof masks missed is still shown. Never
    # tighter than the padding; never more than V3_CROP_SKY_MAX_RISE of the roof box's width above
    # the roof. No sky found: the padding decides. Off: padding only, and no sky prompt.
    v3_crop_sky: bool = True
    v3_crop_sky_max_rise: float = Field(default=0.5, ge=0.0, le=3.0)
    # Second V3 calls for planes the first call missed. After the first answer, a target-house
    # roof plane whose visible outline has under V3_MISSED_COVERAGE of its length near any V3 line
    # is "missed"; V3 is asked again on a crop around it (V3_MISSED_PADDING each side, sky-bounded
    # top), largest plane first, at most V3_MISSED_MAX_CALLS extra calls, each cached and billed.
    # Planes under V3_MISSED_MIN_AREA of the house's roof area are ignored. Lines repeating the
    # first call's are dropped. Off by default: every extra call costs money.
    v3_missed_planes: bool = False
    v3_missed_max_calls: int = Field(default=3, ge=1, le=10)
    v3_missed_coverage: float = Field(default=0.15, ge=0.0, le=1.0)
    v3_missed_min_area: float = Field(default=0.03, ge=0.0, le=1.0)
    v3_missed_padding: float = Field(default=0.25, ge=0.0, le=1.0)
    # Cache V3 answers on disk under STORAGE_DIR/v3_cache, keyed by image, crop, model and prompt,
    # so the same detection is never billed twice.
    v3_cache: bool = True
    # Conservative endpoint snapping of V3 lines onto a nearby parallel image edge. Off: V3
    # geometry is used exactly as the model returned it.
    v3_snap: bool = True
    # How far (share of the roof extent, never under 4 px) a V3 endpoint may be moved, and the
    # quality in [0, 1] a snap must reach before anything moves.
    v3_snap_radius: float = Field(default=0.008, ge=0.0, le=0.05)
    v3_snap_min_quality: float = Field(default=0.55, ge=0.0, le=1.0)
    # Find in Box: a found line is the same edge as one already on the photo, and so not added,
    # within these (distance as a share of the roof extent, never under 5 px).
    roofline_match_distance: float = Field(default=0.012, gt=0.0, le=0.1)
    roofline_match_angle_deg: float = Field(default=6.0, gt=0.0, le=30.0)
    roofline_match_overlap: float = Field(default=0.5, gt=0.0, le=1.0)
    # A line is associated with a vanishing point when it points at it within this angle.
    roofline_vp_deg: float = Field(default=4.0, gt=0.0, le=20.0)
    # Evidence weights for a line's confidence in [0, 1]; renormalised over the components that
    # apply to a line. Heuristic weights, to be validated on the annotated photos, not physics.
    roofline_confidence_weights: dict[str, float] = {
        "source": 0.30, "edge": 0.25, "vp": 0.15, "geometry": 0.10, "plane": 0.15, "occlusion": 0.05,
    }
    # confidence >= high is "high", < low is "low" (drawn dashed in the editor), else "medium".
    roofline_confidence_high: float = Field(default=0.70, ge=0.0, le=1.0)
    roofline_confidence_low: float = Field(default=0.45, ge=0.0, le=1.0)
    # Development: keep every stage of the unified pipeline (raw V3 -> final) with per-line
    # lifecycle records in photo.interpreted.roofline_debug, and write stage overlays to the
    # photo's debug/ folder. Off in production: it only adds data, never changes a line.
    roofline_debug: bool = False

    # SAM 3 on fal.ai: text, box and point prompts from one endpoint.
    fal_key: str = ""
    fal_model: str = "fal-ai/sam-3/image"
    # png | rle — "rle" avoids a mask download per instance but fal documents the
    # encoding only as "Kaggle/COCO order"; verify against the live API before using it.
    fal_mask_format: str = "png"
    fal_max_masks: int = 16
    fal_timeout: float = 120.0
    fal_concurrency: int = 4

    # hybrid | profile | partition — how roof edges are pulled from the plane masks. "profile"
    # restores the top/bottom-profile-only behaviour exactly, without a code rollback.
    # "partition" (services/partition.py) gives every pixel to one plane first and reads the
    # lines off the borders, so hips and ridges between planes come out as lines; opt-in.
    edge_extractor: str = "hybrid"
    # Partition mode's detection request, matching the reference method: working-copy long
    # edge and instances per call (it also posts a JPEG). Both change the cache key of every
    # roof response, so switching a photo between modes re-bills its detection.
    partition_detect_max_edge: int = 2000
    partition_fal_max_masks: int = 12
    # After the energy relaxation each lit line (eave, rake, hip) may move along its own normal
    # to the strongest parallel image edge within the band — soffit to fascia — with direction
    # kept and every junction re-solved. Off: the geometry is left exactly as the graph made it.
    # reference | port. "reference" runs the reference implementation itself (the `roofline`
    # package installed from the roof-detection project) on the reference's own working copy,
    # so the product draws exactly what that project draws. "port" is services/partition.py.
    partition_engine: str = "reference"
    # Reference-engine option: find foliage with a SAM "tree" pass (one more call) instead of
    # by colour, as the reference app's "Find trees with SAM" box does.
    partition_tree_pass: bool = False
    partition_constrained_snap: bool = True
    # The product's free snap (services/snap.py, with the straight-fit fallback) on top of
    # partition's lines as well. Off by default: it refits per line and can turn a corner.
    partition_snap: bool = False
    # Product rule kept from the hybrid path: a lower roof's top edge running into the wall of
    # the storey above is relabelled "junction" (dark). Off: roles are the reference table only.
    partition_junctions: bool = True

    # Move each extracted roof line onto the image edge it follows (services/snap.py). Measured
    # on the annotated houses: mean F1 49.6 -> 54.8, and a drawn line sits 3.1px from the true
    # gutter instead of 4.8px. Set false to keep lines exactly where the mask put them.
    snap_to_edges: bool = True

    # On: a roof request also detects palm trunks, trees and cars, kept as blockers rather than
    # lighting targets. A roof edge one of them cuts in two is joined back into one straight line
    # that passes behind it, an edge it cuts off is carried through to its far side, and the
    # hidden stretch is reported and never lit. Without them the app cannot tell "roof behind a
    # palm" from "no roof here" — in line geometry they are the same. Costs four detection
    # requests per roof photo instead of one. Off: an instruction returns only what it asked for.
    detect_blockers: bool = True

    # On: roof lines only for the house being quoted. Detection still returns every roof plane;
    # services/target.py picks the planes that belong to the target house, only those get
    # lines, and only their segments decide which vanishing points stand and where. Every plane
    # is stored, so a tap in the editor can move one into or out of the house without a new
    # detection call. Off: every plane gets lines and the whole photo decides the vanishing
    # points, as before, and taps are refused.
    target_house_only: bool = False

    # Off: draw hips that fall inside a single mask, where two roof planes came back merged and
    # no mask boundary runs along the line between them. The candidates are real geometry (apex
    # to eave corner) but the photo does not confirm them: measured on the eval set, every
    # candidate sat on image evidence weaker than the lines we already draw, so none survive the
    # check and turning this on would mean drawing them on shape alone. See services/roofline.py.
    interior_hips: bool = False

    google_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash-image"

    max_upload_mb: int = 25
    max_image_edge: int = 6000
    detect_max_edge: int = 1280

    @field_validator("storage_dir")
    @classmethod
    def _anchor_storage(cls, value: Path) -> Path:
        # A relative STORAGE_DIR (api/.env sets "storage") would otherwise resolve against
        # whichever directory the server was started from, splitting the photo store again.
        return value if value.is_absolute() else REPO_DIR / value

    @field_validator("roofline_sources", mode="before")
    @classmethod
    def _parse_sources(cls, value):
        # ROOFLINE_SOURCES=v3,partition in the environment; a JSON list works too.
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("["):
                import json

                value = json.loads(text)
            else:
                value = [part for part in text.replace(" ", "").split(",") if part]
        sources: list[str] = []
        for source in value or []:
            source = str(source).strip().lower()
            if source not in ("v3", "partition"):
                raise ValueError(f"unknown roofline source {source!r}; use v3 and/or partition")
            if source not in sources:
                sources.append(source)
        return sources or ["partition"]

    @model_validator(mode="after")
    def _legacy_detector(self):
        # A deployment that still says ROOFLINE_DETECTOR=v3_sol asked for V3; it now gets V3 as
        # the primary source with partition behind it, never V3 instead of SAM.
        if "roofline_sources" not in self.model_fields_set and self.roofline_detector == "v3_sol":
            self.roofline_sources = ["v3", "partition"]
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
