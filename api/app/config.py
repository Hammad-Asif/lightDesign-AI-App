from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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


@lru_cache
def get_settings() -> Settings:
    return Settings()
