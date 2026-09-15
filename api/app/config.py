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

    # hybrid | profile — how roof edges are pulled from a plane mask. "profile" restores
    # the top/bottom-profile-only behaviour exactly, without a code rollback.
    edge_extractor: str = "hybrid"

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
