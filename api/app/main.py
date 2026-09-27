from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.config import get_settings
from app.db import engine
from app.routers import auth, files, photos, products, regions

settings = get_settings()
(settings.storage_dir / "photos").mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Light Design API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth.router)
app.include_router(photos.router)
app.include_router(regions.router)
app.include_router(products.router)
app.include_router(files.router)


@app.get("/health")
def health(response: Response):
    try:
        with engine().connect() as connection:
            connection.execute(text("select 1"))
        database = "ok"
    except Exception:  # noqa: BLE001 — report the outage, don't raise it
        database = "unreachable"
        response.status_code = 503
    return {
        "status": "ok" if database == "ok" else "degraded",
        "milestone": "M2",
        "database": database,
        "segmentation_provider": settings.segmentation_provider,
        "roofline_detector": settings.roofline_detector,
        # Which roof-line path is live. Settings are read once at startup, so this is the only
        # way to tell whether an .env edit has taken effect without reading the server log.
        "edge_extractor": settings.edge_extractor,
        "partition_engine": settings.partition_engine if settings.edge_extractor == "partition" else None,
        # TARGET_HOUSE_ONLY, for the same reason: the same photo is run with it off and on.
        "target_house_only": settings.target_house_only,
    }
