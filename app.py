"""
Roofline -- backend.

Roof structure lines (ridges, hips, valleys, rakes, eaves) from one street-level photo.

Principle: masks are plane SUPPORT REGIONS, not the source of lines. Lines come from (a) plane-to-
plane adjacency after a hard partition, which gives the interior ridges, hips and valleys, and
(b) robust line fits over trusted boundary arcs, which give the exterior eaves and rakes. Raw mask
boundary and raw LSD segments are never emitted as geometry.

Every phase is callable and inspectable on its own:
  POST /api/segment    phases 1-2 (+4a): SAM 3 instances -> hard partition -> cleanup -> trusted arcs,
                       plus vanishing points from all lines in the photo. Runs once per prompt.
  POST /api/geometry   phases 3-7: line fitting -> direction priors -> planar graph + energy loop ->
                       hidden creases -> roles and pitch. `stop_after` returns any intermediate stage.
                       A few milliseconds, so it reruns on every slider tick.
  GET  /api/layer/...  inspection images: raw instances, partition, trusted arcs, LSD lines by VP.
  /api/design/*        the one line editor (web /design): detection for mock-ups, SAM prompts, Studio lines -> editor.
  /api/auth/*, /api/projects/*   accounts and saved projects (accounts.py). Every other endpoint
                       needs a signed-in user; the Next.js frontend in web/ is the UI.

Run:  uvicorn app:app --reload --port 8765, then the frontend: cd web && npm run dev.   SAM 3 (fal.ai) switches on when FAL_KEY is set in .env.
Eval: python -m eval.harness            metrics against annotated photos; python -m eval.tune for thresholds.
"""

from __future__ import annotations

import io
import math
import os
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel, Field

import accounts
import design
import instruction
from roofline import cleanup, local_seg, partition, pipeline, roles, sam, synth, vanishing
from roofline.config import Params
from roofline.watermark import is_watermarked

BASE_DIR = Path(__file__).resolve().parent
WORKING_MAX_DIM = 2000
MAX_SESSIONS = 32
PARAMS = Params.load()            # params.json (written by eval/tune.py) overrides the defaults

app = FastAPI(title="Roofline")


# --------------------------------------------------------------------------------------------
# Sessions
# --------------------------------------------------------------------------------------------

class Session:
    def __init__(self, image: np.ndarray, source: np.ndarray, filename: str, exif_focal_35mm: Optional[float] = None, owner: str = ""):
        self.image, self.source, self.filename, self.owner = image, source, filename, owner
        self.exif_focal_35mm = exif_focal_35mm
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 94])
        self.encoded = buf.tobytes()
        self.instances: list[dict] = []
        self.prep: Optional[pipeline.Prepared] = None
        self.local_cache: dict[str, np.ndarray] = {}
        self.sample_instances: Optional[list[dict]] = None
        self.sample_name: Optional[str] = None
        self.version = 0
        self.is_sample = False


SESSIONS: "OrderedDict[str, Session]" = OrderedDict()


def get_session(session_id: str, user: dict) -> Session:
    session = SESSIONS.get(session_id)
    if session is None or session.owner != user["id"]:
        raise HTTPException(404, "Session not found. Upload the image again.")
    return session


def put_session(session: Session) -> str:
    session_id = uuid.uuid4().hex[:12]
    SESSIONS[session_id] = session
    while len(SESSIONS) > MAX_SESSIONS:
        SESSIONS.popitem(last=False)
    return session_id


def exif_focal_35mm(data: bytes) -> Optional[float]:
    """35 mm equivalent focal length from EXIF, if the photo still carries it."""
    try:
        from PIL import Image
        exif = Image.open(io.BytesIO(data)).getexif()
        value = exif.get_ifd(0x8769).get(0xA405)             # Exif IFD -> FocalLengthIn35mmFilm
        return float(value) if value else None
    except Exception:
        return None


def clean(obj):
    """numpy scalars and arrays -> plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def session_info(session_id: str, s: Session) -> dict:
    h, w = s.image.shape[:2]
    flagged, reason = is_watermarked(s.image, s.filename)
    return {"session_id": session_id, "filename": s.filename, "width": w, "height": h,
            "source_width": s.source.shape[1], "source_height": s.source.shape[0], "scale_from_source": round(w / s.source.shape[1], 6),
            "image_url": f"/api/image/{session_id}", "exif_focal_35mm": s.exif_focal_35mm,
            "watermark": {"flagged": flagged, "reason": reason}, "is_sample": s.is_sample, "sample_name": s.sample_name}


# --------------------------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------------------------

class SegmentRequest(BaseModel):
    session_id: str
    mode: str = Field("sam3", pattern="^(sam3|sample|mock|threshold)$")
    text_prompt: str = Field(sam.SAM_TEXT_PROMPT, max_length=200)
    points: list[list[int]] = Field(default_factory=list, max_length=24)
    threshold: int = Field(80, ge=0, le=255)
    invert: bool = True
    foliage: str = Field("color", pattern="^(color|sam)$")          # "sam" spends one more call on a "tree" pass
    subtract: list[str] = Field(default_factory=list, max_length=4)  # concepts to cut out of planes, e.g. ["chimney", "skylight"]
    refine_small: bool = False                                        # re-prompt small planes on a high-resolution crop


class GeometryRequest(BaseModel):
    session_id: str
    epsilon: float = Field(PARAMS.epsilon, ge=0.001, le=0.05)
    merge_angle_deg: float = Field(PARAMS.merge_angle_deg, ge=1.0, le=15.0)
    snap_radius_frac: float = Field(PARAMS.snap_radius_frac, ge=0.001, le=0.03)
    w_dir: float = Field(PARAMS.w_dir, ge=0.0, le=50.0)
    stop_after: int = Field(7, ge=3, le=7)
    focal_35mm: Optional[float] = Field(None, ge=8.0, le=400.0)


class SignupRequest(BaseModel):
    email: str = Field(max_length=200)
    name: str = Field("", max_length=100)
    password: str = Field(min_length=1, max_length=200)


class LoginRequest(BaseModel):
    email: str = Field(max_length=200)
    password: str = Field(max_length=200)


class ProjectCreate(BaseModel):
    session_id: str
    name: str = Field(min_length=1, max_length=120)
    data: dict


class ProjectUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=120)
    data: Optional[dict] = None


# --------------------------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------------------------

@app.get("/")
def index() -> RedirectResponse:
    """The UI is the Next.js app in web/."""
    return RedirectResponse(os.environ.get("FRONTEND_URL", "http://localhost:3000"))


@app.post("/api/auth/signup")
def signup(req: SignupRequest, response: Response) -> dict:
    if not accounts.signup_allowed():
        raise HTTPException(403, "Sign-up is closed. Ask the owner for an account.")
    user = accounts.create_user(req.email, req.name, req.password)
    accounts.set_cookie(response, user["id"])
    return {"user": user}


@app.post("/api/auth/login")
def login(req: LoginRequest, request: Request, response: Response) -> dict:
    user = accounts.authenticate(req.email, req.password, request.client.host if request.client else "")
    accounts.set_cookie(response, user["id"])
    return {"user": user}


@app.post("/api/auth/logout")
def logout(response: Response) -> dict:
    accounts.clear_cookie(response)
    return {"ok": True}


@app.get("/api/auth/me")
def me(request: Request) -> dict:
    return {"user": accounts.optional_user(request), "signup_allowed": accounts.signup_allowed()}


@app.get("/api/config")
def config(user: dict = Depends(accounts.current_user)) -> dict:
    return {**sam.describe(), "role_colors": roles.ROLE_COLORS,
            "defaults": {"epsilon": PARAMS.epsilon, "merge_angle_deg": PARAMS.merge_angle_deg, "snap_radius_frac": PARAMS.snap_radius_frac, "w_dir": PARAMS.w_dir},
            "tuned": (BASE_DIR / "params.json").exists()}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), user: dict = Depends(accounts.current_user)) -> dict:
    data = await file.read()
    source = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
    if source is None:
        raise HTTPException(400, "That file is not an image OpenCV can read (use JPG, PNG or WebP).")
    h, w = source.shape[:2]
    scale = WORKING_MAX_DIM / max(h, w)
    image = cv2.resize(source, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else source
    session = Session(image, source, file.filename or "upload", exif_focal_35mm(data), owner=user["id"])
    return session_info(put_session(session), session)


def make_sample_session(name: str, owner: str) -> Session:
    try:
        scene, instances, _ = synth.load_sample(name)
    except KeyError:
        raise HTTPException(404, "Unknown sample house.")
    session = Session(scene.image, scene.image, f"{name.replace('synth_', 'sample-')}.png", round(scene.camera.f / max(scene.image.shape[:2]) * 36.0, 1), owner=owner)
    session.sample_instances, session.is_sample, session.sample_name = instances, True, name
    return session


@app.get("/api/samples")
def samples(user: dict = Depends(accounts.current_user)) -> dict:
    return {"samples": [{"name": k, "label": v} for k, v in synth.SAMPLES.items()]}


@app.post("/api/sample")
def sample(name: str = "synth_cross_high", user: dict = Depends(accounts.current_user)) -> dict:
    """A synthetic house with SAM-like instances (overlapping parent, wobble, holes): the whole pipeline with no API call."""
    session = make_sample_session(name, user["id"])
    return session_info(put_session(session), session)


@app.get("/api/image/{session_id}")
def image(session_id: str, user: dict = Depends(accounts.current_user)) -> Response:
    return Response(get_session(session_id, user).encoded, media_type="image/jpeg")


@app.post("/api/segment")
def segment(req: SegmentRequest, user: dict = Depends(accounts.current_user)) -> dict:
    s = get_session(req.session_id, user)
    started = time.perf_counter()
    points = [(int(p[0]), int(p[1])) for p in req.points if len(p) == 2]
    text = req.text_prompt.strip()
    foliage_override = None
    try:
        if req.mode == "sample":
            if s.sample_instances is None:
                raise HTTPException(409, "This image is not the built-in sample. Use SAM 3 or a local method.")
            instances = [dict(i) for i in s.sample_instances]
        elif req.mode == "sam3":
            if not sam.configured():
                raise HTTPException(409, "SAM 3 is not configured. Put FAL_KEY in .env and restart the server.")
            if not text and not points:
                raise HTTPException(400, "Give SAM 3 a text prompt or click at least one roof plane.")
            instances = sam.text_instances(s.image, text) if text else []
            if req.refine_small and text:
                instances = sam.refine_small_planes(s.source, s.image, instances, text)
            for point in points:
                instances += sam.point_instance(s.image, point)
            if req.subtract:
                cut = sam.concept_union(s.image, [c.strip() for c in req.subtract if c.strip()]) > 0
                instances = [{**i, "mask": np.where(cut, 0, i["mask"]).astype(np.uint8)} for i in instances]
            if req.foliage == "sam":
                foliage_override = sam.concept_union(s.image, ["tree"])
        elif req.mode == "mock":
            instances = []
            for key, point in ([(f"{p[0]},{p[1]}", p) for p in points] or [("centre", None)]):
                if key not in s.local_cache:
                    s.local_cache[key] = local_seg.segment_mock(s.image, point)
                instances.append({"mask": s.local_cache[key], "score": None, "source": "grabcut"})
        else:
            instances = [{"mask": local_seg.segment_threshold(s.image, req.threshold, req.invert), "score": None, "source": "threshold"}]
    except sam.SamError as err:
        raise HTTPException(err.status, err.detail)

    instances = [i for i in instances if i["mask"].any()]
    s.instances = instances
    s.prep = pipeline.prepare(s.image, instances, PARAMS, foliage_override=foliage_override)
    s.version += 1
    part = s.prep.partition
    colors = partition.label_colors(max(len(part.planes), 1))
    table = []
    for i, inst in enumerate(instances):
        plane = next((p for p in part.planes if p["instance"] == i), None)
        gone = next((d for d in part.dropped if d["instance"] == i), None)
        still = plane is not None and bool((s.prep.labels == plane["label"]).any())
        table.append({"id": i, "source": inst.get("source"), "score": inst.get("score"),
                      "area_percent": round(float((inst["mask"] > 0).mean()) * 100, 2),
                      "plane": plane["label"] if still else None, "kept_fraction": plane["kept_fraction"] if plane else (gone or {}).get("kept_fraction"),
                      "status": "plane" if still else ("duplicate" if gone else "too small after cleanup"),
                      "color": "#%02x%02x%02x" % colors[plane["label"] - 1][::-1] if still else None})
    vps = s.prep.extras.get("vps")
    boundary_px = sum(len(b.contour) for b in s.prep.boundaries)
    return clean({
        "version": s.version, "instances": table, "planes": int(len({b.plane for b in s.prep.boundaries})),
        "coverage_percent": round(float((s.prep.labels > 0).mean()) * 100, 2),
        "untrusted_percent": round(100.0 * sum(int((~b.trusted).sum()) for b in s.prep.boundaries) / max(boundary_px, 1), 1),
        "adjacency_percent": round(100.0 * sum(int((b.tags > 0).sum()) for b in s.prep.boundaries) / max(boundary_px, 1), 1),
        "vanishing_points": vps.describe() if vps is not None else None, "lsd_segments": int(len(s.prep.extras.get("lsd", []))),
        "timings": s.prep.timings, "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        "layers": {name: f"/api/layer/{req.session_id}/{name}.png?v={s.version}" for name in ("instances", "partition", "arcs", "lines")},
    })


@app.post("/api/geometry")
def geometry(req: GeometryRequest, user: dict = Depends(accounts.current_user)) -> dict:
    s = get_session(req.session_id, user)
    if s.prep is None:
        raise HTTPException(409, "Run segmentation before requesting geometry.")
    params = replace(PARAMS, epsilon=req.epsilon, merge_angle_deg=req.merge_angle_deg, snap_radius_frac=req.snap_radius_frac, w_dir=req.w_dir)
    focal_35 = req.focal_35mm or s.exif_focal_35mm
    focal_px = focal_35 / 36.0 * max(s.image.shape[:2]) if focal_35 else None
    source = None if not focal_35 else ("entered by hand" if req.focal_35mm else "EXIF")
    result = pipeline.solve(s.prep, params, req.stop_after, focal_px, source)
    junctions = sorted({tuple(p) for seg in result["segments"] for p in seg["points"]})
    return clean({"segments": result["segments"], "planes": result["planes"], "vertices": [list(j) for j in junctions],
                  "stage": result["stage"], "info": result["info"], "timings": result["timings"],
                  "parameters": {"epsilon": req.epsilon, "merge_angle_deg": req.merge_angle_deg, "snap_radius_frac": req.snap_radius_frac,
                                 "w_dir": req.w_dir, "stop_after": req.stop_after, "focal_35mm": focal_35}})


@app.get("/api/layer/{session_id}/{name}.png")
def layer(session_id: str, name: str, user: dict = Depends(accounts.current_user)) -> Response:
    s = get_session(session_id, user)
    if s.prep is None:
        raise HTTPException(409, "Run segmentation first.")
    dim = (s.image * 0.45).astype(np.uint8)
    if name == "instances":                      # raw SAM output: overlaps stay visible
        canvas = dim.astype(np.float32)
        for i, inst in enumerate(s.instances):
            color = np.array(partition.label_colors(len(s.instances))[i], np.float32)
            m = inst["mask"] > 0
            canvas[m] = canvas[m] * 0.45 + color * 0.55
        out = canvas.astype(np.uint8)
    elif name == "partition":
        colored = partition.render_labels(s.prep.labels)
        out = np.where((s.prep.labels > 0)[..., None], (dim * 0.35 + colored * 0.65).astype(np.uint8), dim)
    elif name == "arcs":
        out = cleanup.render_arcs(s.image, s.prep.boundaries, s.prep.foliage)
    elif name == "lines":
        out = dim.copy()
        vps, segs = s.prep.extras.get("vps"), s.prep.extras.get("lsd", np.zeros((0, 4)))
        palette = {"v": (255, 80, 255), "h1": (255, 220, 60), "h2": (60, 220, 255)}
        for x1, y1, x2, y2 in segs:
            color, u, mid = (150, 150, 150), np.array([x2 - x1, y2 - y1]), np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            u = u / (np.linalg.norm(u) or 1)
            for vp_name, vp in (vps.named() if vps is not None else []):
                d = vps.direction_at(vp, mid)
                if d is not None and abs(float(u @ d)) >= math.cos(math.radians(PARAMS.vp_inlier_deg)):
                    color = palette[vp_name]
                    break
            cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2, cv2.LINE_AA)
    else:
        raise HTTPException(404, "Unknown layer.")
    ok, buf = cv2.imencode(".png", out)
    return Response(buf.tobytes(), media_type="image/png", headers={"Cache-Control": "no-store"})


# --------------------------------------------------------------------------------------------
# Line editor and saved projects
# --------------------------------------------------------------------------------------------

@app.get("/api/projects")
def projects(user: dict = Depends(accounts.current_user)) -> dict:
    return {"projects": accounts.list_projects(user["id"])}


@app.post("/api/projects")
def create_project(req: ProjectCreate, user: dict = Depends(accounts.current_user)) -> dict:
    s = get_session(req.session_id, user)
    h, w = s.image.shape[:2]
    data = {**req.data, "exif_focal_35mm": s.exif_focal_35mm, "is_sample": s.is_sample, "sample_name": s.sample_name}
    return accounts.create_project(user["id"], req.name.strip(), s.filename, w, h, s.encoded, data)


@app.get("/api/projects/{project_id}")
def project(project_id: str, user: dict = Depends(accounts.current_user)) -> dict:
    return accounts.get_project(user["id"], project_id)


@app.put("/api/projects/{project_id}")
def update_project(project_id: str, req: ProjectUpdate, user: dict = Depends(accounts.current_user)) -> dict:
    return accounts.update_project(user["id"], project_id, req.name.strip() if req.name else None, req.data)


@app.delete("/api/projects/{project_id}")
def delete_project(project_id: str, user: dict = Depends(accounts.current_user)) -> dict:
    accounts.delete_project(user["id"], project_id)
    return {"ok": True}


@app.get("/api/projects/{project_id}/image")
def project_image(project_id: str, user: dict = Depends(accounts.current_user)) -> Response:
    accounts.get_project(user["id"], project_id)
    return Response(accounts.image_path(project_id).read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


@app.post("/api/projects/{project_id}/session")
def open_project(project_id: str, user: dict = Depends(accounts.current_user)) -> dict:
    """A working session from the saved photo, so its lines can be edited, re-measured or detected again."""
    p = accounts.get_project(user["id"], project_id)
    data = accounts.image_path(project_id).read_bytes()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(410, "The saved photo is missing.")
    session = Session(image, image, p["filename"], p["data"].get("exif_focal_35mm"), owner=user["id"])
    if p["data"].get("is_sample"):
        sample_session = make_sample_session(p["data"].get("sample_name") or "synth_cross_high", user["id"])
        session.sample_instances = sample_session.sample_instances
        session.is_sample, session.sample_name = True, sample_session.sample_name
    return session_info(put_session(session), session)


# --------------------------------------------------------------------------------------------
# Lighting design: instruction -> targets -> detection -> editable lines and areas
# --------------------------------------------------------------------------------------------

class ParseRequest(BaseModel):
    text: str = Field(max_length=2000)


class DesignDetectRequest(BaseModel):
    session_id: str
    targets: list[dict] = Field(default_factory=list, max_length=40)
    all_kinds: bool = False


class RefineRequest(BaseModel):
    session_id: str
    point: Optional[list[int]] = None
    box: Optional[list[int]] = None


@app.get("/api/design/catalogue")
def design_catalogue(user: dict = Depends(accounts.current_user)) -> dict:
    return {**instruction.catalogue(), "edge_colours": design.EDGE_COLOURS, "sam_api": sam.configured()}


@app.post("/api/instruction/parse")
def parse_instruction(req: ParseRequest, user: dict = Depends(accounts.current_user)) -> dict:
    return {"targets": instruction.parse(req.text)}


@app.post("/api/design/detect")
def design_detect(req: DesignDetectRequest, user: dict = Depends(accounts.current_user)) -> dict:
    """Automatic mode. Roof lines come from the phased pipeline exactly as /api/segment + /api/geometry
    produce them with the default settings; every detected roof line is lit. Other places
    are SAM 3 concept masks."""
    s = get_session(req.session_id, user)
    started = time.perf_counter()
    kinds = instruction.KINDS
    wanted: set[str] = set()
    for t in req.targets:
        wanted |= {k for k in t.get("kinds", []) if k in kinds}
    if req.all_kinds:
        wanted |= {k for k, v in kinds.items() if v["detect"]}
    notes: list[str] = []
    lines: list[dict] = []
    areas: list[dict] = []
    use_sam = sam.configured() and not s.is_sample                 # the sample's masks are built in: no API calls for it
    if s.is_sample and sam.configured() and any(kinds[k]["geometry"] == "area" and kinds[k]["detect"] for k in wanted):
        notes.append("The sample house only has roof masks; paint any other areas by hand.")

    # the concept calls only need the photo: start them now so they run while the roof is detected
    concept_kinds = [k for k in kinds if k in wanted and kinds[k]["geometry"] == "area" and kinds[k]["detect"]]
    prompts = list(dict.fromkeys(kinds[k]["detect"] for k in concept_kinds))
    pool = ThreadPoolExecutor(max_workers=6) if use_sam and prompts else None
    futures = {prompt: pool.submit(sam.text_instances, s.image, prompt) for prompt in prompts} if pool else {}

    if "roof" in wanted:
        mode = "sample" if s.is_sample else ("sam3" if use_sam else "mock")
        if mode == "mock":
            notes.append("SAM 3 is off (no FAL_KEY), so the roof was found with GrabCut from the centre of the photo. Check it, or draw the lines.")
        segment(SegmentRequest(session_id=req.session_id, mode=mode, text_prompt=sam.SAM_TEXT_PROMPT), user)
        geo = geometry(GeometryRequest(session_id=req.session_id), user)
        for run in design.chain_segments(geo["segments"]):
            lines.append({"kind": "roof", "edge": run["edge"], "polyline": run["polyline"], "visible": True, "source": "auto",
                          "planes": run["planes"], "inferred": run["inferred"], "length_px": run["length_px"], "pitch_apparent": run["pitch_apparent"]})
        if not geo["segments"]:
            notes.append("No roof lines were found. Draw them with Add a line or area.")

    manual = [k for k in wanted if kinds[k]["geometry"] == "area" and not kinds[k]["detect"]] + \
             [k for k in wanted if kinds[k]["geometry"] == "line" and k != "roof"]
    for k in manual:
        notes.append(f"{kinds[k]['label']} are drawn by hand: use Add a line or area.")
    if use_sam:
        try:
            for k in concept_kinds:
                found = [i for i in futures[kinds[k]["detect"]].result() if i["mask"].any()]
                if not found:
                    notes.append(f"No {kinds[k]['label'].lower()} were found. Paint them with the brush if they are there.")
                    continue
                union = np.zeros(s.image.shape[:2], np.uint8)
                for i in found:
                    union |= (i["mask"] > 0).astype(np.uint8)
                scores = [i["score"] for i in found if i["score"] is not None]
                areas.append({"kind": k, "mask": design.mask_to_data_url(union), "visible": True, "source": "auto",
                              "score": round(max(scores), 3) if scores else None, "role": "target"})
        except sam.SamError as err:
            raise HTTPException(err.status, err.detail)
        finally:
            if pool:
                pool.shutdown(wait=False, cancel_futures=True)
    elif concept_kinds and not s.is_sample:
        notes.append("SAM 3 is off (no FAL_KEY), so " + ", ".join(kinds[k]["label"].lower() for k in concept_kinds) + " must be painted by hand.")

    support = design.mask_to_data_url((s.prep.labels > 0).astype(np.uint8)) if "roof" in wanted and s.prep is not None else None
    return clean({"lines": lines, "areas": areas, "notes": notes, "support_mask": support, "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)})


def design_from_segments(segments: list[dict], support: Optional[str], note: str) -> dict:
    """A design (the editor's document) whose lines are these detected segments grouped per edge role, all lit."""
    targets = instruction.parse("roofline")
    counts: dict[str, int] = {}
    regions = []
    usable = [{**seg, "length_px": seg.get("length_px") or 0.0} for seg in segments
              if isinstance(seg.get("points"), list) and len(seg["points"]) == 2 and seg.get("role") in design.EDGE_COLOURS]
    for run in design.chain_segments(usable):
        edge = run["edge"]
        counts[edge] = counts.get(edge, 0) + 1
        regions.append({"id": uuid.uuid4().hex[:8], "kind": "roof", "label": edge.capitalize() + (f" {counts[edge]}" if counts[edge] > 1 else ""),
                        "geometry": "line", "role": "target", "source": "auto", "color": design.EDGE_COLOURS[edge], "visible": True,
                        "edge": edge, "polyline": run["polyline"], "length_px": run["length_px"], "pitch_apparent": run["pitch_apparent"],
                        "planes": run["planes"], "inferred": run["inferred"]})
    return {"version": 1, "instruction": "roofline", "mode": "automatic", "targets": targets, "regions": regions,
            "detected": bool(regions), "lighting_confirmed": False, "approved": False, "support_mask": support,
            "notes": [note] if regions else []}


@app.post("/api/projects/{project_id}/design")
def project_to_design(project_id: str, user: dict = Depends(accounts.current_user)) -> dict:
    """An older project saved from the detection studio, opened in the editor: its lines (hand-edited ones first,
    else the detected ones) become the editor's lines. Its old data is kept alongside."""
    p = accounts.get_project(user["id"], project_id)
    if p["data"].get("design"):
        return p
    source = p["data"].get("edited") or p["data"].get("detected") or {}
    doc = design_from_segments(source.get("segments") or [], None,
                               "These lines come from an earlier detection of this photo. Re-detect runs a fresh automatic detection; your own lines are kept.")
    return accounts.update_project(user["id"], project_id, None, {"design": doc})


class FromStudioRequest(BaseModel):
    session_id: str
    name: str = Field("Roof", min_length=1, max_length=120)
    segments: list[dict] = Field(max_length=2000)


@app.post("/api/design/from-studio")
def design_from_studio(req: FromStudioRequest, user: dict = Depends(accounts.current_user)) -> dict:
    """The Studio's detected lines, opened in the one line editor (/design): a new project whose lines are the
    Studio's segments grouped per edge role, lit as for "roofline" (eaves, rakes, hips), the rest kept unlit."""
    s = get_session(req.session_id, user)
    for seg in req.segments:
        pts = seg.get("points")
        if not (isinstance(pts, list) and len(pts) == 2) or seg.get("role") not in design.EDGE_COLOURS:
            raise HTTPException(400, "Every segment needs two points and a roof edge role.")
    support = design.mask_to_data_url((s.prep.labels > 0).astype(np.uint8)) if s.prep is not None else None
    design_doc = design_from_segments(req.segments, support, "These lines come from the detection studio. Re-detect replaces them with a fresh automatic detection; your own lines are kept.")
    h, w = s.image.shape[:2]
    data = {"design": design_doc, "detected": None, "edited": None, "segmentation": None,
            "geometry_params": {"epsilon": PARAMS.epsilon, "mergeAngle": PARAMS.merge_angle_deg, "snap": PARAMS.snap_radius_frac, "wdir": PARAMS.w_dir, "stage": 7, "focal": None},
            "exif_focal_35mm": s.exif_focal_35mm, "is_sample": s.is_sample, "sample_name": s.sample_name}
    return accounts.create_project(user["id"], req.name.strip(), s.filename, w, h, s.encoded, data)


class RoofBoxRequest(BaseModel):
    session_id: str
    box: list[int] = Field(min_length=4, max_length=4)


@app.post("/api/design/roof-in-box")
def design_roof_in_box(req: RoofBoxRequest, user: dict = Depends(accounts.current_user)) -> dict:
    """Roof lines the detection missed, inside a box the user drew: SAM 3 finds the roof in the box (a box prompt
    with the text "roof"), those planes join the planes already found, the same pipeline runs, and the lines
    inside the box come back. Joining the existing planes gives the new lines their real roles (a valley where
    the new gable meets the main roof, not a free edge)."""
    s = get_session(req.session_id, user)
    if not sam.configured():
        raise HTTPException(409, "SAM 3 is off. Put FAL_KEY in .env and restart the server, or draw the lines with Add a line or area.")
    h, w = s.image.shape[:2]
    x0, y0, x1, y1 = req.box
    box = (max(0, min(x0, x1)), max(0, min(y0, y1)), min(w - 1, max(x0, x1)), min(h - 1, max(y0, y1)))
    if box[2] - box[0] < 8 or box[3] - box[1] < 8:
        raise HTTPException(400, "Draw a larger box around the roof.")
    try:
        found = sam.box_instances(s.image, "roof", box)
    except sam.SamError as err:
        raise HTTPException(err.status, err.detail)
    inside = np.zeros((h, w), bool)
    inside[box[1]:box[3] + 1, box[0]:box[2] + 1] = True
    known = (s.prep.labels > 0) if s.prep is not None else np.zeros((h, w), bool)   # roof already detected
    new = []
    for inst in found:
        m = (inst["mask"] > 0) & inside                                        # the roof in the box, not the whole house
        if (inst["mask"] > 0).sum() == 0 or m.sum() / (inst["mask"] > 0).sum() < 0.6:
            continue
        if (m & ~known).sum() < 0.5 * m.sum():                                  # mostly roof we already have: not what is missing
            continue
        new.append({**inst, "mask": m.astype(np.uint8) * 255, "source": "box"})
    if not new:
        raise HTTPException(404, "No new roof was found in that box (what SAM found there is already detected). Draw it around the missing roof face.")
    base = [i for i in (s.instances or []) if i["mask"].any()] if s.prep is not None else []
    s.instances = base + new
    s.prep = pipeline.prepare(s.image, s.instances, PARAMS)
    s.version += 1
    geo = pipeline.solve(s.prep, PARAMS)
    pad_x, pad_y = 0.06 * (box[2] - box[0]), 0.06 * (box[3] - box[1])
    def in_box(p) -> bool:
        return box[0] - pad_x <= p[0] <= box[2] + pad_x and box[1] - pad_y <= p[1] <= box[3] + pad_y
    picked = [sg for sg in geo["segments"] if in_box(sg["points"][0]) and in_box(sg["points"][1])]
    band = 0.06 * max(box[2] - box[0], box[3] - box[1])
    boxed = np.zeros((h, w), np.uint8)                                        # the roof SAM found in the box
    for inst in new:
        boxed |= (inst["mask"] > 0).astype(np.uint8)
    picked = design.roofline_only(design.drop_inner_edges(picked, band_px=band), boxed > 0, band_px=band)
    lines = [{"kind": "roof", "edge": run["edge"], "polyline": run["polyline"], "visible": True,
              "planes": run["planes"], "inferred": run["inferred"], "length_px": run["length_px"], "pitch_apparent": run["pitch_apparent"]}
             for run in design.chain_segments(picked)]
    support = design.mask_to_data_url((s.prep.labels > 0).astype(np.uint8))
    return clean({"lines": lines, "support_mask": support, "planes_added": len(new)})


@app.post("/api/design/refine")
def design_refine(req: RefineRequest, user: dict = Depends(accounts.current_user)) -> dict:
    """One object from a click or a box, as a mask (SAM 3)."""
    s = get_session(req.session_id, user)
    if not sam.configured():
        raise HTTPException(409, "SAM 3 is off. Put FAL_KEY in .env and restart the server, or paint the area with the brush.")
    h, w = s.image.shape[:2]
    try:
        if req.box and len(req.box) == 4:
            x0, y0, x1, y1 = req.box
            box = (max(0, min(x0, x1)), max(0, min(y0, y1)), min(w - 1, max(x0, x1)), min(h - 1, max(y0, y1)))
            found = sam.box_instances(s.image, "", box)
            inside = np.zeros((h, w), bool)
            inside[box[1]:box[3] + 1, box[0]:box[2] + 1] = True
            found.sort(key=lambda i: float(((i["mask"] > 0) & inside).sum()) / max(1.0, float((i["mask"] > 0).sum())) + (i["score"] or 0), reverse=True)
            found = found[:1]
        elif req.point and len(req.point) == 2:
            found = sam.point_instance(s.image, (int(req.point[0]), int(req.point[1])))
        else:
            raise HTTPException(400, "Give a point or a box.")
    except sam.SamError as err:
        raise HTTPException(err.status, err.detail)
    if not found or not found[0]["mask"].any():
        raise HTTPException(404, "Nothing was found there. Try a tighter box, or paint it with the brush.")
    return {"mask": design.mask_to_data_url(found[0]["mask"]), "score": found[0]["score"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8765, reload=True)
