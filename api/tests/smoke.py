import io
import os
import sys
import tempfile
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

API = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API))

# The suite drops and rebuilds the schema, so it only ever runs against a *_test database,
# and it writes photos to a throwaway directory instead of the real store.
TEST_DB = os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://localhost/lightdesign_test")
if not TEST_DB.split("?")[0].rsplit("/", 1)[-1].endswith("_test"):
    raise SystemExit(f"Refusing to run: {TEST_DB} is not a *_test database.")
os.environ.setdefault("STORAGE_DIR", tempfile.mkdtemp(prefix="ld-smoke-"))
os.environ.setdefault("SEGMENTATION_PROVIDER", "mock")

from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from sqlalchemy import create_engine, text  # noqa: E402

with create_engine(TEST_DB).begin() as connection:
    connection.execute(text("DROP SCHEMA public CASCADE"))
    connection.execute(text("CREATE SCHEMA public"))
migrations = Config(str(API / "alembic.ini"))
migrations.set_main_option("sqlalchemy.url", TEST_DB)
command.upgrade(migrations, "head")

from app.main import app  # noqa: E402
from app.services import roofline  # noqa: E402

PASSWORD = "correct horse battery"
client = TestClient(app)


def sample_photo() -> bytes:
    image = Image.new("RGB", (900, 600), (140, 170, 200))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def check(label: str, condition: bool) -> None:
    print(f"{'PASS' if condition else 'FAIL'}  {label}")
    if not condition:
        raise SystemExit(1)


def register(session: TestClient, email: str, password: str = PASSWORD):
    return session.post(
        "/auth/register", json={"email": email, "name": "Smoke Test", "password": password}
    )


health = client.get("/health").json()
check("health responds", health["status"] == "ok")
check("database reachable", health["database"] == "ok")
check("provider is mock", health["segmentation_provider"] == "mock")

check("photos require sign-in", client.get("/photos").status_code == 401)
probe = client.get("/auth/me")
check("signed-out session probe answers null, not an error",
      probe.status_code == 200 and probe.json() is None)
check("upload requires sign-in", client.post(
    "/photos", files={"file": ("house.png", sample_photo(), "image/png")}
).status_code == 401)

email = f"smoke-{uuid.uuid4().hex[:8]}@example.com"
registered = register(client, email)
check("registration succeeds", registered.status_code == 201)
check("session cookie is HttpOnly", "httponly" in registered.headers.get("set-cookie", "").lower())
check("registration signs you in", client.get("/auth/me").json()["email"] == email)
check("duplicate email rejected, case-insensitively",
      register(TestClient(app), email.upper()).status_code == 409)
check("short password rejected", register(TestClient(app), "short@example.com", "short").status_code == 422)
check("malformed email rejected", register(TestClient(app), "not-an-email").status_code == 422)

upload = client.post(
    "/photos",
    files={"file": ("house.png", sample_photo(), "image/png")},
    data={"prompt": "C9 warm white on the front roofline"},
)
check("upload accepted", upload.status_code == 200)
photo_id = upload.json()["id"]
check("prompt stored", upload.json()["prompt"] == "C9 warm white on the front roofline")

check("mode set", client.put(f"/photos/{photo_id}/mode", json={"mode": "automatic"}).status_code == 200)

detected = client.post(f"/photos/{photo_id}/detect", json={"all_kinds": True})
check("detect responds", detected.status_code == 200)
regions = detected.json()
check("regions returned", len(regions) > 0)

lines = [r for r in regions if r["geometry"] == "line"]
areas = [r for r in regions if r["geometry"] == "area"]
check("roofline spans produced", len(lines) > 0)
check("roofline spans carry a side", all(r["side"] in {"front", "left", "right"} for r in lines))
check("roofline spans carry a polyline", all(len(r["polyline"]) >= 2 for r in lines))
check("area regions produced", len(areas) > 0)
check("area regions carry a mask", all(r["mask_url"] for r in areas))

kinds = {r["kind"] for r in regions}
for required in ["roof", "hedge", "shrub", "topiary", "palm_trunk", "palm_collar", "palm_frond", "tree_canopy"]:
    check(f"detected {required}", required in kinds)

mask_url = areas[0]["mask_url"].split("?")[0]
check("mask file served", client.get(mask_url).status_code == 200)

trunk = next(r for r in regions if r["kind"] == "palm_trunk")
frond = next(r for r in regions if r["kind"] == "palm_frond")
trunk_mask = np.array(Image.open(io.BytesIO(client.get(trunk["mask_url"].split("?")[0]).content)))
frond_mask = np.array(Image.open(io.BytesIO(client.get(frond["mask_url"].split("?")[0]).content)))
trunk_top = np.where(trunk_mask > 127)[0].min()
frond_bottom = np.where(frond_mask > 127)[0].max()
check("palm trunk truncated at husk base", trunk_top >= frond_bottom)
collar = next(r for r in regions if r["kind"] == "palm_collar")
collar_mask = np.array(Image.open(io.BytesIO(client.get(collar["mask_url"].split("?")[0]).content))) > 127
collar_rows = np.where(collar_mask.any(axis=1))[0]
check("palm collar sits between the fronds and the trunk",
      frond_bottom < collar_rows.min() and collar_rows.max() < trunk_top)
check("palm zones do not overlap", not (collar_mask & (trunk_mask > 127)).any()
      and not (collar_mask & (frond_mask > 127)).any())

polyline = [[10, 10], [400, 12], [800, 15]]
created = client.post(
    f"/photos/{photo_id}/regions",
    json={"kind": "roof", "geometry": "line", "side": "front", "polyline": polyline},
)
check("manual line region created", created.status_code == 200)

manual = client.post(f"/photos/{photo_id}/regions", json={"kind": "hedge", "geometry": "area"})
check("manual area region created", manual.status_code == 200)
manual_id = manual.json()["id"]

painted = Image.new("RGBA", (900, 600), (0, 0, 0, 0))
painted.paste((34, 197, 94, 255), (100, 400, 300, 500))
buffer = io.BytesIO()
painted.save(buffer, format="PNG")
put = client.put(
    f"/photos/{photo_id}/regions/{manual_id}/mask",
    files={"file": ("mask.png", buffer.getvalue(), "image/png")},
)
check("mask upload accepted", put.status_code == 200)

stored = np.array(Image.open(io.BytesIO(client.get(put.json()["mask_url"].split("?")[0]).content)))
check("uploaded mask persisted", bool((stored[450, 200] > 127) and (stored[100, 100] < 128)))
check("mask url is versioned", "?v=" in put.json()["mask_url"])

patched = client.patch(
    f"/photos/{photo_id}/regions/{lines[0]['id']}", json={"side": "left", "label": "Left rake"}
)
check("region patch applied", patched.json()["side"] == "left" and patched.json()["label"] == "Left rake")

refined = client.post(
    f"/photos/{photo_id}/regions/refine", json={"box": [500, 300, 700, 480], "kind": "shrub"}
)
check("box refine creates region", refined.status_code == 200 and refined.json()["mask_url"])

before = len(client.get(f"/photos/{photo_id}").json()["regions"])
check("region deleted", client.delete(f"/photos/{photo_id}/regions/{manual_id}").status_code == 204)
check("region count dropped", len(client.get(f"/photos/{photo_id}").json()["regions"]) == before - 1)

approved = client.post(f"/photos/{photo_id}/approve")
check("approve locks the mask set", approved.status_code == 200 and approved.json()["approved"])
check("reopen unlocks", client.post(f"/photos/{photo_id}/reopen").json()["approved"] is False)

check("missing photo is 404", client.get("/photos/deadbeef0000").status_code == 404)
check(
    "bad upload type rejected",
    client.post("/photos", files={"file": ("a.txt", b"nope", "text/plain")}).status_code == 415,
)

gable = np.zeros((400, 800), dtype=bool)
for x in range(800):
    top = 120 + abs(x - 400) // 4
    gable[top:360, x] = True
spans = roofline.extract_spans(gable, "ridge")
check("gable splits into spans", len(spans) >= 2)
check("gable spans classified", {side for _, side in spans} <= {"front", "left", "right"})

flat = np.zeros((400, 800), dtype=bool)
flat[150:360, 50:750] = True
flat_spans = roofline.extract_spans(flat)
check("flat roof reads as front", all(side == "front" for _, side in flat_spans))



occluded = np.zeros((400, 800), dtype=bool)
occluded[150:360, 50:750] = True
occluded[:, 380:405] = False
broken = roofline.extract_spans(occluded, "ridge")
check("occluder splits one eave into two spans", len(broken) == 2)
check("neither span crosses the occluder", all(
    not (min(p[0] for p in pts) < 380 and max(p[0] for p in pts) > 405) for pts, _ in broken
))

notched = np.zeros((400, 800), dtype=bool)
notched[150:360, 50:750] = True
notched[150:250, 380:405] = False
dipped = roofline.extract_spans(notched, "ridge")
check("vertical dip also splits the span", len(dipped) == 2)

band = np.zeros((400, 800), dtype=bool)
for x in range(800):
    band[140 + abs(x - 400) // 8 : 300, x] = True
eaves = roofline.extract_spans(band, "eave")
ridges = roofline.extract_spans(band, "ridge")
check("eave profile follows the mask bottom", bool(eaves) and all(
    all(p[1] > 290 for p in pts) for pts, _ in eaves
))
check("ridge profile follows the mask top", bool(ridges) and all(
    all(p[1] < 200 for p in pts) for pts, _ in ridges
))
check("eave and ridge differ", eaves[0][0] != ridges[0][0])

detected_edges = {r["edge"] for r in regions if r["geometry"] == "line"}
check("detect returns eave spans", "eave" in detected_edges)

from app.services import intent  # noqa: E402

real = intent.parse(
    "Convert this daytime photo to a realistic twilight/dusk scene with a deep blue sky. "
    "Add warm white mini lights wrapped around the trunks of all three royal palm trees "
    "from the base up to the bottom of the fronds. Add multicolor C9 LED bulbs with even "
    "spacing along the entire front roofline edge."
)
check("intent picks palm trunks and roof", real.kinds == ["roof", "palm_trunk"])
check("a palm named without a part lights all three zones",
      intent.parse("mini lights on the palms").kinds == ["palm_trunk", "palm_collar", "palm_frond"])
check("a collar instruction selects the collar alone",
      intent.parse("red collar on the palm trees").kinds == ["palm_collar"])
check("intent excludes fronds used as a boundary", "palm_frond" not in real.kinds)
check("intent picks front side only", real.sides == ["front"])
check("intent reads the entire front roofline as the full outline", real.edges == ["eave", "rake", "hip"])

check("intent ignores unrelated prompt", intent.parse("make it look nice").matched is False)
check("hedge prompt selects hedge only", intent.parse("mini lights in the hedges").kinds == ["hedge"])
check("gable maps to the sloping edges (rake and hip), not ridge",
      intent.parse("put C9 on the gable").edges == ["rake", "hip"])
check("rake is a roof term",
      intent.parse("C9 on the rakes").kinds == ["roof"] and intent.parse("C9 on the rakes").edges == ["rake", "hip"])
check("ridge still maps to ridge", intent.parse("light the ridge").edges == ["ridge"])

OUTLINE = ["eave", "rake", "hip"]
for phrase in ("C9 on the roofline", "C9 on the roof", "outline the roof", "C9 along the roof line"):
    check(f'"{phrase}" selects the full outline', intent.parse(phrase).edges == OUTLINE)
for phrase, wanted in (
    ("C9 on the eaves only", ["eave"]),
    ("C9 along the gutter line", ["eave"]),
    ("C9 along the bottom of the roof", ["eave"]),
    ("C9 along the top of the roof", ["ridge"]),
    ("C9 on the peak", ["ridge"]),
):
    check(f'"{phrase}" stays narrow ({"/".join(wanted)})', intent.parse(phrase).edges == wanted)
check("roofline plus ridge adds the ridge to the outline",
      intent.parse("C9 on the roofline and the ridge").edges == OUTLINE + ["ridge"])
check('"leave" is not read as "eave"', "eave" not in intent.parse("leave the ridge dark").edges)

intent_photo = client.post(
    "/photos",
    files={"file": ("h.png", sample_photo(), "image/png")},
    data={"prompt": "wrap the palm trunks, C9 on the front roofline"},
).json()["id"]
from app.services import detect  # noqa: E402

scoped = client.post(f"/photos/{intent_photo}/detect", json={"kinds": None}).json()
scoped_kinds = {r["kind"] for r in scoped if r["role"] == "target"}
check("detect honours the prompt", scoped_kinds == {"roof", "palm_trunk"})
# A roofline request also looks for what stands in front of the house, so a line that disappears
# behind a palm can be joined back up. Those are evidence, not lighting targets: they come back
# as blockers, never as something to light, and the instruction's own targets are unchanged.
check("what stands in front of the house is a blocker, never a lighting target",
      all(r["kind"] in detect.ROOF_BLOCKERS for r in scoped if r["role"] == "blocker")
      and not any(r["role"] == "target" and r["kind"] in ("tree_canopy", "obstruction") for r in scoped))
check("with blockers off a roof request asks the model for the roof and nothing else",
      detect.queried_kinds(["roof"], False) == ["roof"])
check("with blockers on a roof request also asks for what stands in front",
      set(detect.queried_kinds(["roof"], True)) == {"roof", *detect.ROOF_BLOCKERS})
check("a request with no roof asks for nothing extra", detect.queried_kinds(["hedge"], True) == ["hedge"])
check("detecting everything still includes what stands in front", "obstruction" in detect.DEFAULT_KINDS)
check("manual regions are lighting targets", all(
    r["role"] == "target" for r in client.get(f"/photos/{photo_id}").json()["regions"] if r["source"] == "manual"))
check("detect keeps front spans only", all(
    r["side"] == "front" for r in scoped if r["geometry"] == "line"
))

everything = client.post(f"/photos/{intent_photo}/detect", json={"all_kinds": True}).json()
check("all_kinds escape hatch still returns everything",
      len({r["kind"] for r in everything}) > len(scoped_kinds))

# Rakes carry their own label, so an instruction naming the gable must keep them through the
# edge filter. "gable" used to map to ridge, which silently returned no gable lines at all.
def detected_lines(prompt: str) -> list[dict]:
    photo = client.post(
        "/photos", files={"file": ("h.png", sample_photo(), "image/png")}, data={"prompt": prompt}
    ).json()["id"]
    # Lit lines only: partition mode also returns the edges the instruction did not ask for,
    # hidden (visible=False), so the editor holds the whole roof structure.
    return [r for r in client.post(f"/photos/{photo}/detect", json={"kinds": None}).json()
            if r["geometry"] == "line" and r["visible"]]


gable_lines = detected_lines("put C9 on the gable")
check("a gable instruction keeps the sloping spans", any(r["edge"] in ("rake", "hip") for r in gable_lines))
check("a gable instruction keeps only sloping edges or edges it could not identify",
      bool(gable_lines) and all(r["edge"] in {"rake", "hip", "unknown"} for r in gable_lines))
check("a ridge instruction no longer sweeps up rakes",
      not any(r["edge"] == "rake" for r in detected_lines("C9 along the ridge")))

# Side belongs to the whole plane. A front gable's rakes slope left and right, but they are on
# the front: a per-span slope test used to drop every one of them from "front roofline".
front_gable = np.zeros((400, 800), dtype=bool)
for x in range(100, 700):
    front_gable[120 + abs(x - 400) // 3 : 300, x] = True
check("a front gable plane faces the front", roofline.plane_side(front_gable, 800) == "front")
gable_spans = roofline.plane_spans(front_gable, 800)
check("every span on a front plane is front, rakes included",
      {side for _, side, _ in gable_spans} == {"front"} and "rake" in {edge for *_, edge in gable_spans})

bands = np.zeros((400, 800), dtype=bool)      # a gable masked only along its rakes
for x in range(100, 700):
    top = 120 + abs(x - 400) // 3
    bands[top : top + 18, x] = True
check("a gable masked only along its rakes still faces the front", roofline.plane_side(bands, 800) == "front")

receding = np.zeros((400, 800), dtype=bool)   # a side plane in perspective: its eave climbs
for x in range(40, 360):
    receding[150 : 300 - (x - 40) // 3, x] = True
check("a receding side plane keeps its own side", roofline.plane_side(receding, 800) in {"left", "right"})

upper = np.zeros((400, 800), dtype=bool)
lower = np.zeros((400, 800), dtype=bool)
for x in range(100, 700):
    border = 100 + (x - 100) // 2              # two planes meeting along a ~27 degree line
    upper[60:border, x] = True
    lower[border:360, x] = True
corner = roofline.seam_ridges([upper, lower], 800)
check("a sloping seam between two planes is a hip", bool(corner) and all(e == "hip" for *_, e in corner))

# A lower roof whose top edges run up into the wall of the storey above has nothing to hang lights
# on there. With sky above, the same edges are rakes and stay lit.
lower_roof = np.zeros((400, 800), dtype=bool)
for x in range(200, 600):
    lower_roof[200 + abs(x - 400) // 3 : 330, x] = True
storey_roof = np.zeros((400, 800), dtype=bool)
storey_roof[120:190, 150:650] = True
beside_roof = np.zeros((400, 800), dtype=bool)
beside_roof[120:190, 620:780] = True
abutting_roof = np.zeros((400, 800), dtype=bool)
for x in range(200, 600):
    abutting_roof[120 : 200 + abs(x - 400) // 3, x] = True


def lower_edges(others: list[np.ndarray]) -> list[str]:
    masks = [lower_roof] + others
    planes = roofline.mark_junctions([roofline.plane_spans(m, 800) for m in masks], masks, 800)
    return [edge for *_, edge in planes[0]]


check("with sky above, a lower roof keeps its rakes",
      "rake" in lower_edges([]) and "junction" not in lower_edges([]))
check("under the wall of the storey above, its top edges are junctions",
      "junction" in lower_edges([storey_roof]) and "rake" not in lower_edges([storey_roof]))
check("a junction never takes the eave", lower_edges([storey_roof]).count("eave") == lower_edges([]).count("eave"))
check("a roof beside it rather than above it changes nothing", lower_edges([beside_roof]) == lower_edges([]))
check("a roof directly abutting above is a seam, not a junction", "junction" not in lower_edges([abutting_roof]))

from app.routers.regions import roof_lines  # noqa: E402

outline_lines = roof_lines([lower_roof, storey_roof], None, intent.parse("C9 on the roofline"), True)
check("a roofline request leaves junctions dark", bool(outline_lines) and all(e != "junction" for *_, e in outline_lines))
check("junctions are still returned when no edge is asked for",
      "junction" in {e for *_, e in roof_lines([lower_roof, storey_roof], None, intent.Intent(), True)})

# Simplifying an outline can cut straight across a concave edge. Such a side follows the mask.
bowed = np.zeros((400, 800), dtype=bool)
for x in range(100, 700):
    bowed[150 : 300 - int(20 * (1 - ((x - 400) / 300) ** 2)), x] = True
bowed_near = cv2.dilate(bowed.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
bowed_sides = roofline._polygon_spans(bowed, 800)
check("an outline side that misses its plane is traced along the mask",
      any(len(points) > 2 for points, *_ in bowed_sides)
      and all(roofline._on_share(bowed_near, points) >= 0.9 for points, *_ in bowed_sides))
check("a traced side keeps the corners it joins", [[100, 299], [699, 299]] == [
    [points[0], points[-1]] for points, *_ in bowed_sides if len(points) > 2][0])
check("sides already on the plane stay straight chords",
      sum(len(points) == 2 for points, *_ in bowed_sides) == len(bowed_sides) - 1)


def low_edge(degrees: float) -> set[str]:
    plane = np.zeros((400, 800), dtype=bool)
    for x in range(200, 600):
        plane[100 : 250 + int((x - 200) * np.tan(np.radians(degrees))), x] = True
    return {edge for points, _, edge in roofline.plane_spans(plane, 800) if points[0][1] > 200}


check("a low edge tilted 12 degrees is an eave", low_edge(12) == {"eave"})
check("a low edge tilted 20 degrees stays unidentified", low_edge(20) == {"unknown"})

piece_a = ([[0, 0], [100, 0]], "front", "eave")
piece_b = ([[100, 0], [200, 10]], "front", "eave")
piece_c = ([[200, 10], [300, 60]], "front", "rake")
piece_d = ([[300, 60], [400, 60]], "front", "rake")
seam_line = ([[0, 100], [50, 150]], "front", "hip")
joined_pieces = roofline.join_pieces([piece_a, piece_b, piece_c, piece_d, seam_line], [[piece_a, piece_b, piece_c], [piece_d]])
check("pieces of one plane's edge that meet end to end become one line",
      ([[0, 0], [100, 0], [200, 10]], "front", "eave") in joined_pieces)
check("pieces are never joined across edge types or planes",
      piece_c in joined_pieces and piece_d in joined_pieces and seam_line in joined_pieces and len(joined_pieces) == 4)
check("plane-relative minimum length is staged off", roofline.PLANE_MIN_SPAN is False)

# ---- snapping a line onto the edge it follows ----------------------------------------------
from app.config import get_settings  # noqa: E402
from app.services import occlusion, snap  # noqa: E402

photo_edge = np.full((200, 400, 3), 30, dtype=np.uint8)
photo_edge[120:, :] = 220
edged = snap.field(Image.fromarray(photo_edge))
check("a line above the edge is pulled onto it",
      snap.line([[20, 112], [380, 112]], edged[:2], edged[2])[0][1] in range(118, 123))
check("a line below the edge is pulled up onto it",
      snap.line([[20, 128], [380, 128]], edged[:2], edged[2])[0][1] in range(118, 123))
check("an edge outside the search band is left alone",
      snap.line([[20, 90], [380, 90]], edged[:2], edged[2]) == [[20, 90], [380, 90]])
flat = snap.field(Image.new("RGB", (300, 200), (120, 140, 160)))
check("a photo with no edges moves nothing",
      snap.line([[20, 100], [280, 100]], flat[:2], flat[2]) == [[20, 100], [280, 100]])
bent = snap.line([[20, 60], [200, 118], [380, 60]], edged[:2], edged[2])
check("a line that turns keeps its corner", len(bent) == 3 and bent[1][1] > bent[0][1] < 200)
check("snapping never changes a line's side or edge type",
      [s[1:] for s in snap.lines([([[20, 112], [380, 112]], "front", "eave")], edged[:2], edged[2])]
      == [("front", "eave")])
check("snapping is on by default, interior hips are not",
      get_settings().snap_to_edges and not get_settings().interior_hips)

# ---- a run that stops at a blocker survives, if it joins up ---------------------------------
split_roof = np.zeros((400, 800), dtype=bool)
split_roof[150:300, 60:260] = True        # 200px of eave
split_roof[150:300, 330:520] = True       # and 190px more, with a gap between
palm = np.zeros((400, 800), dtype=bool)
palm[100:350, 255:335] = True
palm_union = occlusion.blocker_union([palm], (400, 800))
alone = roofline.extract_spans(split_roof, "eave")
beside = roofline.extract_spans(split_roof, "eave", palm_union)
check("pieces long enough on their own are found either way", len(alone) == len(beside) == 2)
check("a plane's own lines do not change when something stands in front of it",
      roofline.plane_spans(split_roof, 800) == roofline.plane_spans(split_roof, 800))

nibbled = np.zeros((400, 800), dtype=bool)
nibbled[150:300, 60:100] = True           # 40px: under the 48px minimum for this width
nibbled[150:300, 180:220] = True
covering = occlusion.blocker_union([np.pad(np.ones((250, 80), dtype=bool), ((100, 50), (100, 620)))], (400, 800))
check("a short piece away from anything is dropped", roofline.extract_spans(nibbled, "eave") == [])
check("a short piece that stops at a blocker is kept",
      len(roofline.extract_spans(nibbled, "eave", covering)) == 2)
check("blocker pieces are offered on top of the plane's own lines",
      len(roofline.blocker_pieces(nibbled, 800, covering)) == 4
      and roofline.blocker_pieces(nibbled, 800, None) == [])
joined_over = [s for s in roof_lines([nibbled], covering, intent.Intent(), True) if s[2] == "eave"]
check("and is only kept if it joins up across the blocker",
      len(joined_over) == 1 and joined_over[0][0][0][0] < 100 < joined_over[0][0][-1][0])
elsewhere = occlusion.blocker_union([np.pad(np.ones((250, 20), dtype=bool), ((100, 50), (0, 780)))], (400, 800))
check("a stub that never joins is dropped", all(
    min(p[0] for p in span) >= 100
    for span, *_ in roof_lines([nibbled], elsewhere, intent.Intent(), True)))

# ---- partition: one roof edge, one line, straight through whatever stands in front ----------
# A roof band severed by two palm trunks with a sliver of roof between them, a second band
# across an open gap nothing covers, and a third whose far end a trunk hides.
tall = np.zeros((1000, 2400), dtype=bool); tall[600:700, 200:1000] = True
t1 = np.zeros((1000, 2400), dtype=bool); t1[:, 500:560] = True
t2 = np.zeros((1000, 2400), dtype=bool); t2[:, 590:650] = True
tall &= ~(t1 | t2)
right = np.zeros((1000, 2400), dtype=bool); right[600:700, 1300:2100] = True
t3 = np.zeros((1000, 2400), dtype=bool); t3[:, 1900:1960] = True
right &= ~t3
palms = occlusion.blocker_union([t1, t2, t3], (1000, 2400))
bare = roof_lines([tall, right], None, intent.Intent(), False, partitioned=True, scores=[0.9, 0.9], junctions=True)
through = roof_lines([tall, right], palms, intent.Intent(), False, solid=palms, partitioned=True,
                     scores=[0.9, 0.9], junctions=True)
bare_eaves = sorted([s for s in bare if s[2] == "eave"], key=lambda s: s[0][0][0])
eaves = sorted([s for s in through if s[2] == "eave"], key=lambda s: s[0][0][0])
check("without the palms the severed band comes back in pieces", len(bare_eaves) > len(eaves))
check("a roof edge two trunks cut apart is one straight line again",
      len(eaves) == 2 and eaves[0][0][0][0] <= 205 and eaves[0][0][-1][0] >= 995 and len(eaves[0][0]) == 2)
check("the line reports the stretches it passes behind",
      len(occlusion.hidden_stretches(eaves[0][0], palms, 2400)) == 2)
check("a gap nothing stands in front of is left open", eaves[1][0][0][0] >= 1295)
check("a band whose far end a trunk hides is carried to where the roof is known to reach",
      1955 <= eaves[1][0][-1][0] <= 2105)
check("no line steeper than a roof can be is lit",
      not [s for s in through if s[2] in ("eave", "rake", "hip")
           and abs(s[0][-1][0] - s[0][0][0]) < abs(s[0][-1][1] - s[0][0][1])])

front_lines = detected_lines("C9 on the front roofline")
check("front roofline keeps the front plane's sloping edges as well as its eave",
      "eave" in {r["edge"] for r in front_lines} and {"rake", "hip"} & {r["edge"] for r in front_lines})
check("front roofline keeps only front spans",
      bool(front_lines) and all(r["side"] == "front" for r in front_lines))
check("front gable keeps the front plane's sloping edges",
      any(r["edge"] in ("rake", "hip") and r["side"] == "front" for r in detected_lines("put C9 on the front gable")))

from app.providers.prompts import collapse_facade_boxes  # noqa: E402

facade = (0.60, (24, 183, 603, 288))
roof_band = (0.52, (24, 180, 617, 230))
elsewhere = (0.40, (700, 300, 900, 340))
kept = collapse_facade_boxes([facade, roof_band, elsewhere], 358)
check("facade box collapsed onto the roof band", facade not in kept)
check("roof band survives", roof_band in kept)
check("unrelated box untouched", elsewhere in kept)

stacked = (0.5, (10, 10, 200, 40))
lower = (0.5, (10, 120, 200, 200))
check("boxes with different tops are not collapsed",
      len(collapse_facade_boxes([stacked, lower], 400)) == 2)
check("every kind has a colour", set(detect.COLORS) == set(detect.LABELS))
check("colours are unique per kind", len(set(detect.COLORS.values())) == len(detect.COLORS))
check("house kinds present", {"window", "door", "garage_door", "column", "wreath"} <= set(detect.COLORS))
check("edge-lighting line kinds present", {"driveway_edge", "walkway_edge", "fence"} <= detect.LINE_KINDS)
check("detection still only claims detectable kinds",
      set(detect.DEFAULT_KINDS) <= set(detect.COLORS) and "window" not in detect.DEFAULT_KINDS)

drawn = client.post(
    f"/photos/{photo_id}/regions",
    json={"kind": "driveway_edge", "geometry": "line", "side": "front",
          "polyline": [[10, 300], [300, 320], [600, 330]]},
)
check("manual line region can be drawn", drawn.status_code == 200)
check("drawn line is marked manual", drawn.json()["source"] == "manual")
check("drawn line gets its kind colour", drawn.json()["color"] == detect.COLORS["driveway_edge"])

roofline_manual = client.post(
    f"/photos/{photo_id}/regions",
    json={"kind": "roof", "geometry": "line", "side": "left", "polyline": [[5, 5], [200, 40]]},
)
check("roofline can be added without detection", roofline_manual.status_code == 200)

no_points = client.post(f"/photos/{photo_id}/regions", json={"kind": "fence", "geometry": "line"})
check("line region without a polyline is rejected", no_points.status_code == 400)

tinted = client.post(
    f"/photos/{photo_id}/regions",
    json={"kind": "custom", "geometry": "area", "color": "#123abc"},
)
check("custom region accepts a chosen colour", tinted.json()["color"] == "#123abc")

recoloured = client.patch(
    f"/photos/{photo_id}/regions/{tinted.json()['id']}", json={"color": "#ff8800"}
)
check("region colour can be changed", recoloured.json()["color"] == "#ff8800")
check("bad colour rejected", client.patch(
    f"/photos/{photo_id}/regions/{tinted.json()['id']}", json={"color": "red"}
).status_code == 422)

big = Image.new("RGB", (4000, 3000), (150, 170, 190))
buffer = io.BytesIO()
big.save(buffer, format="PNG")
big_upload = client.post(
    "/photos",
    files={"file": ("big.png", buffer.getvalue(), "image/png")},
    data={"prompt": "roofline"},
).json()
check("full resolution is preserved on upload",
      (big_upload["width"], big_upload["height"]) == (4000, 3000))

stored = Image.open(io.BytesIO(client.get(big_upload["image_url"]).content))
check("stored original keeps its pixels", stored.size == (4000, 3000))

big_regions = client.post(f"/photos/{big_upload['id']}/detect", json={"all_kinds": True}).json()
check("detection still returns regions on a full-res photo", len(big_regions) > 0)
big_area = next(r for r in big_regions if r["geometry"] == "area")
big_mask = Image.open(io.BytesIO(client.get(big_area["mask_url"].split("?")[0]).content))
check("masks come back in full-resolution coordinates", big_mask.size == (4000, 3000))
big_line = next((r for r in big_regions if r["geometry"] == "line"), None)
if big_line:
    check("polylines are in full-resolution coordinates",
          max(p[0] for p in big_line["polyline"]) > 2048)

clicked = client.post(
    f"/photos/{photo_id}/regions/refine",
    json={"point": [450, 300], "kind": "shrub"},
)
check("click adds an object by point prompt", clicked.status_code == 200)
check("point-added region has a mask", bool(clicked.json()["mask_url"]))

check("refine rejects both box and point", client.post(
    f"/photos/{photo_id}/regions/refine",
    json={"point": [10, 10], "box": [1, 2, 3, 4]},
).status_code == 422)
check("refine rejects neither box nor point", client.post(
    f"/photos/{photo_id}/regions/refine", json={"kind": "shrub"}
).status_code == 422)

from app.services import occlusion  # noqa: E402

notched = np.zeros((400, 800), dtype=bool)
notched[150:300, 50:750] = True
notched[200:300, 380:420] = False  # something in front cuts into the eave
pieces = [(p, s, "eave") for p, s in roofline.extract_spans(notched, "eave")]
check("a notch splits the eave in two", len(pieces) == 2)
check("a gap with no detected object in front is left alone",
      len(occlusion.bridge(pieces, None, 800)) == 2)
tree = np.zeros((400, 800), dtype=bool)
tree[100:350, 370:430] = True
blockers = occlusion.blocker_union([tree], (400, 800))
joined = occlusion.bridge(pieces, blockers, 800)
check("a gap a detected object covers is joined back up", len(joined) == 1)
hidden = occlusion.hidden_stretches(joined[0][0], blockers, 800)
check("the stretch behind the object is measured",
      len(hidden) == 1 and abs((hidden[0][1] - hidden[0][0]) - 62) <= 6)

stepped = np.zeros((400, 800), dtype=bool)
stepped[150:300, 50:380] = True
stepped[150:260, 420:750] = True  # the next roof section's eave sits 40px higher
steps = [(p, s, "eave") for p, s in roofline.extract_spans(stepped, "eave")]
check("pieces at different heights are never joined, even with an object between",
      len(steps) == 2 and len(occlusion.bridge(steps, blockers, 800)) == 2)
check("a line clear of every object has no hidden stretch",
      occlusion.hidden_stretches([[50, 50], [750, 50]], blockers, 800) == [])

# A photo with nothing detected on it, so the only object in front of the line is the one
# painted here — the detected photo above already has a tree and a palm across y=300.
clear_id = client.post(
    "/photos", files={"file": ("clear.png", sample_photo(), "image/png")}, data={"prompt": ""}
).json()["id"]
behind = client.post(
    f"/photos/{clear_id}/regions",
    json={"kind": "roof", "geometry": "line", "side": "front", "polyline": [[100, 300], [800, 300]]},
).json()
check("a new line with nothing in front has no hidden stretch", not behind["hidden"])
car = client.post(f"/photos/{clear_id}/regions", json={"kind": "obstruction", "geometry": "area"}).json()
painted = Image.new("RGBA", (900, 600), (0, 0, 0, 0))
painted.paste((148, 163, 184, 255), (400, 250, 500, 350))
buffer = io.BytesIO()
painted.save(buffer, format="PNG")
client.put(f"/photos/{clear_id}/regions/{car['id']}/mask",
           files={"file": ("mask.png", buffer.getvalue(), "image/png")})


def line_now(region_id: str) -> dict:
    return next(r for r in client.get(f"/photos/{clear_id}").json()["regions"] if r["id"] == region_id)


covered = line_now(behind["id"])["hidden"] or []
check("painting an obstruction hides the stretch behind it",
      len(covered) == 1 and abs(sum(e - s for s, e in covered) - 100) <= 8)
moved = client.patch(f"/photos/{clear_id}/regions/{behind['id']}",
                     json={"polyline": [[100, 100], [800, 100]]}).json()
check("moving the line clear of the obstruction clears it", not moved["hidden"])
client.patch(f"/photos/{clear_id}/regions/{behind['id']}", json={"polyline": [[100, 300], [800, 300]]})
check("moving it back hides it again", bool(line_now(behind["id"])["hidden"]))
client.delete(f"/photos/{clear_id}/regions/{car['id']}")
check("deleting the obstruction clears the hidden stretch", not line_now(behind["id"])["hidden"])

intruder = TestClient(app)
register(intruder, f"intruder-{uuid.uuid4().hex[:8]}@example.com")
check("another user cannot open the photo", intruder.get(f"/photos/{photo_id}").status_code == 404)
check("another user cannot fetch its image",
      intruder.get(f"/files/photos/{photo_id}/original.png").status_code == 404)
check("another user cannot edit its regions", intruder.patch(
    f"/photos/{photo_id}/regions/{lines[0]['id']}", json={"label": "taken"}
).status_code == 404)
check("photo lists are per user", photo_id not in {p["id"] for p in intruder.get("/photos").json()})
check("owner's list includes the photo", photo_id in {p["id"] for p in client.get("/photos").json()})
check("files require sign-in",
      TestClient(app).get(f"/files/photos/{photo_id}/original.png").status_code == 401)
check("path traversal refused",
      client.get(f"/files/photos/{photo_id}/..%2f..%2f..%2fetc%2fpasswd").status_code == 404)

# ---- product library ----------------------------------------------------------------------
catalogue = client.get("/products").json()
check("built-in catalogue covers every product type",
      {p["category"] for p in catalogue if p["builtin"]} == {"c9", "c7", "mini", "icicle", "wreath", "garland"})
check("mixed-colour mini strands are products with their colour sequence",
      any(p["category"] == "mini" and p["colours"] == ["warm_white", "cool_white"] for p in catalogue))
check("twinkle is product data for both C9 and mini lights",
      {p["category"] for p in catalogue if p["twinkle"]} >= {"c9", "mini"})
check("category filter", {p["category"] for p in client.get("/products?category=wreath").json()} == {"wreath"})
palette = {c["key"] for c in client.get("/products/palette").json()}
check("palette offers the whites and the common colours", {"warm_white", "cool_white", "red", "green"} <= palette)
check("products require sign-in", TestClient(app).get("/products").status_code == 401)

strand = client.post("/products", json={
    "category": "mini", "name": "Mini, red/white/green", "bulb_diameter_in": 0.2, "spacing_in": 4,
    "strand_length_ft": 24, "bulb_count": 70, "colours": ["red", "pure_white", "green"],
})
check("a custom mixed strand can be added", strand.status_code == 201 and not strand.json()["builtin"])
strand_id = strand.json()["id"]
check("a socket line refuses built-in colours", client.post("/products", json={
    "category": "c9", "name": "Bad C9", "bulb_diameter_in": 1.125, "spacing_in": 12, "colours": ["red"],
}).status_code == 422)
check("a strand must say its spacing", client.post("/products", json={
    "category": "mini", "name": "No spacing", "bulb_diameter_in": 0.2, "strand_length_ft": 24, "colours": ["red"],
}).status_code == 422)
check("unknown colours are refused", client.post("/products", json={
    "category": "wreath", "name": "Wreath", "diameter_in": 24, "colours": ["chartreuse"],
}).status_code == 422)
check("a default spacing outside the offered spacings is refused", client.post("/products", json={
    "category": "c7", "name": "C7", "bulb_diameter_in": 0.875, "spacing_in": 10, "spacing_options_in": [12, 18],
}).status_code == 422)
check("own product can be edited", client.patch(f"/products/{strand_id}", json={"spacing_in": 6}).json()["spacing_in"] == 6)
check("an edit is checked as the whole product it makes",
      client.patch(f"/products/{strand_id}", json={"category": "c9"}).status_code == 422)
builtin_id = next(p["id"] for p in catalogue if p["builtin"])
check("built-in products cannot be edited", client.patch(f"/products/{builtin_id}", json={"name": "x"}).status_code == 403)
check("built-in products cannot be deleted", client.delete(f"/products/{builtin_id}").status_code == 403)
check("another user sees the built-in catalogue", any(p["id"] == builtin_id for p in intruder.get("/products").json()))
check("another user cannot see my products", intruder.get(f"/products/{strand_id}").status_code == 404
      and all(p["id"] != strand_id for p in intruder.get("/products").json()))
check("own product can be deleted", client.delete(f"/products/{strand_id}").status_code == 204)
check("a deleted product is gone", client.get(f"/products/{strand_id}").status_code == 404)

# ---- colour patterns ----------------------------------------------------------------------
from types import SimpleNamespace  # noqa: E402

from app.schemas import LightingSettings, Pattern  # noqa: E402
from app.services import patterns  # noqa: E402

red_white = Pattern(steps=[{"colour": "red"}, {"colour": "pure_white"}])
check("a two-colour sequence repeats", patterns.colours(red_white, 5) == ["red", "pure_white"] * 2 + ["red"])
three = Pattern(steps=[{"colour": "red", "count": 2}, {"colour": "green"}, {"colour": "pure_white"}])
check("steps can repeat a colour", patterns.colours(three, 6) == ["red", "red", "green", "pure_white", "red", "red"])
check("offset shifts where the sequence starts",
      patterns.colours(Pattern(steps=red_white.steps, offset=1), 3) == ["pure_white", "red", "pure_white"])
check("reverse starts the sequence at the far end",
      patterns.colours(Pattern(steps=three.steps, direction="reverse"), 3) == ["green", "red", "red"])

c9 = SimpleNamespace(name="C9 LED", category="c9", colours=[], spacing_in=12.0,
                     spacing_options_in=[12.0, 15.0, 18.0, 24.0, 36.0], twinkle=False)
mixed = SimpleNamespace(name="Mini WW/CW", category="mini", colours=["warm_white", "cool_white"],
                        spacing_in=4.0, spacing_options_in=[], twinkle=True)
product_id = uuid.uuid4()
candy_cane = LightingSettings(product_id=product_id, pattern={
    "steps": [{"colour": "red"}, {"colour": "pure_white"}], "unit": "turn"}, wrap={"start": 0.1, "stop": 0.9})
check("a candy-cane trunk wrap is valid", patterns.problems(candy_cane, c9, "palm_trunk") == [])
check("per-turn colours need a wrapped zone", patterns.problems(candy_cane.model_copy(update={"wrap": None}), c9, "roof") != [])
check("only trunks and collars are wrapped",
      any("wrapped" in p for p in patterns.problems(candy_cane, c9, "tree_canopy")))
canopy = LightingSettings(product_id=product_id, spacing_in=18, pattern={
    "steps": [{"colour": "red"}, {"colour": "green"}, {"colour": "pure_white"}]})
check("a canopy in three C9 colours at 18 inch spacing is valid", patterns.problems(canopy, c9, "tree_canopy") == [])
check("a spacing the product is not sold at is refused",
      patterns.problems(canopy.model_copy(update={"spacing_in": 10}), c9, "tree_canopy") != [])
check("a socket line needs a pattern", patterns.problems(LightingSettings(product_id=product_id), c9, "roof") != [])
check("a mixed strand keeps its own colours",
      patterns.problems(LightingSettings(product_id=product_id, pattern=red_white.model_dump()), mixed, "hedge") != [])
pattern, spacing, twinkle, wrap = patterns.resolve(LightingSettings(product_id=product_id), mixed, "palm_frond")
check("a mixed strand lights with its built-in sequence and twinkle",
      patterns.colours(pattern, 3) == ["warm_white", "cool_white", "warm_white"] and twinkle and spacing == 4.0)
check("a trunk with no wrap given is wrapped top to bottom",
      patterns.resolve(candy_cane.model_copy(update={"wrap": None}), c9, "palm_trunk")[3].stop == 1.0)
check("wreaths only go on wreath regions",
      patterns.problems(LightingSettings(product_id=product_id), SimpleNamespace(
          name="Wreath", category="wreath", colours=["warm_white"], spacing_in=None, spacing_options_in=[],
          twinkle=False), "roof") != [])
try:
    Pattern(steps=[{"colour": "chartreuse"}])
    check("patterns refuse colours outside the palette", False)
except ValueError:
    check("patterns refuse colours outside the palette", True)
try:
    LightingSettings(product_id=product_id, wrap={"start": 0.8, "stop": 0.2})
    check("a wrap cannot stop below its start", False)
except ValueError:
    check("a wrap cannot stop below its start", True)

# ---- instruction -> specification -> confirmed design -------------------------------------
spec = intent.specification(
    "C9 warm white on the front roofline at 12 inch spacing, mini lights in the hedges, "
    "24 inch wreath on the front door"
)
check("an instruction splits into one request per clause", len(spec.requests) == 3)
first, second, third = spec.requests
check("a request keeps its place and its lighting together",
      first.kinds == ["roof"] and first.sides == ["front"] and first.edges == OUTLINE
      and first.lighting.category == "c9" and first.lighting.colours == ["warm_white"] and first.lighting.spacing_in == 12)
check("the second clause is its own request", second.kinds == ["hedge"] and second.lighting.category == "mini"
      and second.lighting.colours == [])
check("a wreath is a hand-drawn wreath region with a size, not a spacing",
      third.kinds == ["wreath"] and third.manual == ["wreath"] and third.lighting.size_in == 24 and third.lighting.spacing_in is None)
check("the union still drives detection", spec.intent.kinds == ["roof", "hedge"] and spec.intent.sides == ["front"])
check("colour counts are read",
      intent.specification("two red one white C9 along the eaves").requests[0].lighting.counts == [2, 1])
check("a colour sequence keeps its order",
      intent.specification("red and green C9 on the roof").requests[0].lighting.colours == ["red", "green"])
check("a lighting-only clause refines the request before it",
      intent.specification("C9 on the roof, twinkling, 15 inch spacing").requests[0].lighting.twinkle is True
      and intent.specification("C9 on the roof, twinkling, 15 inch spacing").requests[0].lighting.spacing_in == 15)
check("'all' refines every request", all(
    r.lighting.colours == ["warm_white"] for r in
    intent.specification("C9 on the roofline and mini lights in the hedges, all warm white").requests))
check("a place with no lighting takes the lighting before it",
      intent.specification("red C9 on the roof and on the hedges").requests[0].kinds == ["roof", "hedge"])
check("no product named splits by what each place takes by default",
      [r.kinds for r in intent.specification("warm white on the roof and the hedges").requests] == [["roof"], ["hedge"]])
check("multicolour is the catalogue sequence",
      intent.specification("multicolour minis on the palms").requests[0].lighting.colours == ["red", "green", "blue", "amber"])
check("candy cane is red and white per turn",
      intent.specification("candy cane wrap on the palm trunks").requests[0].lighting.per_turn is True)
check("a colour we do not carry is reported, not guessed",
      intent.specification("teal C9 on the roof").requests[0].lighting.unread == ["teal"])
check("steady means no twinkle", intent.specification("steady white C9 on the gutters").requests[0].lighting.twinkle is False)
check("an instruction with no lighting gives no requests", intent.specification("make it look nice").requests == [])

spec_photo = client.post(
    "/photos", files={"file": ("h.png", sample_photo(), "image/png")},
    data={"prompt": "C9 warm white on the front roofline at 12 inch spacing, mini lights in the hedges, "
                    "24 inch wreath on the front door"},
).json()
requests = spec_photo["interpreted"]["requests"]
check("upload reads the instruction before any detection", len(requests) == 3)
check("a socket line resolves with its pattern and spacing",
      requests[0]["product_name"] == "C9 LED" and requests[0]["settings"]["spacing_in"] == 12
      and requests[0]["settings"]["pattern"]["steps"] == [{"colour": "warm_white", "count": 1}] and requests[0]["problems"] == [])
check("a strand with no colour named is warm white, and says so",
      requests[1]["product_name"] == "Mini lights, warm white" and "warm white" in requests[1]["assumed"])
check("a wreath resolves to the size asked for", requests[2]["product_name"] == "Wreath, 24 in, warm white")
reread = client.post(f"/photos/{spec_photo['id']}/interpret", json={"prompt": "blue mini lights in the hedges"}).json()
unmatched = reread["interpreted"]["requests"][0]
check("re-reading needs no detection and replaces the instruction", reread["prompt"] == "blue mini lights in the hedges")
check("a strand the catalogue lacks is a problem with choices, not a guess",
      unmatched["product_id"] is None and unmatched["problems"] and unmatched["choices"])
odd = client.post(f"/photos/{spec_photo['id']}/interpret", json={"prompt": "C9 at 10 inch spacing on the roof"}).json()
check("a spacing the product is not sold at is reported and left unset",
      odd["interpreted"]["requests"][0]["settings"]["spacing_in"] is None
      and any("spacing" in p for p in odd["interpreted"]["requests"][0]["problems"]))
back = client.post(f"/photos/{spec_photo['id']}/interpret", json={"prompt": spec_photo["prompt"]}).json()
check("nothing is confirmed until the user confirms", client.get(f"/photos/{spec_photo['id']}/design").json() is None)

client.put(f"/photos/{spec_photo['id']}/mode", json={"mode": "automatic"})
client.post(f"/photos/{spec_photo['id']}/detect", json={"kinds": None})
spec_regions = client.get(f"/photos/{spec_photo['id']}").json()["regions"]
lit_roof = {r["id"] for r in spec_regions if r["kind"] == "roof" and r["visible"] and r["edge"] in OUTLINE and r["side"] == "front"}
confirmable = [r for r in back["interpreted"]["requests"] if r["settings"]]
design = client.put(f"/photos/{spec_photo['id']}/design", json={"requests": confirmable})
check("confirming puts each request's settings on the regions it lights", design.status_code == 200
      and set(design.json()["requests"][0]["region_ids"]) == lit_roof and lit_roof)
check("every lit region carries the confirmed product", all(
    row["settings"]["product_id"] == requests[0]["product_id"]
    for row in design.json()["regions"] if row["region_id"] in lit_roof))
check("a request with nothing drawn yet is kept with no regions",
      design.json()["requests"][-1]["kinds"] == ["wreath"] and design.json()["requests"][-1]["region_ids"] == [])
check("the confirmed design is read back", client.get(f"/photos/{spec_photo['id']}/design").json()["id"] == design.json()["id"])
bad_spacing = {**confirmable[0], "settings": {**confirmable[0]["settings"], "spacing_in": 10}}
check("confirming a spacing the product is not sold at is refused",
      client.put(f"/photos/{spec_photo['id']}/design", json={"requests": [bad_spacing]}).status_code == 422)
wreath_on_roof = {**confirmable[0], "settings": confirmable[-1]["settings"]}
check("a wreath cannot be confirmed onto a roofline",
      client.put(f"/photos/{spec_photo['id']}/design", json={"requests": [wreath_on_roof]}).status_code == 422)
check("re-confirming replaces the design rather than adding to it",
      client.put(f"/photos/{spec_photo['id']}/design", json={"requests": confirmable[:1]}).json()["regions"].__len__() == len(lit_roof)
      and client.get(f"/photos/{spec_photo['id']}/design").json()["id"] == design.json()["id"])
own = client.post("/products", json={"category": "c9", "name": "My C9", "bulb_diameter_in": 1.125, "spacing_in": 12}).json()
mine = {**confirmable[0], "settings": {**confirmable[0]["settings"], "product_id": own["id"]}}
client.put(f"/photos/{spec_photo['id']}/design", json={"requests": [mine]})
check("a product in a confirmed design cannot be deleted", client.delete(f"/products/{own['id']}").status_code == 409)
check("another user cannot read my design", intruder.get(f"/photos/{spec_photo['id']}/design").status_code == 404)

# ---- a painted obstacle mends the roofline behind it ------------------------------------------
# The model missed a palm: the eave came back in two pieces either side of it, plus a third
# piece across a real gap. Painting the trunk joins the two; the real gap is left alone; a
# line someone drew by hand is never touched.
from app.db import _sessions  # noqa: E402
from app.models import Photo as PhotoRow, Region as RegionRow, User as UserRow  # noqa: E402

mend_photo = client.post(
    "/photos", files={"file": ("h.png", sample_photo(), "image/png")}, data={"prompt": "C9 on the roofline"}
).json()["id"]
with _sessions()() as db:
    owner = db.get(UserRow, db.get(PhotoRow, uuid.UUID(mend_photo)).owner_id)
    for start, end in ((100, 400), (460, 800), (840, 890)):          # trunk will cover 400-460; 800-840 stays a real gap
        db.add(RegionRow(id=uuid.uuid4(), photo_id=uuid.UUID(mend_photo), kind="roof", role="target",
                         label="Eave", geometry="line", source="auto", color="#39ff14", score=0.5,
                         visible=True, side="front", edge="eave", polyline=[[start, 300], [end, 300]]))
    db.add(RegionRow(id=uuid.uuid4(), photo_id=uuid.UUID(mend_photo), kind="roof", role="target",
                     label="Drawn", geometry="line", source="manual", color="#39ff14", score=1.0,
                     visible=True, side="front", edge="eave", polyline=[[100, 340], [400, 340]]))
    db.commit()
trunk_region = client.post(f"/photos/{mend_photo}/regions", json={"kind": "palm_trunk", "geometry": "area"}).json()
trunk_png = io.BytesIO()
trunk_image = Image.new("L", (900, 600), 0)
trunk_image.paste(255, (400, 0, 460, 600))
trunk_image.save(trunk_png, format="PNG")
client.put(f"/photos/{mend_photo}/regions/{trunk_region['id']}/mask",
           files={"file": ("trunk.png", trunk_png.getvalue(), "image/png")})
mended = [r for r in client.get(f"/photos/{mend_photo}").json()["regions"] if r["geometry"] == "line"]
auto_eaves = sorted([r for r in mended if r["source"] == "auto"], key=lambda r: r["polyline"][0][0])
check("painting the trunk joins the two pieces either side of it into one straight line",
      len(auto_eaves) == 2 and auto_eaves[0]["polyline"] == [[100, 300], [800, 300]])
check("the joined line reports the stretch behind the trunk",
      auto_eaves[0]["hidden"] and 295 <= auto_eaves[0]["hidden"][0][0] <= 305 and 355 <= auto_eaves[0]["hidden"][0][1] <= 365)
check("a gap nothing stands in front of is left open", auto_eaves[1]["polyline"] == [[840, 300], [890, 300]])
check("a line drawn by hand is never joined or moved",
      [r["polyline"] for r in mended if r["source"] == "manual"] == [[[100, 340], [400, 340]]])

check("logout succeeds", client.post("/auth/logout").status_code == 204)
check("session is gone after logout",
      client.get("/auth/me").json() is None and client.get("/photos").status_code == 401)
check("wrong password rejected", client.post(
    "/auth/login", json={"email": email, "password": "wrong password"}
).status_code == 401)
check("unknown email rejected the same way", client.post(
    "/auth/login", json={"email": "nobody@example.com", "password": PASSWORD}
).status_code == 401)
check("login restores access",
      client.post("/auth/login", json={"email": email, "password": PASSWORD}).status_code == 200
      and client.get(f"/photos/{photo_id}").status_code == 200)

print("\nAll smoke checks passed.")
