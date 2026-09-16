import io
import os
import sys
import tempfile
import uuid
from pathlib import Path

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
for required in ["roof", "hedge", "shrub", "topiary", "palm_trunk", "palm_frond", "tree_canopy"]:
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
check("intent excludes fronds used as a boundary", "palm_frond" not in real.kinds)
check("intent picks front side only", real.sides == ["front"])
check("intent picks the eave edge", real.edges == ["eave"])

check("intent ignores unrelated prompt", intent.parse("make it look nice").matched is False)
check("hedge prompt selects hedge only", intent.parse("mini lights in the hedges").kinds == ["hedge"])
check("gable maps to rake, not ridge", intent.parse("put C9 on the gable").edges == ["rake"])
check("rake is a roof term",
      intent.parse("C9 on the rakes").kinds == ["roof"] and intent.parse("C9 on the rakes").edges == ["rake"])
check("ridge still maps to ridge", intent.parse("light the ridge").edges == ["ridge"])

intent_photo = client.post(
    "/photos",
    files={"file": ("h.png", sample_photo(), "image/png")},
    data={"prompt": "wrap the palm trunks, C9 on the front roofline"},
).json()["id"]
scoped = client.post(f"/photos/{intent_photo}/detect", json={"kinds": None}).json()
scoped_kinds = {r["kind"] for r in scoped}
check("detect honours the prompt", scoped_kinds == {"roof", "palm_trunk"})
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
    return [r for r in client.post(f"/photos/{photo}/detect", json={"kinds": None}).json()
            if r["geometry"] == "line"]


gable_lines = detected_lines("put C9 on the gable")
check("a gable instruction keeps the rake spans", any(r["edge"] == "rake" for r in gable_lines))
check("a gable instruction keeps only rakes or edges it could not identify",
      bool(gable_lines) and all(r["edge"] in {"rake", "unknown"} for r in gable_lines))
check("a ridge instruction no longer sweeps up rakes",
      not any(r["edge"] == "rake" for r in detected_lines("C9 along the ridge")))

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
from app.services import detect  # noqa: E402

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
