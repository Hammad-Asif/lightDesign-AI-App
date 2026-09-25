# Roofline

Roof structure lines (ridges, hips, valleys, rakes, eaves) from one street-level photo of a
house. Output is a set of straight vector segments drawn over the photo and exported as JSON,
with pixel endpoints, role, slope, angle, roofing pitch (rise:12) and length per segment, and a
closed polygon per roof plane.

**Principle.** Masks are plane *support regions*, not the source of lines. Lines come from
(a) plane-to-plane adjacency after a hard partition, which gives interior ridges, hips and
valleys, and (b) robust line fits over trusted boundary arcs, which give exterior eaves and rakes.
Raw mask boundary and raw LSD segments are never output.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8765        # backend (FastAPI)

cd web && npm install && npm run dev        # frontend (Next.js), then open http://localhost:3000
```

Create an account on the sign-in page, then start a project or open `/studio?sample=1`. The frontend
proxies `/api/*` to the backend (`BACKEND_URL`, default `http://127.0.0.1:8765`), so both run on one
origin and the session cookie just works.

**Accounts and projects** (`accounts.py`). SQLite in `data/` (git-ignored): users with scrypt password
hashes, and saved projects (the photo, the detected lines, the hand edits, the settings). The browser
holds one HttpOnly cookie signed with `AUTH_SECRET` (from `.env`, or generated once into
`data/secret.key`). Every endpoint except `/api/auth/*` needs a signed-in user, and a user only sees
their own sessions and projects. `ALLOW_SIGNUP=false` closes sign-up; `COOKIE_SECURE=true` behind HTTPS.

**Lighting mock-ups** (`/new` → `/design`). A photo plus "Where do the lights go?" is read by
`instruction.py` (fixed rules, no model call) into targets: places, roof edges, product, colours,
spacing, twinkle, and what was assumed. *Automatic* then runs `POST /api/design/detect`: anything about
the roof goes through the unchanged roof detector (the same `/api/segment` + `/api/geometry` calls with
default settings), its segments are grouped per edge role into lines, and every detected line (eaves, rakes,
hips, ridges, valleys) is lit and drawn solid; a line can be switched off (○) by hand. Other places (hedges, trees, palms,
dormers, …) are SAM 3 concept masks, run in parallel with the roof. Lines passing behind an obstacle area
you paint show which stretch is hidden. *Manual* detects nothing. The editor has one toolbar row:
Select (drag a point, drag a line, double-click to add a point, Alt-click to remove one), Brush / Erase for
areas, Find in box / Find by click (SAM 3 prompts, `POST /api/design/refine`; with no area selected, Find in box
looks for roof the detection missed, `POST /api/design/roof-in-box`: a "roof" box prompt, only roof not yet
detected, run through the pipeline with the planes already found, and only the roofline inside the box is
kept, the upper outline and the eaves of full roof faces, not the inner edge of a trim band), zoom, undo/redo, Re-detect and
Approve (locks the geometry), with a hint line that explains the current tool. The right panel lists lines
and areas and holds per-request lighting settings.
Everything autosaves into the project. Like the Light Design product, lighting is stored as settings;
nothing is rendered on the photo yet.

**One line editor.** Lines are edited in one place only: the editor at `/design`. In the detection studio,
"Edit lines" opens the current detected lines there (`POST /api/design/from-studio`: the segments grouped per
edge, lit as for "roofline"); the studio itself only detects, inspects and tunes. Projects saved with the
studio's former editor still show their edited lines, read-only.

The sample house runs the whole pipeline with built-in SAM-like masks and no API call. For real
photos put a fal.ai key in `.env` (`FAL_KEY=...`, see `.env.example`) and restart. Each SAM 3 call
costs about half a cent and is cached on disk per photo and prompt (`.sam_cache/`).

## Pipeline

| Phase | Module | What it does |
| --- | --- | --- |
| 0 | `eval/` | Harness, two metrics, held-out split, watermark warning. Built first; nothing is tuned without it. |
| 1 | `roofline/partition.py` | SAM 3 text prompt `roof` returns one instance per plane, overlapping. They become a hard label map: the smallest instance wins a contested pixel (children beat parents), score is folded into the priority with a soft floor at 0.35, and an instance is a duplicate only if nothing solid is left of it. Narrow gaps between planes are split between them. Borders between labels are the candidate interior lines. |
| 2 | `roofline/cleanup.py` | Per plane: close, open, fill holes, drop specks, fill narrow notches from convexity defects (never a convex hull). Growth only takes background, so the partition stays hard. Foliage (excess green, or a SAM `tree` pass) marks boundary as *suspect*; a suspect stretch loses trust only if it does not split into long straight pieces, because a roof edge is straight and a tree outline is not. Output: ordered boundary with each pixel tagged free or adjacent-to-plane-j, trusted or not. |
| 3 | `roofline/fitting.py` | `approxPolyDP` only seeds corners. Each edge is a Huber total-least-squares fit over its trusted pixels. Neighbours within the merge angle become one line, even across an occlusion gap. Every corner is the intersection of two fitted lines; a corner hidden by a tree is recovered by extending the neighbouring lines through the gap. |
| 4 | `roofline/vanishing.py`, `directions.py` | Vanishing points from every LSD segment in the photo (walls, windows, doors), as one consistent Manhattan frame: the vertical VP fixes the horizon's orientation, a focal length fixes its offset, one horizontal VP fixes the other by orthogonality; the winners are then refit freely. Edges close to a VP direction get a quadratic direction prior. Sharing 2-D directions is the fallback when no VP is trustworthy. |
| 5 | `roofline/graph.py` | Edges cluster into infinite lines (angle + offset), corners into junctions (snap radius), T-junctions are detected. Energy = pixel fit + direction prior + junction closure, minimised by block coordinate descent with closed-form steps (a 2x2 eigenproblem per line, a 2x2 solve per junction), so it never increases; it stops when the relative drop is under `energy_tol`. Junctions the relaxation has carried to within `junction_coalesce_frac` of each other are then merged (`coalesce`), so a plane thinner than the offset tolerance, whose two sides became one line, is emitted once rather than twice. Coverage (polygons vs partition) is reported before and after. |
| 6 | `roofline/interior.py` | Only where one mask spans two faces: a chord between two corners of a plane is kept if LSD segments cover most of it, it is long, and it is not parallel to the eave (tile rows). LSD only confirms a hypothesis; it is never output. |
| 7 | `roofline/roles.py` | Roles from topology: shared + horizontal in plan = ridge; shared + sloped = hip, or valley if its low end sits in a re-entrant corner; free + horizontal = eave or silhouette ridge by which side the plane lies on; free + sloped = rake, or hip when the plane narrows upward. Pitch is reported as seen, calibrated (needs a focal length: EXIF, typed in, or estimated from the VPs and labelled low-confidence), and as the nearest standard pitch by a MAP estimate. There is no height-cutoff slider. |
| 8 | `eval/tune.py` | Random search plus coordinate refinement over the thresholds on the train split, reported against the held-out split. It writes `params.json` only if held-out does not get worse. |

`POST /api/segment` runs phases 1, 2 and the VP estimation once per prompt. `POST /api/geometry`
runs phases 3 to 7 in a few milliseconds on every slider tick; `stop_after` returns any
intermediate stage. `GET /api/layer/<session>/<instances|partition|arcs|lines>.png` are the
inspection images shown in view A.

## Obstacles in front of the roof

Work on carrying roof lines behind trees, palms and cars (detector rules, a SAM obstacle pass, and joining /
extending lines through obstacles in the editor) was removed on 2026-09-26 at the owner's request: SAM 3
detection in the editor was less accurate with it. The detector is back to its earlier behaviour (the harness
reproduces `eval/results/baseline_before_obstacles.json` exactly). The sample houses with obstacles
(`synth.OBSTACLE_SPECS`) stay in the evaluation set as test data. In the editor, an "Obstruction" area painted
by hand still shows which stretch of a line passes behind it ("N % hidden"); it does not move any line.

## Evaluation

```bash
python -m eval.build_dataset                       # synthetic scenes with exact ground truth
python -m eval.harness --pipeline legacy           # the pre-rebuild pipeline, the baseline
python -m eval.harness                             # the phased pipeline (add --stop-after 3..7, --split heldout)
python -m eval.debug_view out.png <stem> ...       # prediction vs ground truth, per photo
python -m eval.ab_prompts owner_house_annotated    # SAM 3 prompt A/B, scored with the same metrics
python -m eval.tune --write                        # outer tuning loop
```

Add a photo by putting `eval/photos/<stem>.jpg`, an annotation `eval/annotations/<stem>.json`
(closed polygons, one per plane; format in `eval/build_dataset.py`), and cached instances
`eval/cache/<stem>.npz`. List watermarked photos in `eval/watermarked.txt`: bold tiled watermarks
are detected automatically, faint ones are not.

Metrics: **chamfer distance** both ways between predicted and ground-truth edge pixels, as a
percentage of the image diagonal; and **matching F1** at endpoint tolerances of 0.5, 1 and 2 % of
the diagonal, with mean angular error and the count of unmatched predictions.

## Files

- `app.py`, `index.html`: API and the single-page frontend (Tailwind via CDN, vanilla JS).
- `roofline/`: the pipeline, the SAM 3 client, local fallbacks, and the synthetic scene generator.
- `eval/`: harness, metrics, tuning loop, prompt A/B, dataset builder, and `legacy_app.py` (the frozen baseline).
