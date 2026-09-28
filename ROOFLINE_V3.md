# V3 roof-line source

## Milestone 1 setup disclosure

- Model: `openai/gpt-6-sol`, low reasoning, through the hosted OpenRouter API.
- Code runs inside this repository's FastAPI process. The model runs externally;
  photos are transmitted to OpenRouter and its selected model provider.
- Key: the deploying customer supplies and owns `OPENROUTER_API_KEY`; never commit it.
- No contractor server, IP address, account, proxy, example-image directory or local
  reference-engine package is needed for V3 roof-only detection.
- Per-photo time and cost vary with image and output. The API stores measured model
  latency, token usage and OpenRouter's returned cost in
  `photo.interpreted.detection_run`. Missing cost is unknown, never zero. Mixed
  requests also incur the existing provider's charges, which are not included there.

Initial live smoke measurement (2026-09-26, image 12, 1254 x 1254): 22.506 seconds,
$0.0175535 reported by OpenRouter, 63 valid segments. This is one direct-provider
measurement, not an app/database latency benchmark or an accuracy score. Visual
review found questionable interior lines and duplicate-looking outlines; manual
fix counts and tuning remain pending.

## Enable / roll back

V3 draws the roof lines; SAM3 still runs on every detection and supplies everything else. For local development put the following in `api/.env`; for Docker Compose put it in the
root `.env`. Keep the existing database, authentication and other settings.

```dotenv
ROOFLINE_SOURCES=v3,partition
OPENROUTER_API_KEY=your-customer-owned-key
ROOFLINE_MODEL=openai/gpt-6-sol
ROOFLINE_REASONING=low
ROOFLINE_TIMEOUT=360
```

Apply the migration (`alembic upgrade head`: two nullable columns, `regions.confidence` and
`regions.sources`), then restart the API (or `docker compose up -d --build`). `/health` reports
`roofline_sources`. Set `ROOFLINE_SOURCES=partition` (the default) and restart to use the partition
pipeline alone; it draws exactly the lines it drew before this change, now with confidence and
source metadata. An old `ROOFLINE_DETECTOR=v3_sol` with no `ROOFLINE_SOURCES` is read as
`v3,partition`.

## One detection path

```text
SAM3: roof planes + blockers + every other requested kind
   -> target house (m0, services/target.py) -> padded crop
   -> V3 on the crop (cached) -> crop -> photo pixels -> canonical RoofLine
   -> conservative snap, vanishing-point facade, neighbour-plane filter, corner closure
   -> occlusion against SAM's blockers -> confidence -> editor regions
```

- **Roof lines come from one source.** The first source in `ROOFLINE_SOURCES` that answers draws
  all of them. With `v3,partition` that is V3; partition's own lines are never mixed in, and are
  used only when V3 fails (or with `ROOFLINE_SOURCES=partition`).
- **SAM always runs**, for every detection: roof planes, blockers, the target house and every
  other requested kind. For the roof it is context for V3's lines, never extra lines.
- **Crop.** The target house's roof planes (the existing m0 selection, also when
  `TARGET_HOUSE_ONLY` is off) give a bounding box. Sides and bottom are padded by
  `V3_CROP_PADDING` (default 10% of its size). The top goes up to where the sky begins above the
  roof, plus a strip of sky (`V3_CROP_SKY`; one more SAM prompt, "sky", only when V3 runs), so an
  upper storey, chimney or gable top SAM's roof masks missed is still in view. It is never tighter
  than the padding and never rises more than `V3_CROP_SKY_MAX_RISE` (0.5) of the box's width above
  the roof; with no sky above the roof the padding decides. `crop_basis` says which
  (`target_house_sky` / `target_house`). Only that crop is sent. With no SAM roof plane at all, V3
  gets the photo.
- **Second calls for missed planes** (`V3_MISSED_PLANES=true`, off by default). After the first
  answer, each target-house roof plane is checked: if under `V3_MISSED_COVERAGE` (0.15) of its
  visible outline lies near any V3 line, V3 missed it. Largest first, up to `V3_MISSED_MAX_CALLS`
  (3), V3 is asked again on a closer crop around that plane (`V3_MISSED_PADDING` 0.25, sky-bounded
  top, at least 192 px); planes under `V3_MISSED_MIN_AREA` (3%) of the house's roof area, and
  crops mostly inside an earlier second call's, are skipped. The answers' lines (ids `v3m1_001`...)
  go through the same pipeline; any that repeat a first-call line are dropped. Every call is
  cached and never fails detection; `detection_run.missed_planes` lists the planes and calls and
  `detection_run.total_cost_usd` the bill for all of them. House taps reuse the stored answers.
- **Coordinates.** V3 answers in crop pixels; `services/rooflines.crop_to_image` maps every line
  to photo pixels before anything else sees it. The raw answer is kept on each line.
- **Conservative geometry.** A V3 endpoint moves only onto a strong, consistent, parallel image
  edge within a small radius (`V3_SNAP_RADIUS`), never onto a blocker, never away from the roof
  masks, and never so the line turns more than 2.5 degrees (`V3_SNAP=false` turns it off).
  Vanishing points (the target house's own, as m0 refits them) give each level line its facade:
  metadata only, the line is not moved or rotated, and image slope is never used. A V3 line is
  rejected only when it lies mostly on a plane the target selection excluded, within a tolerance.
  Corners between two V3 lines are closed only when the lines really cross there, no object sits
  in the corner, and neither line shrinks by half.
- **Occlusion** joins same-type pieces across a covered gap and carries an end behind an object
  only where the roof mask continues, with SAM's blockers.
- **Confidence** in [0, 1] from measured evidence: source, edge strength along the line,
  vanishing-point fit, connectivity and length, plane-outline agreement, visibility; weights in
  `ROOFLINE_CONFIDENCE_WEIGHTS`. These are heuristic weights to validate on the annotated photos,
  not a calibrated probability. It replaces the fixed 0.5 in `score` and is also in `confidence`.
  Below `ROOFLINE_CONFIDENCE_LOW` (0.45) the editor draws the line dashed.
- **Sides.** A V3 line takes its facade from a vanishing point, or from the level lines it meets.
  Without one it has `side: null`; on a side-specific instruction it is kept hidden and labelled
  "assign side and enable", as before.

## Failure, cost and caching

- V3 failing (no key, timeout, HTTP error, refused/truncated/invalid JSON, no lines) never fails
  detection: the partition lines are used exactly as the partition-only pipeline draws them, and
  `interpreted.roofline` records `v3_status`, `v3_error` and `fallback_used`. A SAM failure still
  fails detection (HTTP 503) and keeps the previous regions.
- V3 answers are cached under `STORAGE_DIR/v3_cache`, keyed by photo pixels, crop box, crop
  pixels, model, reasoning, prompt and schema (never by filename). Re-detecting the same photo is
  a cache hit and makes no call; `detection_run.cache` says which. Failures are not cached.
- `detection_run` keeps the billing record (latency, tokens, OpenRouter's cost); on a cache hit
  `cost_usd` is 0 and the original bill is in `cached_cost_usd`.
- Editor actions never call V3: dragging, drawing, painting, relabelling, re-rendering, and the
  House tool's taps (which reuse V3's stored answer for the original crop). Only Detect,
  Re-detect and Find in Box can, and those go through the cache. Lines someone drew are never
  moved, merged, relabelled or deleted by any of it.

## Find in Box

Where detection missed a roof edge, the editor's "Find in box" tool takes a box drawn around it.
The server pads it slightly and runs V3 on it (cached with the box in the key) through the same
pipeline; only if V3 fails (and partition is a source) does SAM find roof planes there instead. Lines whose middle lies in the
box and that do not repeat a line already on the photo are added as new detected lines; nothing
existing changes. `interpreted.roofline_find` records the last run.

## Diagnosing a bad line

Every run stores counts per stage in `interpreted.roofline.counts` (SAM masks and blockers, crop,
V3 raw / valid / converted / snapped / plane-filtered, occlusion, final by confidence level,
and which source drew the lines) and logs them. With `ROOFLINE_DEBUG=true`, `interpreted.roofline_debug` also
holds the lines at every stage (A raw V3, B converted, C snapped, D vanishing points, E plane
filter, F corners, H occlusion, I final; P partition when partition drew them) and each V3
line's lifecycle (`v3_001`: snapped by how much, facade, plane filter, occlusion, final confidence), and PNG overlays are written to the photo's `debug/` folder. The editor then shows a
Stage picker over the photo. The raw V3 stage is placed by the crop's corner in the browser,
independently of the server's conversion. If the raw V3 line is right and the final one is wrong,
the first stage where it changed is where to look.

## Verification

```sh
cd api
python -m pip install -r requirements.txt pytest
python -m pytest tests/test_roofline_v3.py tests/test_roofline_fusion.py -q
```

These tests use mocked network responses and a store double. They do not establish model accuracy
or Postgres-backed production acceptance. Run the existing `tests/smoke.py` only against an
isolated database whose name ends in `_test`: it destroys that test schema. Tune the match and
confidence settings against `tests/roofline_baseline.py` on the annotated photos.

## Milestone 2 benchmark protocol

Agree on 14 tuning photos and 6 held-out photos before tuning. Do not use previously
reviewed photos (including image 12) as held-out acceptance cases. Byte-identical
photos must stay in the same set. Freeze model, prompt hash, config and code commit
before the held-out run. Do not tune against held-out failures.

In the supplied 20-photo folder, `13.jpeg` and `16.jpeg` have identical SHA-256
hashes. Keep both in the same set or obtain a replacement photograph before
acceptance; the benchmark runner rejects duplicates across the two sets.

Create a local JSON split with `tuning` and `held_out` arrays of filenames. Start the
app with real Postgres and a test account, configure the requested detector, and set
`BENCH_EMAIL` and `BENCH_PASSWORD` in your shell. Run from `api`:

```sh
python tests/benchmark_roofline_api.py --photos /path/to/20-photos --split /path/to/split.json --detector v3_sol --subset tuning --out ../out/v3-before
# Review the dry run, then append --spend to upload and call the hosted model.
```

Repeat on the same photos after switching the app to `ROOFLINE_SOURCES=partition`, with
`SEGMENTATION_PROVIDER=fal` and the customer's `FAL_KEY`. Use a separate output
folder for every run. The original reference extractor also needs the customer's
existing reference package; do not silently switch algorithms for the baseline.

The runner checks the live config, performs upload -> detect -> retrieve -> editor
API round-trip, and saves photo IDs, geometry, time and available model cost. The
dry run makes no calls. `--spend` uploads images and incurs charges. Failures are
recorded without automatic retry; outputs cannot be overwritten accidentally.

In `measurements.csv`, a human must enter vertex drags, adds, deletes, relabels,
their sum in `manual_fixes`, and failure notes. A no-op API edit is not a manual fix.
Do not equate line count with accuracy or automatically infer the number of fixes.
Reconcile missing SAM cost using fal's billing records. Compare baseline and tuned
outputs on the tuning set, then run the frozen candidate once on `held_out`.
Acceptance requires at least 5 of those 6 photos to need no more than 5 fixes each.

The split, full comparison, tuning round and human acceptance counts are pending;
the integration alone does not claim that either milestone's acceptance is met.
