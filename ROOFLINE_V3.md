# Alternative roof detector (Phase A)

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

For local development put the following in `api/.env`; for Docker Compose put it
in the root `.env`. Keep the existing database, authentication and other settings.

```dotenv
ROOFLINE_DETECTOR=v3_sol
OPENROUTER_API_KEY=your-customer-owned-key
ROOFLINE_MODEL=openai/gpt-6-sol
ROOFLINE_REASONING=low
ROOFLINE_TIMEOUT=360
```

Restart the API (or `docker compose up -d --build`). `/health` reports the selected
detector. Set `ROOFLINE_DETECTOR=sam` and restart to restore the existing segmentation
and extraction path, including its existing `EDGE_EXTRACTOR` / `PARTITION_ENGINE`
configuration. The default remains `sam`. No database migration is needed.

The bundled nginx detection timeout is now 900 seconds, allowing V3 and a subsequent
non-roof segmentation pass. Custom reverse proxies must allow comparable request
durations. React components and editor interactions are unchanged.

## Behavior and limits

- The prompt in `api/app/prompts/roofline_v3.txt` is copied from the supplied ZIP.
  There are no in-context examples. Default model, reasoning, schema, token limit
  and lossless image encoding match the CLI.
- Uses the stored, EXIF-oriented photo at its editor resolution. There is no
  additional detection resize or coordinate scaling. The app's existing upload
  size limit / downscaling still applies.
- Returns eave, rake, hip, ridge, valley, junction and unknown edges as ordinary
  editable `roof` line regions. One segment becomes one two-vertex polyline.
- A roof-only request makes no SAM call, including no automatic blocker pass.
  Explicitly requested non-roof kinds still use the existing segmentation provider.
  Mixed requests are one operation: failure preserves previous regions.
- No mask fitting, snapping or occlusion bridging is applied to V3 output at
  detection time. Existing later editor operations, including painting blockers,
  retain their existing behavior and may rejoin automatically detected lines.
- Existing edge-intent filtering is respected; unrequested edge types are retained
  as hidden regions for review. Explicit `kinds: ["roof"]` requests all edge types.
- The CLI does **not** infer front/left/right facade labels. V3 regions have
  `side: null`. For side-specific text instructions, the lines are retained hidden
  and labelled "assign side and enable"; assign the side and enable them in the
  editor. This limitation is recorded in `side_review_required`. Do not interpret
  image slope as a facade label.
- Confidence is the app's existing placeholder 0.5, not a calibrated model score.
- Malformed, out-of-bounds, zero-length, refused or truncated responses fail with
  HTTP 503 before replacing regions. Errors expose no provider body or key. There
  is no automatic billed retry or silent fallback to SAM.

## Verification

```sh
cd api
python -m pip install -r requirements.txt pytest
python -m pytest tests/test_roofline_v3.py -q
```

These tests use mocked network responses and a store double. They exercise upload,
detect, retrieval and editing through FastAPI at five image sizes, the feature flag,
non-roof routing and failure handling. They do not establish model accuracy or
Postgres-backed production acceptance. Run the existing `tests/smoke.py` only against
an isolated database whose name ends in `_test`: it destroys that test schema.

Initial verification: 24 offline integration/validation tests passed. Additionally,
the supplied source and the new code produced identical serialized SAM-path outputs
for hybrid, profile and partition/port at two image sizes with the offline mock
segmentation provider (six configurations). Live SAM comparison is still pending.

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

Repeat on the same photos after switching the app to `ROOFLINE_DETECTOR=sam`, with
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
