# Light Design

AI holiday light design mock-up tool.

**Milestone 1 complete; Milestone 2 in progress.** Detection runs on SAM 3 via the fal.ai
API — no local model weights, no GPU. See `docs/M1_STATUS.md` for measured results on the
client's real photos.

## Run (development)

Needs Python 3.11, Node 20 and a local PostgreSQL.

```bash
createdb lightdesign
cd api && python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/alembic upgrade head
.venv/bin/uvicorn app.main:app --reload

cd web && npm install && npm run dev
```

Open http://localhost:5173 and create an account — every photo belongs to the account that
uploaded it. API docs are at http://localhost:8000/docs. Settings come from `api/.env`;
`DATABASE_URL` defaults to `postgresql+psycopg://localhost/lightdesign`.

Schema changes go through Alembic: edit `app/models.py`, run
`.venv/bin/alembic revision --autogenerate -m "..."`, and review the generated file.

## Run (production)

```bash
cp .env.example .env      # set POSTGRES_PASSWORD and FAL_KEY; COOKIE_SECURE=true behind HTTPS
docker compose up -d --build
```

Everything is served from one origin on port 80: nginx serves the web app and forwards `/auth`,
`/photos` and `/files` to the API, which applies migrations on start. Postgres data and uploaded
photos live in the named volumes `pgdata` and `photos`.

## Detection providers

| `SEGMENTATION_PROVIDER` | What it does | Cost | Speed |
|---|---|---|---|
| `mock` (default) | Fixed placeholder shapes. For UI work, and what the smoke tests assert. | none | instant |
| `fal` | SAM 3 on fal.ai — text, box and point prompts from one endpoint. Needs `FAL_KEY`. | ~$0.005/call | one call per class |

There are no local weights to download and no GPU requirement.

```bash
cd api
FAL_KEY=... SEGMENTATION_PROVIDER=fal .venv/bin/uvicorn app.main:app --reload
```

Before relying on it, run the contract probe once. It settles mask resolution, prompt
support and the RLE encoding against the live API:

```bash
cd api && FAL_KEY=... .venv/bin/python tests/fal_probe.py
```

## Test detection on real photos

```bash
cd api
.venv/bin/python tests/detect_eval.py            # dry run: how many billed calls it needs
.venv/bin/python tests/detect_eval.py --spend    # make only the uncached calls
```

Runs the production detection path on the client's photos and prints a per-class hit table
against the old GroundingDINO + SAM 2 baseline. Every fal response is cached in
`out/fal_cache/`, so re-running after a filter change, or with `--prompt kind="text"` on an
already-tried prompt, costs nothing. Annotated overlays go to `out/detect-eval/`.

## Verify

```bash
createdb lightdesign_test                                          # once
cd api && .venv/bin/python tests/smoke.py                          # API, sign-in and ownership
cd web && node tests/ui-smoke.mjs && node tests/manual-mask.mjs    # browser
```

`smoke.py` rebuilds `lightdesign_test` from the migrations on every run, refuses any database
whose name doesn't end in `_test`, and writes photos to a temporary directory. The browser tests
need both servers running with `SEGMENTATION_PROVIDER=mock`, and each run registers a fresh
account in the development database. Screenshots go to `test-output/`, which is gitignored.

## What M1 does

1. Upload a photo with a plain-English instruction (stored, interpreted in M2).
2. Choose **Automatic** or **Manual** masking.
3. Automatic detects roofline, hedges, shrubs, topiaries, palm trunks, palm fronds and
   tree canopies as separate regions. Rooflines come back as individually selectable
   spans tagged front / left / right. Palm trunks are truncated at the bottom of the husk.
4. Correct anything: brush and erase on area masks, drag/insert/delete vertices on
   roofline spans, or drag a box to re-run segmentation on one area. Undo/redo throughout.
5. Approve to lock the mask set.

Shortcuts: `b` brush, `e` erase, `v` vertices, `[` / `]` brush size, `cmd+z` / `shift+cmd+z`.
