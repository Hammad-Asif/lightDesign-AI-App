"""Benchmark the actual app API. Dry-run by default; no contractor service required.

Run each detector on a separate app instance or restart after switching the flag.
Upload, detect, read back and edit one vertex to its existing coordinates to verify
the editor API contract. This no-op check is NOT counted as a human correction.
"""
import argparse
import csv
import hashlib
import json
import mimetypes
import os
import time
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--photos", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True,
                        help='JSON {"tuning":[14 filenames],"held_out":[6 filenames]} agreed before tuning')
    parser.add_argument("--subset", choices=("tuning", "held_out"), default="tuning")
    parser.add_argument("--detector", choices=("sam", "v3_sol"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--spend", action="store_true", help="Upload photos and make billed detector calls")
    args = parser.parse_args()
    split = json.loads(args.split.read_text(encoding="utf-8"))
    tuning, held = split["tuning"], split["held_out"]
    if len(tuning) != 14 or len(held) != 6 or len(set(tuning + held)) != 20:
        parser.error("Require exactly 14 tuning and 6 distinct held-out filenames")
    root = args.photos.resolve()
    hashes = {}
    for name in tuning + held:
        path = (root / name).resolve()
        if path.parent != root or not path.is_file():
            parser.error(f"Missing file or invalid direct filename: {name}")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if {hashes[n] for n in tuning} & {hashes[n] for n in held}:
        parser.error("Identical photo content appears in tuning and held-out sets; revise the split")
    names = split[args.subset]
    print(f"{args.detector}: {len(names)} {args.subset} photos. " + ("LIVE" if args.spend else "DRY RUN; add --spend to run"))
    if not args.spend:
        return
    if not os.environ.get("BENCH_EMAIL") or not os.environ.get("BENCH_PASSWORD"):
        parser.error("Set BENCH_EMAIL and BENCH_PASSWORD for an existing app test account")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Output directory must be empty; refusing to overwrite or rebill a previous run")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {"detector": args.detector, "subset": args.subset, "split": split, "sha256": hashes}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows = []
    with httpx.Client(base_url=args.api, timeout=900) as client:
        health = client.get("/health")
        health.raise_for_status()
        config = health.json()
        if config.get("roofline_detector") != args.detector:
            parser.error("App detector differs from --detector; update config and restart the API")
        if args.detector == "sam" and config.get("segmentation_provider") != "fal":
            parser.error("SAM benchmark requires SEGMENTATION_PROVIDER=fal, not mock")
        (args.out / "health.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        login = client.post("/auth/login", json={"email": os.environ["BENCH_EMAIL"], "password": os.environ["BENCH_PASSWORD"]})
        login.raise_for_status()
        for name in names:
            row = {"photo": name, "detector": args.detector, "subset": args.subset,
                   "photo_id": "", "status": "failed", "elapsed_seconds": None, "cost_usd": None,
                   "line_count": None, "vertex_drags": "", "adds": "", "deletes": "", "relabels": "",
                   "manual_fixes": "", "failure_notes": ""}
            started = None
            try:
                path = root / name
                upload = client.post("/photos", data={"prompt": "roofline"}, files={
                    "file": (name, path.read_bytes(), mimetypes.guess_type(name)[0] or "image/jpeg")})
                upload.raise_for_status()
                photo_id = upload.json()["id"]
                row["photo_id"] = photo_id
                started = time.perf_counter()
                detected = client.post(f"/photos/{photo_id}/detect", json={"kinds": ["roof"]})
                row["elapsed_seconds"] = round(time.perf_counter() - started, 3)
                detected.raise_for_status()
                detail = client.get(f"/photos/{photo_id}")
                detail.raise_for_status()
                data = detail.json()
                lines = [r for r in data["regions"] if r["kind"] == "roof" and r["geometry"] == "line"]
                if lines:
                    first = lines[0]
                    edited = client.patch(f"/photos/{photo_id}/regions/{first['id']}", json={"polyline": first["polyline"]})
                    edited.raise_for_status()
                    if edited.json()["polyline"] != first["polyline"]:
                        raise ValueError("Editor round-trip changed coordinates")
                run = (data.get("interpreted") or {}).get("detection_run") or {}
                row.update(status="ok", cost_usd=run.get("cost_usd"), line_count=len(lines))
                (args.out / (name + ".json")).write_text(json.dumps(data, indent=2), encoding="utf-8")
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                # Preserve timing and status; do not log response bodies or credentials.
                if started is not None and row["elapsed_seconds"] is None:
                    row["elapsed_seconds"] = round(time.perf_counter() - started, 3)
                row["failure_notes"] = type(exc).__name__
            rows.append(row)
            with (args.out / "measurements.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)
            print(f"{name}: {row['status']}, {row['line_count']} lines, {row['elapsed_seconds']}s")
    print("Manual fix fields are intentionally blank: a human must count each drag/add/delete/relabel.")
    print("Missing cost means unavailable, not zero. SAM billing must be reconciled with fal usage.")


if __name__ == "__main__":
    main()
