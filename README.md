# Light Design

Two roof-detection features are merged in and **both are off by default**, so detection behaves
exactly as before until one is switched on. Turn on at most one at a time.

| Feature | Flag (`api/.env`, or root `.env` for Docker) | Notes |
|---|---|---|
| V3 roof detector | `ROOFLINE_DETECTOR=v3_sol` | Needs `OPENROUTER_API_KEY`. Setup and limits in [ROOFLINE_V3.md](ROOFLINE_V3.md). |
| Target house only | `TARGET_HOUSE_ONLY=true` | Works on the SAM/partition path. Described below. |

`/health` reports `roofline_detector` and `target_house_only`, so you can see which mode is live
without reading the log. They are independent settings, but target-house selects among SAM roof
planes — with `v3_sol` there are no such planes, so it has nothing to act on.

---

- m0-target-house.

To switch it on, set TARGET_HOUSE_ONLY=true in api/.env and restart the API. /health now shows target_house_only, so you can check which mode is running. With the flag off, detection is exactly as before.

What it does:
1. Roof lines only on the house being quoted. Neighbours' roofs get none.
2. The house's own lines decide the vanishing points. One only a neighbour supports is dropped. One a neighbour pulled more than 4° aside moves back to the house's lines. Smaller differences are left alone so the house's lines don't jitter.
3. No new SAM 3 calls. It uses the roof planes' position, size and shared outlines, plus roof colour for a neighbour set back behind a porch.
4. A House tool in the editor: tap a roof to add it to the house or take it out. The lines are redrawn from the stored planes with no detection call, and any confirmed lighting moves to the new lines.
5. Palms, hedges, trees and cars are unchanged and still act as blockers.

tests/smoke.py now covers the tap too, and passes with the flag on and off.

One heads-up before you test: SAM 3 on fal doesn't return identical masks on every call. I detected your 5 photos twice, both times with the flag off. The roof lines differed on 4 of them, and so did the palm and tree masks (one run found an extra palm on photo 4). So comparing a flag-off detection with a separate flag-on detection will show SAM's differences on top of the flag's.





Two ways to compare on the same detection:
1. tests/target_eval.py (commands in the README). It runs both settings on one cached detection, prints the checks and saves side-by-side overlays. Lines the flag removed are drawn dashed in magenta, so you can see each one sits on a neighbour's roof.
2. In the app with the flag on: pick the House tool and tap every red plane in. That gives exactly the flag-off lines for that detection.





If you'd rather flip the flag and re-detect in the app, I can add a small dev setting that reuses a photo's SAM result on re-detect, so both runs see the same masks.

Known limits, each fixed with one tap:
1. A neighbour roof of the same material that touches the porch stays in.
2. A large neighbour fully in the frame and not touching the house is kept, like a detached garage.
3. A small detached garage, or a part of the house's roof cut off by the photo's edge, is left out.
4. Attached row houses count as one building.