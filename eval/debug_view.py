"""Render prediction vs ground truth for chosen photos: python -m eval.debug_view OUT.png stem [stem ...] [--stop N]"""
import sys
import cv2
import numpy as np
from eval import harness, metrics
from roofline import pipeline
from roofline.config import Params


def render(stem: str, stop_after: int, params: Params) -> np.ndarray:
    item = {i.stem: i for i in harness.discover()}[stem]
    image = item.image()
    result = pipeline.run(image, item.instances(), params, ignore_mask=item.ignore(), stop_after=stop_after)
    vis = (image * 0.6).astype(np.uint8)
    for x1, y1, x2, y2 in metrics.annotation_segments(item.annotation):
        cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 5, cv2.LINE_AA)
    for s in result["segments"]:
        (x1, y1), (x2, y2) = s["points"]
        color = (255, 0, 255) if s.get("inferred") else (40, 255, 60)
        if not s.get("fitted", True):
            color = (0, 140, 255)
        cv2.line(vis, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), color, 2, cv2.LINE_AA)
        for x, y in s["points"]:
            cv2.circle(vis, (int(round(x)), int(round(y))), 5, (0, 0, 255), -1)
    gt = metrics.annotation_segments(item.annotation)
    xs, ys = np.concatenate([gt[:, 0], gt[:, 2]]), np.concatenate([gt[:, 1], gt[:, 3]])
    x0, x1, y0, y1 = int(max(xs.min() - 80, 0)), int(xs.max() + 80), int(max(ys.min() - 80, 0)), int(ys.max() + 80)
    crop = vis[y0:y1, x0:x1]
    cv2.putText(crop, f"{stem} (stage {result['stage']})", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return crop


if __name__ == "__main__":
    args = sys.argv[1:]
    stop = 7
    if "--stop" in args:
        i = args.index("--stop"); stop = int(args[i + 1]); del args[i:i + 2]
    out, stems = args[0], args[1:]
    tiles = [render(s, stop, Params.load()) for s in stems]
    width = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, 6, 0, width - t.shape[1], cv2.BORDER_CONSTANT, value=(30, 30, 30)) for t in tiles]
    cv2.imwrite(out, np.vstack(tiles))
    print("wrote", out)
