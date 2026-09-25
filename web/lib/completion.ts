// Stretches of a line that pass behind an obstacle area the user painted (an "Obstruction", a tree): shown dimmed
// with "N% hidden". Display only: nothing here changes where the lines are.

import type { Point } from "./types";
import { BLOCKER_KINDS, d2, type Region } from "./design";

const STEP = 2;                  // sampling along lines, px
const MIN_HIDDEN_FRAC = 0.004;   // hidden runs shorter than this share of the image width are dropped

export type Cover = (x: number, y: number) => boolean;

export const isBlocker = (r: Region) => r.geometry === "area" && r.visible && !!r.mask && (r.role === "blocker" || BLOCKER_KINDS.has(r.kind));

/** A cover test dilated by one pixel. */
export function dilated(test: Cover): Cover {
  return (x, y) => {
    const xi = Math.round(x), yi = Math.round(y);
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) if (test(xi + dx, yi + dy)) return true;
    return false;
  };
}

/** Hidden stretches along a line: [start, end] distances behind obstacles, and their share of the line. */
export function hidden(pts: Point[], cover: Cover, width: number): { runs: [number, number][]; share: number } {
  const test = dilated(cover), minRun = MIN_HIDDEN_FRAC * width;
  const runs: [number, number][] = [];
  let along = 0, start = -1;
  for (let i = 0; i < pts.length - 1; i++) {
    const a = pts[i], b = pts[i + 1], len = d2(a, b), n = Math.max(1, Math.ceil(len / STEP));
    for (let k = 0; k <= n; k++) {
      if (k === 0 && i > 0) continue;
      const t = k / n, at = along + len * t;
      if (test(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)) { if (start < 0) start = at; }
      else if (start >= 0) { if (at - start >= minRun) runs.push([start, at]); start = -1; }
    }
    along += len;
  }
  if (start >= 0 && along - start >= minRun) runs.push([start, along]);
  const total = along || 1;
  return { runs, share: runs.reduce((s, [a, b]) => s + (b - a), 0) / total };
}
