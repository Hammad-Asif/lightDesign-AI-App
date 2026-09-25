// Geometry helpers for the Studio's read-only line view (junctions, hit tests). Lines are edited in the
// one editor at /design (DesignEditor), not here.

import type { LineDoc, Plane, Point, Role, Segment } from "./types";

/** Endpoints closer than this (image px) are one junction: dragging one drags them all. */
export const JOIN_TOL = 0.75;

export type EndRef = { seg: number; end: 0 | 1; origin: Point };
export type PolyRef = { plane: number; vertex: number; origin: Point };
export type Refs = { ends: EndRef[]; polys: PolyRef[] };

export type Selection = { kind: "segment"; index: number } | { kind: "vertex"; point: Point } | null;

export const dist = (a: Point, b: Point) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const round2 = (v: number) => Math.round(v * 100) / 100;
const rp = (p: Point): Point => [round2(p[0]), round2(p[1])];

export function cloneDoc(doc: LineDoc): LineDoc {
  return {
    ...doc,
    segments: doc.segments.map((s) => ({ ...s, points: [[...s.points[0]], [...s.points[1]]] as [Point, Point], planes: [...(s.planes || [])] })),
    planes: doc.planes.map((p) => ({ ...p, polygon: p.polygon.map((q) => [...q] as Point) })),
  };
}

/** Unique junctions: every segment endpoint, merged within JOIN_TOL. */
export function vertices(doc: LineDoc): Point[] {
  const out: Point[] = [];
  for (const s of doc.segments) for (const p of s.points) if (!out.some((q) => dist(p, q) < JOIN_TOL)) out.push(p);
  return out;
}

/** Everything that sits on a junction: segment ends and plane-outline corners. */
export function refsAt(doc: LineDoc, p: Point, tol = JOIN_TOL): Refs {
  const ends: EndRef[] = [];
  doc.segments.forEach((s, seg) => s.points.forEach((q, end) => { if (dist(p, q) < tol) ends.push({ seg, end: end as 0 | 1, origin: [...q] as Point }); }));
  const polys: PolyRef[] = [];
  doc.planes.forEach((pl, plane) => pl.polygon.forEach((q, vertex) => { if (dist(p, q) < Math.max(tol, 1.5)) polys.push({ plane, vertex, origin: [...q] as Point }); }));
  return { ends, polys };
}

export function mergeRefs(a: Refs, b: Refs): Refs {
  const ends = [...a.ends];
  for (const r of b.ends) if (!ends.some((e) => e.seg === r.seg && e.end === r.end)) ends.push(r);
  const polys = [...a.polys];
  for (const r of b.polys) if (!polys.some((e) => e.plane === r.plane && e.vertex === r.vertex)) polys.push(r);
  return { ends, polys };
}

/** Move every ref by the same offset from where it started (a junction drag or a whole-line drag). */
export function applyOffset(base: LineDoc, refs: Refs, dx: number, dy: number): LineDoc {
  const doc = cloneDoc(base);
  for (const r of refs.ends) {
    const s = doc.segments[r.seg];
    s.points[r.end] = rp([r.origin[0] + dx, r.origin[1] + dy]);
    s.edited = true;
  }
  for (const r of refs.polys) doc.planes[r.plane].polygon[r.vertex] = rp([r.origin[0] + dx, r.origin[1] + dy]);
  return doc;
}

/** Move a junction to an absolute position. */
export function moveTo(base: LineDoc, refs: Refs, from: Point, to: Point): LineDoc {
  return applyOffset(base, refs, to[0] - from[0], to[1] - from[1]);
}

export function projectOnSegment(p: Point, [a, b]: [Point, Point]): { point: Point; t: number; d: number } {
  const dx = b[0] - a[0], dy = b[1] - a[1], len2 = dx * dx + dy * dy || 1;
  const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2));
  const point: Point = [a[0] + t * dx, a[1] + t * dy];
  return { point, t, d: dist(p, point) };
}

/**
 * Snap a point to the nearest junction, else onto the nearest line (a T-junction), within `radius`
 * image px. `skip` excludes the ends being dragged, so a junction never snaps to itself.
 */
export function snap(doc: LineDoc, p: Point, radius: number, skip?: Refs): { point: Point; kind: "vertex" | "line" | null } {
  const skipped = (seg: number, end: number) => !!skip?.ends.some((r) => r.seg === seg && r.end === end);
  let best: Point | null = null, bestD = radius;
  doc.segments.forEach((s, i) => s.points.forEach((q, end) => {
    if (skipped(i, end)) return;
    const d = dist(p, q);
    if (d < bestD) { bestD = d; best = q; }
  }));
  if (best) return { point: [...(best as Point)] as Point, kind: "vertex" };
  let line: Point | null = null; bestD = radius * 0.8;
  doc.segments.forEach((s, i) => {
    if (skip?.ends.some((r) => r.seg === i)) return;
    const pr = projectOnSegment(p, s.points);
    if (pr.t > 0.02 && pr.t < 0.98 && pr.d < bestD) { bestD = pr.d; line = pr.point; }
  });
  return line ? { point: rp(line), kind: "line" } : { point: p, kind: null };
}

function pointInPolygon([x, y]: Point, poly: Point[]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/** Planes on either side of a new line, found by probing just off its midpoint. */
export function guessPlanes(planes: Plane[], a: Point, b: Point): number[] {
  const len = dist(a, b) || 1, nx = -(b[1] - a[1]) / len, ny = (b[0] - a[0]) / len;
  const mid: Point = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], step = Math.max(4, len * 0.03);
  const found = new Set<number>();
  for (const sgn of [1, -1]) {
    const probe: Point = [mid[0] + sgn * step * nx, mid[1] + sgn * step * ny];
    for (const pl of planes) if (pl.polygon.length >= 3 && pointInPolygon(probe, pl.polygon)) { found.add(pl.plane); break; }
  }
  return [...found].sort((x, y) => x - y);
}

function blankSegment(a: Point, b: Point, role: Role, planes: number[]): Segment {
  return { id: -1, role, planes, points: [rp(a), rp(b)], slope: null, slope_pixel: null, angle_deg: 0, length_px: dist(a, b),
           pitch_apparent: null, edited: true, fitted: true, inferred: false };
}

export function addSegment(base: LineDoc, a: Point, b: Point, role: Role): LineDoc {
  const doc = cloneDoc(base);
  doc.segments.push(blankSegment(a, b, role, guessPlanes(doc.planes, a, b)));
  return renumber(doc);
}

export function splitSegment(base: LineDoc, index: number, at: Point): LineDoc {
  const doc = cloneDoc(base);
  const s = doc.segments[index];
  const { point, t } = projectOnSegment(at, s.points);
  if (t < 0.03 || t > 0.97) return base;
  const second: Segment = { ...s, points: [rp(point), s.points[1]], edited: true, planes: [...s.planes] };
  s.points = [s.points[0], rp(point)];
  s.edited = true;
  doc.segments.splice(index + 1, 0, second);
  return renumber(doc);
}

export function deleteSegments(base: LineDoc, indices: number[]): LineDoc {
  const doc = cloneDoc(base);
  doc.segments = doc.segments.filter((_, i) => !indices.includes(i));
  return renumber(doc);
}

export function setRole(base: LineDoc, index: number, role: Role): LineDoc {
  const doc = cloneDoc(base);
  doc.segments[index] = { ...doc.segments[index], role, edited: true };
  return doc;
}

export function segmentsAt(doc: LineDoc, p: Point): number[] {
  return doc.segments.flatMap((s, i) => (s.points.some((q) => dist(q, p) < JOIN_TOL) ? [i] : []));
}

export function renumber(doc: LineDoc): LineDoc {
  doc.segments.forEach((s, i) => (s.id = i));
  return doc;
}

export const isEdited = (doc: LineDoc) => doc.segments.some((s) => s.edited);
