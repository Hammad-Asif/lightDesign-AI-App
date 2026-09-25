// The lighting-design model: an instruction read into targets, and the lines and areas on the photo
// that the targets light. Roof lines come from the roof detector; everything else is a mask.

import type { Point, Role } from "./types";

export type RegionGeometry = "line" | "area";

export interface Region {
  id: string;
  kind: string;                         // "roof" for roof edges, else a place kind ("hedge", "window", ...) or custom_line / custom_area / obstruction
  label: string;
  geometry: RegionGeometry;
  role: "target" | "blocker";
  source: "auto" | "manual" | "contour" | "refined";
  color: string;
  visible: boolean;                     // lines: lit; areas: shown and lit
  edge?: Role | null;                   // roof lines
  side?: string | null;
  polyline?: Point[];
  mask?: string | null;                 // areas: RGBA PNG data URL, white where set
  score?: number | null;
  length_px?: number;
  pitch_apparent?: string | null;
  planes?: number[];
  inferred?: boolean;
}

export interface PatternStep { colour: string; count: number }

export interface Target {
  id: string; text: string; kinds: string[]; edges: Role[]; sides: string[]; manual: string[];
  product_id: string; product_name: string; pattern: PatternStep[]; spacing_in: number | null; twinkle: boolean;
  assumed: string[]; problems: string[];
}

export interface Design {
  version: 1;
  instruction: string;
  mode: "automatic" | "manual";
  targets: Target[];
  regions: Region[];
  detected: boolean;
  notes: string[];
  lighting_confirmed: boolean;
  approved: boolean;
  support_mask?: string | null;         // the detected roof (for carrying lines through obstacles)
}

export interface Product {
  id: string; name: string; category: "socket" | "strand" | "accent"; spacing_in: number | null; spacing_options: number[];
  twinkle?: boolean;
}

export interface Catalogue {
  products: Product[];
  colours: Record<string, { label: string; hex: string }>;
  kinds: Record<string, { label: string; geometry: RegionGeometry; detect: string | null }>;
  edges: Role[];
  default_product: Record<string, string>;
  edge_colours: Record<string, string>;
  sam_api: boolean;
}

export const EDGE_LABEL: Record<Role, string> = { eave: "Eave", rake: "Rake", hip: "Hip", ridge: "Ridge", valley: "Valley" };

/** Kinds the editor knows, beyond what the instruction reader names. */
export const EXTRA_KINDS: Catalogue["kinds"] = {
  custom_line: { label: "Custom line", geometry: "line", detect: null },
  custom_area: { label: "Custom area", geometry: "area", detect: null },
  obstruction: { label: "Obstruction", geometry: "area", detect: null },
};

export const BLOCKER_KINDS = new Set(["obstruction", "palm_trunk", "palm_frond", "tree_canopy", "hedge", "shrub", "topiary"]);

export const AREA_COLOURS: Record<string, string> = {
  hedge: "#22c55e", shrub: "#4ade80", topiary: "#86efac", palm_trunk: "#d97706", palm_frond: "#65a30d", tree_canopy: "#16a34a",
  dormer: "#f472b6", chimney: "#fb7185", soffit: "#fbbf24", window: "#38bdf8", door: "#a78bfa", garage_door: "#818cf8",
  column: "#e2e8f0", wreath: "#ef4444", garland: "#10b981", obstruction: "#94a3b8", custom_area: "#f59e0b",
  fence: "#eab308", driveway_edge: "#f97316", walkway_edge: "#fb923c", custom_line: "#facc15",
};

export const ADD_GROUPS: { title: string; items: { kind: string; edge?: Role; label: string }[] }[] = [
  { title: "Roof — lines", items: (["eave", "rake", "hip", "ridge", "valley"] as Role[]).map((e) => ({ kind: "roof", edge: e, label: EDGE_LABEL[e] })) },
  { title: "Roof — surfaces & accents", items: [{ kind: "dormer", label: "Dormer" }, { kind: "chimney", label: "Chimney" }, { kind: "soffit", label: "Soffit" }] },
  { title: "House — draw a line", items: [{ kind: "fence", label: "Fence" }, { kind: "driveway_edge", label: "Driveway edge" }, { kind: "walkway_edge", label: "Walkway edge" }, { kind: "custom_line", label: "Custom line" }] },
  { title: "House — paint an area", items: [{ kind: "window", label: "Window" }, { kind: "door", label: "Door" }, { kind: "garage_door", label: "Garage door" }, { kind: "column", label: "Column" }, { kind: "wreath", label: "Wreath" }, { kind: "garland", label: "Garland" }] },
  { title: "Landscape", items: [{ kind: "hedge", label: "Hedge" }, { kind: "shrub", label: "Shrub" }, { kind: "topiary", label: "Topiary" }, { kind: "palm_trunk", label: "Palm trunk" }, { kind: "palm_frond", label: "Palm fronds" }, { kind: "tree_canopy", label: "Tree canopy" }] },
  { title: "Other", items: [{ kind: "obstruction", label: "Obstruction" }, { kind: "custom_area", label: "Custom area" }] },
];

export const newId = () => Math.random().toString(36).slice(2, 10);

export function kindLabel(cat: Catalogue | null, kind: string): string {
  return (cat?.kinds[kind] ?? EXTRA_KINDS[kind])?.label ?? kind;
}

export function kindGeometry(cat: Catalogue | null, kind: string): RegionGeometry {
  return (cat?.kinds[kind] ?? EXTRA_KINDS[kind])?.geometry ?? "area";
}

export function regionColour(cat: Catalogue | null, kind: string, edge?: Role | null): string {
  if (kind === "roof" && edge) return cat?.edge_colours[edge] ?? "#39ff14";
  return AREA_COLOURS[kind] ?? "#f59e0b";
}

/** Numbered default names: "Eave 2", "Hedges". */
export function defaultLabel(cat: Catalogue | null, regions: Region[], kind: string, edge?: Role | null): string {
  const base = kind === "roof" && edge ? EDGE_LABEL[edge] : kindLabel(cat, kind);
  const n = regions.filter((r) => r.kind === kind && (r.edge ?? null) === (edge ?? null)).length;
  return n ? `${base} ${n + 1}` : base;
}

/** Which targets light a region: its kind matches. Every roof line that is on (◉) is a proper roofline, whatever
 *  its edge: a roof request lights them all. */
export function targetsFor(design: Design, r: Region): Target[] {
  if (r.role === "blocker" || !r.visible) return [];
  return design.targets.filter((t) => t.kinds.includes(r.kind));
}

export function regionsFor(design: Design, t: Target): Region[] {
  return design.regions.filter((r) => targetsFor(design, r).some((x) => x.id === t.id));
}

export function lightingStatus(design: Design): string {
  if (!design.lighting_confirmed) return "Lighting not confirmed yet";
  const check = design.targets.filter((t) => t.problems.length || !t.product_id || regionsFor(design, t).length === 0).length;
  if (check) return `Lighting: ${check} to check`;
  const lit = new Set(design.targets.flatMap((t) => regionsFor(design, t).map((r) => r.id)));
  return `Lighting confirmed for ${lit.size} line${lit.size === 1 ? "" : "s"} and area${lit.size === 1 ? "" : "s"}`;
}

export function placeOf(cat: Catalogue | null, t: Target): string {
  return t.kinds.map((k) => (k === "roof" ? `Roofline${t.edges.length ? ` · ${t.edges.map((e) => `${e}s`).join(", ")}` : ""}` : kindLabel(cat, k))).join(" + ")
    + (t.sides.length ? ` · ${t.sides.join(", ")} only` : "");
}

// ── polyline geometry ──

export const d2 = (a: Point, b: Point) => Math.hypot(a[0] - b[0], a[1] - b[1]);

export function projectOn(p: Point, a: Point, b: Point) {
  const dx = b[0] - a[0], dy = b[1] - a[1], len2 = dx * dx + dy * dy || 1;
  const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2));
  const q: Point = [a[0] + t * dx, a[1] + t * dy];
  return { point: q, t, d: d2(p, q) };
}

export function nearestOnPolyline(p: Point, pts: Point[]) {
  let best = { seg: -1, point: p as Point, d: Infinity, t: 0 };
  for (let i = 0; i < pts.length - 1; i++) {
    const r = projectOn(p, pts[i], pts[i + 1]);
    if (r.d < best.d) best = { seg: i, point: r.point, d: r.d, t: r.t };
  }
  return best;
}

export function polylineLength(pts: Point[]): number {
  let s = 0;
  for (let i = 0; i < pts.length - 1; i++) s += d2(pts[i], pts[i + 1]);
  return s;
}

/** The sub-polylines of `pts` between distances [from, to]. */
export function slice(pts: Point[], from: number, to: number): Point[] {
  const out: Point[] = [];
  let along = 0;
  for (let i = 0; i < pts.length - 1; i++) {
    const a = pts[i], b = pts[i + 1], len = d2(a, b);
    const s0 = Math.max(from, along), s1 = Math.min(to, along + len);
    if (s1 > s0 && len > 0) {
      const p0: Point = [a[0] + ((b[0] - a[0]) * (s0 - along)) / len, a[1] + ((b[1] - a[1]) * (s0 - along)) / len];
      const p1: Point = [a[0] + ((b[0] - a[0]) * (s1 - along)) / len, a[1] + ((b[1] - a[1]) * (s1 - along)) / len];
      if (!out.length) out.push(p0);
      out.push(p1);
    }
    along += len;
  }
  return out;
}
