export type RegionKind =
  | "roof"
  | "driveway_edge"
  | "walkway_edge"
  | "fence"
  | "column"
  | "window"
  | "door"
  | "garage_door"
  | "wreath"
  | "hedge"
  | "shrub"
  | "topiary"
  | "palm_trunk"
  | "palm_frond"
  | "tree_canopy"
  | "obstruction"
  | "custom";

export type Geometry = "line" | "area";
export type Side = "front" | "left" | "right";
export type Edge = "eave" | "ridge";
export type MaskingMode = "manual" | "automatic";
export type Tool = "brush" | "erase" | "vertex" | "box" | "line" | "click";

export interface Region {
  id: string;
  kind: RegionKind;
  label: string;
  geometry: Geometry;
  source: "auto" | "manual";
  color: string;
  score: number;
  visible: boolean;
  side: Side | null;
  edge: Edge | null;
  polyline: number[][] | null;
  mask_url: string | null;
  // Lines only: [start, end] pixel distances along the polyline that pass behind an object.
  hidden?: number[][] | null;
}

export interface Interpreted {
  kinds: RegionKind[];
  sides: Side[];
  edges: Edge[];
  matched: boolean;
}

export interface Photo {
  id: string;
  prompt: string;
  width: number;
  height: number;
  image_url: string;
  mode: MaskingMode | null;
  approved: boolean;
  interpreted: Interpreted | null;
}

export interface PhotoDetail extends Photo {
  regions: Region[];
}

export const KIND_LABELS: Record<RegionKind, string> = {
  roof: "Roofline",
  driveway_edge: "Driveway edge",
  walkway_edge: "Walkway edge",
  fence: "Fence / railing",
  column: "Column / post",
  window: "Window",
  door: "Entry door",
  garage_door: "Garage door",
  wreath: "Wreath",
  hedge: "Hedge",
  shrub: "Shrub",
  topiary: "Topiary",
  palm_trunk: "Palm trunk",
  palm_frond: "Palm fronds",
  tree_canopy: "Tree canopy",
  obstruction: "Obstruction",
  custom: "Custom region",
};

export const KIND_COLORS: Record<RegionKind, string> = {
  roof: "#38bdf8",
  driveway_edge: "#a855f7",
  walkway_edge: "#d946ef",
  fence: "#6366f1",
  column: "#f472b6",
  window: "#fde047",
  door: "#fb923c",
  garage_door: "#ea580c",
  wreath: "#ef4444",
  hedge: "#22c55e",
  shrub: "#84cc16",
  topiary: "#14b8a6",
  palm_trunk: "#f59e0b",
  palm_frond: "#a3e635",
  tree_canopy: "#059669",
  obstruction: "#94a3b8",
  custom: "#ec4899",
};

export const LINE_KINDS: RegionKind[] = ["roof", "driveway_edge", "walkway_edge", "fence"];

export const AREA_KINDS: RegionKind[] = [
  "column",
  "window",
  "door",
  "garage_door",
  "wreath",
  "hedge",
  "shrub",
  "topiary",
  "palm_trunk",
  "palm_frond",
  "tree_canopy",
  "obstruction",
  "custom",
];

// Things that can stand between the camera and the house; a line gets no lights where it
// passes behind one. Mirrors occlusion.BLOCKER_KINDS on the server.
export const BLOCKER_KINDS: RegionKind[] = [
  "obstruction",
  "palm_trunk",
  "palm_frond",
  "tree_canopy",
  "hedge",
  "shrub",
  "topiary",
];

export const KIND_GROUPS: { title: string; kinds: RegionKind[] }[] = [
  { title: "House — draw a line", kinds: LINE_KINDS },
  { title: "House — paint an area", kinds: ["column", "window", "door", "garage_door", "wreath"] },
  {
    title: "Landscape",
    kinds: ["hedge", "shrub", "topiary", "palm_trunk", "palm_frond", "tree_canopy"],
  },
  { title: "Other", kinds: ["obstruction", "custom"] },
];

export const geometryOf = (kind: RegionKind): Geometry =>
  LINE_KINDS.includes(kind) ? "line" : "area";

export interface User {
  id: string;
  email: string;
  name: string;
  created_at: string;
}
