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
  | "palm_collar"
  | "palm_frond"
  | "tree_canopy"
  | "obstruction"
  | "custom";

export type Geometry = "line" | "area";
export type Side = "front" | "left" | "right";
export type Edge = "eave" | "rake" | "hip" | "ridge" | "valley" | "junction" | "unknown";
export type MaskingMode = "manual" | "automatic";
export type Tool = "select" | "brush" | "erase" | "vertex" | "move" | "box" | "line" | "click" | "house";

// "blocker" is something standing in front of the house — a palm, a tree, a car. It is detected
// so a line passing behind it can be joined up and reported, and it is never lit itself.
export type RegionRole = "target" | "blocker";

export interface Region {
  id: string;
  kind: RegionKind;
  role: RegionRole;
  label: string;
  geometry: Geometry;
  source: "auto" | "manual" | "inferred";
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

export type ProductCategory = "c9" | "c7" | "mini" | "icicle" | "wreath" | "garland";

export interface Product {
  id: string;
  builtin: boolean;
  // Retired: out of the library and out of the parser, still resolvable for saved designs.
  archived: boolean;
  category: ProductCategory;
  name: string;
  bulb_diameter_in: number | null;
  spacing_in: number | null;
  // Socket lines can be ordered at several spacings; empty means spacing_in is fixed.
  spacing_options_in: number[];
  strand_length_ft: number | null;
  // A wreath's diameter, or a garland's thickness.
  diameter_in: number | null;
  bulb_count: number | null;
  // Icicle drop lengths, repeating along the strand.
  drops_in: number[];
  colours: string[];
  twinkle: boolean;
  // A picture of the product, or null. Uploaded separately, not part of the JSON body.
  image_url: string | null;
}

// What a product looks like before it has an id: the fields the forms edit.
export type ProductDraft = Omit<Product, "id" | "builtin" | "archived" | "image_url">;

export interface PaletteColour {
  key: string;
  name: string;
  hex: string;
}

export interface PatternStep {
  colour: string;
  count: number;
}

export interface Pattern {
  steps: PatternStep[];
  direction?: "forward" | "reverse";
  offset?: number;
  unit?: "bulb" | "turn";
}

// What lights one region: render parameters, never pixels. Mirrors schemas.LightingSettings.
export interface LightingSettings {
  product_id: string;
  pattern?: Pattern | null;
  spacing_in?: number | null;
  twinkle?: boolean | null;
  wrap?: { start: number; stop: number; turn_spacing_in: number } | null;
}

// One clause of the instruction, resolved against the catalogue on the server.
export interface InterpretedRequest {
  text: string;
  kinds: RegionKind[];
  sides: Side[];
  edges: Edge[];
  // Kinds detection never finds; the salesperson draws them.
  manual: RegionKind[];
  lighting: {
    category: ProductCategory | null;
    colours: string[];
    counts: number[];
    spacing_in: number | null;
    size_in: number | null;
    twinkle: boolean | null;
    per_turn: boolean;
  };
  product_id: string | null;
  product_name: string | null;
  settings: LightingSettings | null;
  assumed: string[];
  problems: string[];
  choices: { id: string; name: string }[];
}

export interface DesignRequest {
  text: string;
  kinds: RegionKind[];
  sides: Side[];
  edges: Edge[];
  settings: LightingSettings;
}

export interface Design {
  id: string;
  name: string;
  requests: (DesignRequest & { product_name: string; region_ids: string[] })[];
  regions: { region_id: string; settings: LightingSettings }[];
}

export interface Interpreted {
  kinds: RegionKind[];
  sides: Side[];
  edges: Edge[];
  matched: boolean;
  requests?: InterpretedRequest[] | null;
  // Partition mode: the detected roof planes, closed polygons in photo pixels with their partition colour.
  planes?: { plane: number; polygon: number[][]; color: string }[] | null;
  // Which reader produced `requests`, and what it could not read.
  read_with?: ReadWith | null;
  // TARGET_HOUSE_ONLY: every roof plane detection returned, and whether it is on the house being quoted.
  target?: TargetHouse | null;
}

export interface ReadWith {
  reader: "rules" | "llm";
  // One line naming the reader, and the reason when it fell back to the rules.
  sentence: string;
  // What the reader could not read, in its own words.
  unclear: string[];
  // What it asked for that we do not sell, dropped rather than guessed at.
  dropped: string[];
  // Places it named that detection never finds, so the salesperson draws them.
  by_hand: string[];
}

export interface TargetHouse {
  planes: { outline: number[][][]; keep: boolean; score: number }[];
  sides: Side[];
  edges: Edge[];
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
  palm_collar: "Palm collar",
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
  palm_collar: "#e11d48",
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
  "palm_collar",
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
  "palm_collar",
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
    kinds: ["hedge", "shrub", "topiary", "palm_trunk", "palm_collar", "palm_frond", "tree_canopy"],
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

export const CATEGORY_LABELS: Record<ProductCategory, string> = {
  c9: "C9",
  c7: "C7",
  mini: "Mini lights",
  icicle: "Icicle lights",
  wreath: "Wreath",
  garland: "Garland",
};

// Whether a confirmed request lights this region. Mirrors lighting.matches on the server.
export function requestLights(
  request: { kinds: RegionKind[]; sides: Side[]; edges: Edge[] },
  region: Region,
): boolean {
  if (region.role !== "target" || !region.visible || !request.kinds.includes(region.kind)) return false;
  if (request.sides.length && region.side && !request.sides.includes(region.side)) return false;
  if (region.kind === "roof" && request.edges.length) return region.edge !== null && request.edges.includes(region.edge);
  return region.kind !== "roof" || (region.edge !== "junction" && region.edge !== "unknown");
}
