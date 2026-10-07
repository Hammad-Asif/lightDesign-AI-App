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
// "find": Find in Box — drag a box where detection missed roof lines; the server detects inside it.
export type Tool = "select" | "brush" | "erase" | "vertex" | "move" | "line" | "click" | "house" | "find" | "measure" | "wreath";

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
  // Detected roof lines: evidence-based confidence in [0, 1] and every source that found the line
  // ("v3", "partition"). Null for areas and for lines someone drew.
  confidence?: number | null;
  sources?: string[] | null;
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
  // Mini lights on an area: bulbs per 10 sq ft at the "medium" preset.
  density_per_10sqft?: number | null;
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
  // The unified roof-line run: sources, V3 status and fallback, and the confidence levels to draw with.
  roofline?: RooflineReport | null;
  // ROOFLINE_DEBUG only: the lines at every stage of the pipeline, for development.
  roofline_debug?: RooflineDebug | null;
  // The pinhole camera the photo's vanishing points give; what scale and placement use.
  camera?: CameraInfo | null;
}

export interface CameraInfo {
  focal_px: number;
  focal_35mm: number;
  // vp: read from the photo's perspective; exif: the photo's own data; assumed: a typical phone lens.
  focal_source: "vp" | "exif" | "assumed";
  facades: Side[];
}

// One roof line in real units, from GET /photos/{id}/geometry. `method` says how it was placed and
// `assumption` what that needed; `needs` says why a line could not be placed.
export interface LineGeometry {
  length_in: number | null;
  plane: string | null;
  // "estimate": no rule placed it (or its own length was unsure), so it is read at the scale of the
  // placed lines around it; always "assumed", and set by hand for a firm figure.
  method: "measured" | "wall" | "gable" | "ridge" | "hip" | "pitch" | "valley" | "override" | "estimate" | "none";
  // A typed length the photo disagrees with, in words.
  warning?: string | null;
  assumption: string | null;
  confidence: "exact" | "stated" | "assumed" | null;
  needs: "measurement" | "measurement_wall" | "side" | "override" | "length" | "anchor" | "edge_on" | "unreliable" | null;
  // The wall it was placed on ("front", "front.2" for another run of that side, "front+1" for a
  // wall set back on a roof), and the side a level line was given and how ("direction", "chain",
  // "above", "borrowed", "chosen" by consistency, ...).
  wall?: string | null;
  side?: Side | null;
  side_how?: string | null;
  // Relative: how far the length moves under the photo's own small errors (0.06 = ±6%).
  spread?: number | null;
  // What the length rests on (the wall's distance and what gave it, a roof face, anchors).
  evidence?: Record<string, unknown>[] | null;
}

// One independent check that failed (services/checks.py), with the lines it is about.
export interface ScaleCheck {
  kind:
    | "size"
    | "meets"
    | "pitch"
    | "eave_height"
    | "plausibility"
    | "side"
    | "second_reference"
    | "scale"
    | "reference"
    | "symmetry";
  message: string;
  lines: string[];
  labels?: string[];
}

// SCALE_DEBUG only: the solve step by step (services/scale_debug.py) and one overlay per step.
export interface ScaleDebug {
  photo_id: string;
  written_at: string;
  scale: unknown;
  // Optional: an API still running older code sends a payload without them.
  steps?: Record<string, Record<string, unknown>>;
  numbers?: Record<string, string>;
  lines?: ({ id: string; n: string | null; label: string | null } & Record<string, unknown>)[];
  files?: Record<string, string>;
}

export interface PhotoGeometry {
  // The line the scale was measured on has been edited since: the scale moved with it.
  reference_edited?: boolean;
  walls: Record<
    string,
    { distance_in: number; source: "measured" | "corner" | "roof face" | "solved" | "same plane"; facade?: Side }
  >;
  lines: Record<string, LineGeometry>;
  needs: string | null;
  // The gravest failed check, in words (the measurement makes the house implausible, a door the wrong size).
  warning?: string | null;
  checks?: ScaleCheck[];
  pitch?: { rise: number; source: "measured" | "estimated" | "assumed" } | null;
  // Where the measurement went: the wall, its side, and how the side was decided.
  reference?: { wall: string; facade: Side; how: string } | null;
  // A second known length against the photo's length for its line; `off` is relative (0.03 = 3% long).
  // `used`: it measured a wall the first length does not reach, so it is part of the scale, not a check.
  reference_check?: {
    region_id: string;
    typed_in: number;
    solved_in: number | null;
    off: number | null;
    used?: "measurement";
    wall?: string;
  } | null;
  // How far the measurement itself can be trusted: its length in the photo, how squarely its wall
  // is seen, whether an object hides an end.
  reference_quality?: {
    grade: "good" | "fair" | "poor";
    pixels: number;
    tap_error: number | null;
    view_deg: number;
    blocked: boolean;
    reasons: string[];
  } | null;
  // The line whose length would help the solve most: "instead" of a weak measurement, or as a
  // "second" length that measures a wall the first does not reach.
  next_measurement?: {
    line_id: string;
    label?: string | null;
    kind: "instead" | "second";
    wall: string;
    would_place: number | null;
    why: string;
  } | null;
  // The doors' heights against the typed length: a warning only, the typed length stands.
  scale_check?: {
    typed_in: number;
    factor: number;
    suggested_in: number;
    conflict: boolean;
    message: string | null;
    sources: { id: string; kind: string; label: string | null; height_in: number; factor: number }[];
  } | null;
  debug?: ScaleDebug;
  camera: { focal_source: "vp" | "exif" | "assumed"; focal_35mm: number; facades: Side[] };
}

// The bulbs of one lit region (GET /photos/{id}/placement): [x, y, drawn diameter px, colour, hidden].
// Hidden bulbs are counted and never drawn. `needs` says why a region has none yet.
export interface RegionPlacement {
  bulbs?: [number, number, number, string, boolean][];
  count?: number;
  hidden?: number;
  path?: number;
  path_length_in?: number;
  drops?: number[][];
  band?: number[][];
  needs?: string;
  // Areas: an estimate the salesperson can replace (canopies, fronds, hedges, trunk strands).
  estimated?: boolean;
  estimate?: number;
  area_sqft?: number;
  density?: "light" | "medium" | "heavy";
  strand_ft?: number;
  strand_estimate_ft?: number;
  strand_set?: boolean;
  trunk?: { diameter_in: number; height_ft: number; turns: number };
  // Lit fronds strung crown to tip: how many fronds, and each one's traced path in photo pixels.
  fronds?: { count: number; paths: number[][][] };
  scale_from?: string;
  // Wreaths: [x, y, diameter px] of the ring, and its size.
  ring?: number[];
  size_in?: number;
}

export interface Placement {
  regions: Record<string, RegionPlacement>;
  needs: string | null;
}

// A line the geometry cannot place: its real middle dragged onto it, and its typed length.
export interface ScaleOverride {
  mid?: number[];
  length_in?: number;
  count?: number;
  density?: "light" | "medium" | "heavy";
  strand_ft?: number;
  at?: number[];
  size_in?: 24 | 36 | 48;
}

// The salesperson's one known length: two points on a feature, or a line (an eave).
export interface Measurement {
  points?: number[][];
  region_id?: string;
  length_in: number;
  facade?: Side | null;
  // Stored by the server: whether the side was chosen by hand (else it is read from the photo).
  facade_given?: boolean;
  line_points?: number[][];
}

// A second known length on a line: it checks the measurement and changes nothing.
export interface ScaleCheckInput {
  region_id: string;
  length_in: number;
}

export interface RooflineReport {
  sources: string[];
  v3_status?: "ok" | "empty" | "failed" | "skipped";
  v3_error?: string | null;
  fallback_used?: boolean;
  crop?: number[] | null;
  counts?: Record<string, Record<string, unknown>>;
  confidence_levels?: { high: number; low: number };
}

export interface DebugLine {
  id: string | null;
  points?: number[][];
  // A_raw_v3 only: the model's own answer, in the pixels of the crop it was asked about.
  crop_points?: number[];
  crop?: number[];
  type?: string | null;
  source?: string[];
  confidence?: number | null;
}

export interface RooflineDebug {
  stages: Record<string, DebugLine[]>;
  lines: Record<string, unknown>[];
  overlays?: string[];
}

// Below this a detected roof line is drawn dashed. Mirrors ROOFLINE_CONFIDENCE_LOW's default; the
// photo's own roofline report carries the live value.
export const LOW_CONFIDENCE = 0.45;

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
  // The project's name (customer or address); empty for photos uploaded before names existed.
  name?: string;
  width: number;
  height: number;
  image_url: string;
  mode: MaskingMode | null;
  approved: boolean;
  interpreted: Interpreted | null;
  focal_35mm?: number | null;
  scale?: { measurement?: Measurement; overrides?: Record<string, ScaleOverride>; check?: ScaleCheckInput } | null;
}

// One card of the projects list (GET /photos/projects): no interpretation or regions.
export type ProjectStatus = "not_started" | "in_progress" | "locked";

export interface ProjectCard {
  id: string;
  prompt: string;
  name?: string;
  mode: MaskingMode | null;
  approved: boolean;
  created_at?: string | null;
  updated_at?: string | null;
  thumbnail_url: string;
}

export interface ProjectPage {
  items: ProjectCard[];
  next_offset: number | null;
  total: number;
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
