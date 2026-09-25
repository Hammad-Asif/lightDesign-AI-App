// Shapes returned by the FastAPI backend (../app.py). Only the fields the UI reads are typed.

export type Point = [number, number];
export type Role = "ridge" | "hip" | "valley" | "eave" | "rake";
export const ROLES: Role[] = ["ridge", "hip", "valley", "eave", "rake"];

export interface User { id: string; email: string; name: string }

export interface SessionInfo {
  session_id: string; filename: string; width: number; height: number;
  source_width: number; source_height: number; scale_from_source: number;
  image_url: string; exif_focal_35mm: number | null;
  watermark: { flagged: boolean; reason: string | null };
  is_sample: boolean;
  sample_name?: string | null;
}

export interface Config {
  sam_api: boolean; text_prompt?: string; sam_api_host?: string; sam_model?: string;
  role_colors: Record<string, string>;
  defaults: { epsilon?: number; merge_angle_deg?: number; snap_radius_frac?: number; w_dir?: number };
  tuned?: boolean;
}

export interface Instance {
  id: number; source: string | null; score: number | null; area_percent: number;
  plane: number | null; kept_fraction: number | null; status: string; color: string | null;
}

export interface Segmentation {
  params: Record<string, unknown>;
  version: number; instances: Instance[]; planes: number;
  coverage_percent: number; untrusted_percent: number; adjacency_percent: number;
  vanishing_points: unknown; lsd_segments: number;
  timings: Record<string, number>; elapsed_ms: number;
  layers?: Record<string, string>;
}

export interface Segment {
  id: number; role: Role; planes: number[]; points: [Point, Point];
  slope: number | null; slope_pixel: number | null; angle_deg: number; length_px: number;
  pitch_apparent: string | null; color?: string; vp?: string | null;
  inferred?: boolean; fitted?: boolean; crease?: boolean;
  edited?: boolean;          // moved, drawn or relabelled by hand
  [key: string]: unknown;
}

export interface StandardPitch { pitch: string | null; probability: number }
export interface CalibratedPitch { pitch_deg: number; pitch: string; standard: StandardPitch; edges_used?: number }

export interface Plane {
  plane: number; part?: number; polygon: Point[];
  pitch_apparent?: string; pitch_apparent_deg?: number; pitch_apparent_standard?: StandardPitch;
  pitch_calibrated?: CalibratedPitch;
}

export interface PitchInfo {
  focal_px: number | null; focal_source: string | null; calibrated_planes: number;
  roof: CalibratedPitch | null | undefined; note?: string;
}

export interface Geometry {
  segments: Segment[]; planes: Plane[]; vertices: Point[]; stage: number;
  info: { pitch?: PitchInfo; graph?: { energy_start: number; energy_end: number; rounds: number; coverage: { after: { total: number } } }; creases?: unknown[] };
  timings: Record<string, number>;
  parameters: Record<string, unknown>;
}

/** What the editor works on: lines and plane outlines, plus the pitch summary measured from them. */
export interface LineDoc { segments: Segment[]; planes: Plane[]; pitch?: PitchInfo }

export interface GeometryParams { epsilon: number; mergeAngle: number; snap: number; wdir: number; stage: number; focal: number | null }

export interface ProjectData {
  detected: Geometry | null;
  edited: LineDoc | null;
  segmentation: Omit<Segmentation, "layers"> | null;
  geometry_params: GeometryParams;
  exif_focal_35mm?: number | null;
  is_sample?: boolean;
}

export interface ProjectSummary {
  id: string; name: string; filename: string; width: number; height: number;
  created_at: number; updated_at: number; segments: number; edited: boolean; image_url: string;
  design: boolean; lines: number; areas: number; approved: boolean;
}

export interface Project extends ProjectSummary { data: ProjectData }
