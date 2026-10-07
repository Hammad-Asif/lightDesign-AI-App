import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from "react";

import { fileUrl } from "../api";
import { formatLength, spreadText } from "../components/MeasurePanel";
import { KIND_LABELS, LOW_CONFIDENCE, type LineGeometry, type Photo, type Region, type RegionPlacement, type Tool } from "../types";
import {
  createLayer,
  ctx2d,
  distanceToSegment,
  layerToBlob,
  loadImage,
  paintMaskInto,
  slicePolyline,
} from "./layers";

// Wheel zoom: how fast a notch zooms, and how far in it may go (1 fits the photo).
const WHEEL_ZOOM_RATE = 0.0015;
export const MAX_ZOOM = 6;
// The canvas is drawn at the size it is shown, times the screen's pixel density, so lines and the
// photo stay sharp when zoomed in; held under what browsers allow (iOS: about 16 million pixels).
const MAX_CANVAS_AREA = 16_000_000;
const MAX_CANVAS_SIDE = 8192;
// A finger on a tablet waits this long before it paints or picks, so a second finger arriving
// for a pinch does not leave a dot or a tap behind.
const TOUCH_HOLD_MS = 90;
const TOUCH_SLOP_PX = 6;
const EDGE_NAMES: Record<string, string> = {
  eave: "Eave", rake: "Rake", hip: "Hip", ridge: "Ridge", valley: "Valley", junction: "Roof meets wall", unknown: "Type to check",
};
// Halo under a roof line whose edge type could not be identified.
const UNKNOWN_HALO = "rgba(244, 63, 94, 0.6)";
// Debug overlay: which source a line at a pipeline stage came from.
const DEBUG_COLOURS: Record<string, string> = {
  v3: "#22d3ee",
  partition: "#f59e0b",
};

// A bulb's "#rrggbb" colour with an alpha, for its glow.
function withAlpha(colour: string, alpha: number): string {
  const hex = /^#?([0-9a-f]{6})$/i.exec(colour)?.[1];
  if (!hex) return `rgba(255, 220, 160, ${alpha})`;
  const value = parseInt(hex, 16);
  return `rgba(${(value >> 16) & 255}, ${(value >> 8) & 255}, ${value & 255}, ${alpha})`;
}

// Even-odd ray test: is the point inside the closed polygon?
function insidePolygon(point: number[], polygon: number[][]): boolean {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];
    if (yi > point[1] !== yj > point[1] && point[0] < ((xj - xi) * (point[1] - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

// The editor's undo history lives in the app, with every other kind of change; the canvas hands it
// the state before each stroke or line edit, and puts a state back when asked.
export type Snapshot =
  | { type: "mask"; regionId: string; data: string }
  | { type: "poly"; regionId: string; data: number[][] };

export interface MaskCanvasHandle {
  // The region's current mask or line, to put back on redo.
  capture: (regionId: string, type: "mask" | "poly") => Snapshot | null;
  restore: (snapshot: Snapshot) => Promise<void>;
  // An area's painted mask as a PNG data URL (null for a line, or one not loaded).
  maskData: (regionId: string) => string | null;
}

interface Props {
  photo: Photo;
  regions: Region[];
  activeId: string | null;
  tool: Tool;
  brushSize: number;
  // 1 fits the whole photo in view; higher values enlarge it within the scrolling bed.
  zoom: number;
  // Objects in front of the house are detected so lines can continue behind them, not because
  // anyone asked for them: off the photo unless shown on purpose or one is selected.
  showObstacles: boolean;
  onMaskCommit: (regionId: string, blob: Blob) => Promise<Region>;
  onPolylineCommit: (regionId: string, polyline: number[][]) => void;
  // A line or area picked (null: a tap on empty space clears the selection).
  onSelect: (regionId: string | null) => void;
  // Select tool: a tap on the roof itself, off every line and area, selects the whole house.
  houseSelected?: boolean;
  onSelectHouse?: () => void;
  onBox: (box: number[]) => void;
  onPoint: (point: number[]) => void;
  // The House tool: a tap on a roof plane moves it into or out of the house being quoted.
  onHouseTap: (point: number[]) => void;
  onRemoveAt: (regionId: string) => void;
  // The state before a stroke or a line edit: one undo step.
  onRecord: (snapshot: Snapshot) => void;
  // The mouse wheel over the photo zooms it, about the point under the pointer.
  onZoom?: (zoom: number) => void;
  drawingColor: string | null;
  onDrawFinish: (polyline: number[][]) => void;
  onDrawCancel: () => void;
  // Development: draw the roof lines as they were at one stage of the pipeline (ROOFLINE_DEBUG).
  debugStage?: string | null;
  // Measure tool: two tapped points on a feature, or (mode "line") a tap that picks a line.
  measureMode?: "points" | "line";
  measurePoints?: number[][];
  onMeasureTap?: (point: number[]) => void;
  // A line the geometry cannot place: a handle at its real middle, dragged along it. `mid` is the
  // saved one, or null for the drawn middle.
  overrideLine?: { regionId: string; mid: number[] | null } | null;
  onOverrideMid?: (regionId: string, mid: number[]) => void;
  // The confirmed lighting's bulbs, by region id, drawn over the lines they light.
  bulbs?: Record<string, RegionPlacement> | null;
  // Wreath tool: a tap places a wreath there.
  onWreathTap?: (point: number[]) => void;
  // Each roof line's real length (GET /photos/{id}/geometry), drawn on the line when showLengths.
  geometry?: Record<string, LineGeometry> | null;
  showLengths?: boolean;
  // Development (SCALE_DEBUG): one step's overlay, drawn instead of the photo and its regions;
  // only the selected line is drawn over it, so a click still picks a line to inspect.
  backdrop?: string | null;
  // Night preview: the photo darkened, nothing but the lights, glowing.
  night?: boolean;
}

type Drag =
  | { mode: "stroke"; regionId: string; erase: boolean }
  | { mode: "vertex"; regionId: string; index: number }
  | { mode: "translate"; regionId: string; start: number[]; origin: number[][] }
  // Lengthening or shortening a run: the end slides along the line's own direction, so the
  // line keeps its angle. `limit` stops the end being dragged back past its neighbour.
  | { mode: "endpoint"; regionId: string; index: number; origin: number[][]; axis: number[]; limit: number }
  | { mode: "box"; start: number[] }
  // The midpoint handle of an overridden line, slid along that line.
  | { mode: "mid"; regionId: string }
  | null;

export const MaskCanvas = forwardRef<MaskCanvasHandle, Props>(function MaskCanvas(
  {
    photo,
    regions,
    activeId,
    tool,
    brushSize,
    zoom,
    showObstacles,
    onMaskCommit,
    onPolylineCommit,
    onSelect,
    houseSelected = false,
    onSelectHouse,
    onBox,
    onPoint,
    onHouseTap,
    onRemoveAt,
    onRecord,
    onZoom,
    drawingColor,
    onDrawFinish,
    onDrawCancel,
    debugStage = null,
    measureMode = "points",
    measurePoints = [],
    onMeasureTap,
    overrideLine = null,
    onOverrideMid,
    bulbs = null,
    onWreathTap,
    geometry = null,
    showLengths = false,
    backdrop = null,
    night = false,
  },
  ref,
) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const backdropRef = useRef<HTMLImageElement | null>(null);
  const layers = useRef(new Map<string, HTMLCanvasElement>());
  const loadedKeys = useRef(new Map<string, string>());

  const regionsRef = useRef(regions);
  const activeRef = useRef(activeId);
  const draftRef = useRef<{ regionId: string; polyline: number[][] } | null>(null);
  const dragRef = useRef<Drag>(null);
  const pointerRef = useRef<number[] | null>(null);
  const boxRef = useRef<number[] | null>(null);
  const drawRef = useRef<number[][]>([]);
  const frameRef = useRef(0);

  const photoRef = useRef(photo);
  const obstaclesRef = useRef(showObstacles);
  const measureRef = useRef(measurePoints);
  measureRef.current = measurePoints;
  const bulbsRef = useRef(bulbs);
  bulbsRef.current = bulbs;
  const lengthsRef = useRef(showLengths ? geometry : null);
  lengthsRef.current = showLengths ? geometry : null;
  const overrideRef = useRef(overrideLine);
  overrideRef.current = overrideLine;
  const houseRef = useRef(houseSelected);
  houseRef.current = houseSelected;
  const nightRef = useRef(night);
  nightRef.current = night;
  // The line or area under the pointer, named on the photo.
  const hoverRef = useRef<string | null>(null);
  const midDraftRef = useRef<number[] | null>(null);

  // Where the midpoint handle sits: being dragged, saved, or the drawn line's own middle.
  const handleAt = (): number[] | null => {
    const target = overrideRef.current;
    const line = target && regionsRef.current.find((r) => r.id === target.regionId)?.polyline;
    if (!target || !line || line.length < 2) return null;
    if (midDraftRef.current) return midDraftRef.current;
    if (target.mid) return target.mid;
    const [a, b] = [line[0], line[line.length - 1]];
    return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
  };

  // A point slid onto the line's chord, kept just inside its ends.
  const ontoLine = (point: number[]): number[] | null => {
    const target = overrideRef.current;
    const line = target && regionsRef.current.find((r) => r.id === target.regionId)?.polyline;
    if (!line || line.length < 2) return null;
    const [a, b] = [line[0], line[line.length - 1]];
    const [dx, dy] = [b[0] - a[0], b[1] - a[1]];
    const t = Math.min(0.95, Math.max(0.05, ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / (dx * dx + dy * dy || 1)));
    return [a[0] + t * dx, a[1] + t * dy];
  };
  regionsRef.current = regions;
  activeRef.current = activeId;
  photoRef.current = photo;
  obstaclesRef.current = showObstacles;

  // Zoom 1 fits the photo to the bed in both directions, so the whole house is on screen; the
  // bed is measured rather than the window, since the panels around it change size.
  const bedRef = useRef<HTMLDivElement>(null);
  const [fitWidth, setFitWidth] = useState<number | null>(null);
  useLayoutEffect(() => {
    const bed = bedRef.current;
    if (!bed) return;
    const measure = () => {
      const style = getComputedStyle(bed);
      const width = bed.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
      const height = bed.clientHeight - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom);
      setFitWidth(Math.max(80, Math.min(width, (height * photo.width) / photo.height)));
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(bed);
    return () => observer.disconnect();
  }, [photo.width, photo.height]);

  const [backing, setBacking] = useState({ width: photo.width, height: photo.height });
  useLayoutEffect(() => {
    if (!fitWidth) return;
    const ratio = photo.height / photo.width;
    let width = fitWidth * zoom * (window.devicePixelRatio || 1);
    width = Math.min(width, MAX_CANVAS_SIDE, MAX_CANVAS_SIDE / ratio, Math.sqrt(MAX_CANVAS_AREA / ratio));
    const next = { width: Math.max(1, Math.round(width)), height: Math.max(1, Math.round(width * ratio)) };
    setBacking((current) => (Math.abs(current.width - next.width) > 1 ? next : current));
  }, [fitWidth, zoom, photo.width, photo.height]);

  const quietObstacle = (region: Region) =>
    region.role === "blocker" && !obstaclesRef.current && region.id !== activeRef.current;

  // The house's roofs as polygons: the planes on the house being quoted, else (a photo with no
  // stored planes) every plane the partition drew.
  const housePolygons = (): number[][][] => {
    const interpreted = photoRef.current.interpreted;
    const planes = interpreted?.target?.planes;
    if (planes?.length) return planes.filter((plane) => plane.keep).flatMap((plane) => plane.outline).filter((p) => p.length >= 3);
    return (interpreted?.planes ?? []).map((plane) => plane.polygon).filter((p): p is number[][] => Boolean(p && p.length >= 3));
  };

  const onHouse = (point: number[]) => housePolygons().some((polygon) => insidePolygon(point, polygon));

  const lowConfidence = () => photoRef.current.interpreted?.roofline?.confidence_levels?.low ?? LOW_CONFIDENCE;

  // The lines of one pipeline stage, coloured by source, with each line's id at its middle. The raw
  // V3 stage is in crop pixels: it is placed by the crop's corner here, independently of the server's
  // conversion, so the two can be compared.
  const drawDebugStage = (context: CanvasRenderingContext2D, factor: number) => {
    const debug = photoRef.current.interpreted?.roofline_debug;
    // Only a photo detected with ROOFLINE_DEBUG has stages; without them nothing is drawn, the
    // crop outline included.
    if (!debug?.stages?.[debugStage ?? ""]) return;
    const lines = debug.stages[debugStage ?? ""];
    const crop = photoRef.current.interpreted?.roofline?.crop;
    if (crop) {
      context.strokeStyle = "rgba(34, 211, 238, 0.8)";
      context.lineWidth = 1.5 / factor;
      context.setLineDash([6 / factor, 6 / factor]);
      context.strokeRect(crop[0], crop[1], crop[2] - crop[0], crop[3] - crop[1]);
      context.setLineDash([]);
    }
    context.font = `${12 / factor}px sans-serif`;
    for (const line of lines) {
      let points = line.points;
      // A second call on a missed plane has its own crop.
      const origin = line.crop ?? crop;
      if (!points && line.crop_points && origin) {
        const [x1, y1, x2, y2] = line.crop_points;
        points = [
          [x1 + origin[0], y1 + origin[1]],
          [x2 + origin[0], y2 + origin[1]],
        ];
      }
      if (!points || points.length < 2) continue;
      const sources = line.source ?? ["v3"];
      context.strokeStyle = DEBUG_COLOURS[sources[0]] ?? DEBUG_COLOURS.v3;
      context.lineWidth = 2.5 / factor;
      context.beginPath();
      points.forEach((p, i) => (i ? context.lineTo(p[0], p[1]) : context.moveTo(p[0], p[1])));
      context.stroke();
      if (line.id) {
        const [start, end] = [points[0], points[points.length - 1]];
        context.fillStyle = "#ffffff";
        context.fillText(line.id, (start[0] + end[0]) / 2, (start[1] + end[1]) / 2 - 4 / factor);
      }
    }
  };

  // Each measured roof line's real length in a small pill at its middle, set off the line toward
  // the sky so it covers neither the line nor its bulbs. Approximate lengths read "≈", a line that
  // cannot be measured yet shows an amber "?". Pills never overlap: one that would is tried on
  // the other side of its line, then left out (the sidebar still lists it). The selected line's
  // pill is drawn last, always.
  const drawLengths = (context: CanvasRenderingContext2D, factor: number) => {
    const lines = lengthsRef.current;
    if (!lines) return;
    const size = 12 / factor;
    const pad = 4 / factor;
    const height = size + 2 * pad;
    const gap = 7 / factor;
    context.font = `600 ${size}px system-ui, -apple-system, sans-serif`;
    context.textAlign = "center";
    context.textBaseline = "middle";
    const taken: number[][] = [];
    const overlaps = (box: number[]) =>
      taken.some((t) => box[0] < t[2] && box[2] > t[0] && box[1] < t[3] && box[3] > t[1]);
    const pill = (box: number[], text: string, fill: string, ink: string, rim: string | null) => {
      context.beginPath();
      context.roundRect(box[0], box[1], box[2] - box[0], box[3] - box[1], height / 2);
      context.fillStyle = fill;
      context.fill();
      if (rim) {
        context.strokeStyle = rim;
        context.lineWidth = 1.5 / factor;
        context.stroke();
      }
      context.fillStyle = ink;
      context.fillText(text, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2 + 0.5 / factor);
    };
    // The salesperson's reference: amber, like its ruler, and placed first so nothing covers it.
    const reference = photoRef.current.scale?.measurement;
    if (reference?.points && reference.points.length >= 2) {
      const [a, b] = [reference.points[0], reference.points[reference.points.length - 1]];
      const text = `${formatLength(reference.length_in)} reference`;
      const width = context.measureText(text).width + 2 * pad;
      const cx = (a[0] + b[0]) / 2;
      const cy = (a[1] + b[1]) / 2 - gap - height / 2;
      const box = [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2];
      taken.push(box);
      pill(box, text, "#f59e0b", "#1f2937", null);
    }
    const NEEDS: Record<string, string> = { side: "? side", override: "? length", length: "? length", measurement: "? measure",
                                             measurement_wall: "? measure", anchor: "? length", edge_on: "? length",
                                             unreliable: "? unsure" };
    const dragging = draftRef.current?.regionId;
    const order = regionsRef.current
      .filter((r) => r.geometry === "line" && r.kind === "roof" && (r.polyline?.length ?? 0) >= 2 && lines[r.id])
      .filter((r) => r.id !== dragging)
      .sort((a, b) => Number(a.id === activeRef.current) - Number(b.id === activeRef.current));
    for (const region of order) {
      const line = lines[region.id];
      const exact = line.confidence === "exact";
      const isReference = reference?.region_id === region.id || line.method === "measured";
      const doubtful = Boolean(line.warning);
      const text = line.length_in != null
        ? `${exact || isReference ? "" : "≈ "}${formatLength(line.length_in)}${isReference ? " reference" : spreadText(line.spread)}`
        : line.needs ? NEEDS[line.needs] ?? "?" : null;
      const shown = doubtful && text ? `${text} ?` : text;
      if (!text) continue;
      const points = region.polyline!;
      // The middle by length along the line, and the line's direction there.
      const spans = points.slice(1).map((p, i) => Math.hypot(p[0] - points[i][0], p[1] - points[i][1]));
      let rest = spans.reduce((a, b) => a + b, 0) / 2;
      let index = 0;
      while (index < spans.length - 1 && rest > spans[index]) rest -= spans[index++];
      const [a, b] = [points[index], points[index + 1]];
      const t = spans[index] ? rest / spans[index] : 0;
      const mid = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
      const length = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
      let normal = [-(b[1] - a[1]) / length, (b[0] - a[0]) / length];
      if (normal[1] > 0) normal = [-normal[0], -normal[1]];
      const width = context.measureText(shown!).width + 2 * pad;
      const active = region.id === activeRef.current;
      const at = (side: number) => {
        const cx = mid[0] + normal[0] * side * (gap + height / 2);
        const cy = mid[1] + normal[1] * side * (gap + height / 2);
        return [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2];
      };
      // The sky side, else the other side; the selected line's pill is drawn even if it collides.
      const box = [at(1), at(-1)].find((candidate) => !overlaps(candidate)) ?? (active ? at(1) : null);
      if (!box) continue;
      taken.push(box);
      const needs = line.length_in == null;
      context.globalAlpha = region.visible || active ? 1 : 0.6;
      // A line still needing input is pale yellow with a dark amber rim: a warning, and plainly not
      // the amber reference.
      if (isReference) pill(box, shown!, "#f59e0b", "#1f2937", active ? "#ffffff" : null);
      else if (needs || doubtful) pill(box, shown!, "#fde68a", "#78350f", "#b45309");
      else pill(box, shown!, exact ? "rgba(17, 24, 39, 0.88)" : "rgba(55, 65, 81, 0.78)", exact ? "#ffffff" : "#e5e7eb",
                active ? "#ffffff" : null);
      context.globalAlpha = 1;
    }
    context.textAlign = "start";
    context.textBaseline = "alphabetic";
  };

  // The bulbs: garland bands under their lights, icicle drops as thin wires, then each bulb at its
  // placed size and colour. Strung lights (trunk wraps, fronds) show their wire and a glow; at night
  // every bulb glows and nothing else is drawn. Hidden ones (behind an object) are never drawn.
  const drawBulbs = (context: CanvasRenderingContext2D, factor: number, night: boolean) => {
    for (const placed of Object.values(bulbsRef.current ?? {})) {
      if (placed.ring) {
        const [x, y, diameter] = placed.ring;
        context.strokeStyle = night ? "rgba(20, 52, 28, 0.9)" : "rgba(34, 92, 44, 0.9)";
        context.lineWidth = diameter * 0.16;
        context.beginPath();
        context.arc(x, y, diameter * 0.42, 0, Math.PI * 2);
        context.stroke();
      }
      if (placed.band && placed.band.length > 1) {
        context.strokeStyle = night ? "rgba(20, 52, 28, 0.6)" : "rgba(34, 92, 44, 0.55)";
        context.lineCap = "round";
        context.lineWidth = placed.band.reduce((sum, b) => sum + b[2], 0) / placed.band.length;
        context.beginPath();
        placed.band.forEach((b, i) => (i ? context.lineTo(b[0], b[1]) : context.moveTo(b[0], b[1])));
        context.stroke();
      }
      context.strokeStyle = night ? "rgba(255, 255, 255, 0.25)" : "rgba(255, 255, 255, 0.55)";
      context.lineWidth = 1 / factor;
      for (const [x0, y0, x1, y1] of placed.drops ?? []) {
        context.beginPath();
        context.moveTo(x0, y0);
        context.lineTo(x1, y1);
        context.stroke();
      }
      const strung = Boolean(placed.trunk || placed.fronds);
      if (strung) {
        const sizes = (placed.bulbs ?? []).map((bulb) => bulb[2]);
        context.strokeStyle = "rgba(24, 48, 28, 0.75)";
        context.lineWidth = Math.max(0.8 / factor, (sizes.reduce((sum, d) => sum + d, 0) / (sizes.length || 1)) * 0.3);
        context.lineCap = "round";
        context.lineJoin = "round";
        if (placed.fronds) {
          for (const path of placed.fronds.paths) {
            context.beginPath();
            path.forEach(([px, py], i) => (i ? context.lineTo(px, py) : context.moveTo(px, py)));
            context.stroke();
          }
        } else {
          context.beginPath();
          let open = false;
          for (const [px, py, , , hidden] of placed.bulbs ?? []) {
            if (hidden) {
              open = false;
              continue;
            }
            if (open) context.lineTo(px, py);
            else context.moveTo(px, py);
            open = true;
          }
          context.stroke();
        }
      }
      if (strung || night) {
        context.save();
        context.globalCompositeOperation = "lighter";
        for (const [x, y, diameter, colour, hidden] of placed.bulbs ?? []) {
          if (hidden) continue;
          const radius = Math.max(2 / factor, diameter * (night ? 2.6 : 1.8));
          const glow = context.createRadialGradient(x, y, 0, x, y, radius);
          glow.addColorStop(0, withAlpha(colour, night ? 0.8 : 0.55));
          glow.addColorStop(1, withAlpha(colour, 0));
          context.fillStyle = glow;
          context.beginPath();
          context.arc(x, y, radius, 0, Math.PI * 2);
          context.fill();
        }
        context.restore();
      }
      for (const [x, y, diameter, colour, hidden] of placed.bulbs ?? []) {
        if (hidden) continue;
        context.beginPath();
        context.arc(x, y, Math.max(1.2 / factor, diameter / 2), 0, Math.PI * 2);
        context.fillStyle = colour;
        context.fill();
        if (night) continue;
        context.strokeStyle = "rgba(0, 0, 0, 0.55)";
        context.lineWidth = 0.8 / factor;
        context.stroke();
      }
    }
  };

  // What a line or area is, in words: "Eave · front", "Hedge".
  const nameOf = (region: Region) =>
    region.kind === "roof" && region.geometry === "line"
      ? `${EDGE_NAMES[region.edge ?? "unknown"] ?? region.edge}${region.side ? ` · ${region.side}` : ""}${region.visible ? "" : " · not lit"}`
      : KIND_LABELS[region.kind] ?? region.label;

  // The selected line named beside its middle, on the side away from its length; the hovered one
  // beside the pointer. A line type never rests on colour alone.
  const drawNames = (context: CanvasRenderingContext2D, factor: number) => {
    const size = 12 / factor;
    const pad = 5 / factor;
    context.font = `600 ${size}px system-ui, -apple-system, sans-serif`;
    context.textBaseline = "middle";
    const tag = (x: number, y: number, text: string, colour: string) => {
      const width = context.measureText(text).width + 2 * pad + 10 / factor;
      const height = size + 2 * pad;
      context.beginPath();
      context.roundRect(x, y - height / 2, width, height, height / 2);
      context.fillStyle = "rgba(17, 24, 39, 0.9)";
      context.fill();
      context.beginPath();
      context.arc(x + pad + 3 / factor, y, 3.5 / factor, 0, Math.PI * 2);
      context.fillStyle = colour;
      context.fill();
      context.fillStyle = "#ffffff";
      context.fillText(text, x + pad + 10 / factor, y + 0.5 / factor);
    };
    const selected = regionsRef.current.find((r) => r.id === activeRef.current);
    if (selected?.geometry === "line" && selected.polyline && selected.polyline.length >= 2 && !draftRef.current) {
      const points = selected.polyline;
      const [a, b] = [points[0], points[points.length - 1]];
      const length = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
      let normal = [-(b[1] - a[1]) / length, (b[0] - a[0]) / length];
      if (normal[1] < 0) normal = [-normal[0], -normal[1]];
      const middle = points[Math.floor(points.length / 2)];
      const centre = points.length === 2 ? [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2] : middle;
      tag(centre[0] + normal[0] * 16 / factor, centre[1] + normal[1] * 16 / factor, nameOf(selected), selected.color);
    }
    const hovered = hoverRef.current && hoverRef.current !== activeRef.current
      ? regionsRef.current.find((r) => r.id === hoverRef.current)
      : undefined;
    if (hovered && pointerRef.current && !dragRef.current) {
      tag(pointerRef.current[0] + 14 / factor, pointerRef.current[1] - 16 / factor, nameOf(hovered), hovered.color);
    }
    context.textBaseline = "alphabetic";
  };

  const scale = () => {
    const canvas = canvasRef.current;
    if (!canvas) return 1;
    const rect = canvas.getBoundingClientRect();
    return rect.width / photoRef.current.width || 1;
  };

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    if (!canvas || !image) return;

    // Not ctx2d: the shown canvas is never read back, so it can stay on the GPU.
    const context = canvas.getContext("2d");
    if (!context) return;
    const factor = scale();
    const { width: photoWidth, height: photoHeight } = photoRef.current;
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.clearRect(0, 0, canvas.width, canvas.height);
    // Everything below is drawn in photo pixels.
    const toBacking = canvas.width / photoWidth;
    context.setTransform(toBacking, 0, 0, toBacking, 0, 0);
    context.imageSmoothingQuality = "high";
    if (backdrop && backdropRef.current) {
      context.drawImage(backdropRef.current, 0, 0, photoWidth, photoHeight);
      const selected = regionsRef.current.find((r) => r.id === activeRef.current);
      if (selected?.polyline && selected.polyline.length >= 2) {
        context.strokeStyle = "#ffffff";
        context.lineWidth = 5 / factor;
        context.setLineDash([10 / factor, 6 / factor]);
        context.beginPath();
        selected.polyline.forEach((p, i) => (i ? context.lineTo(p[0], p[1]) : context.moveTo(p[0], p[1])));
        context.stroke();
        context.setLineDash([]);
      }
      return;
    }
    context.drawImage(image, 0, 0, photoWidth, photoHeight);

    // Night preview: the photo as at dusk, and only the lights on it.
    if (nightRef.current) {
      context.fillStyle = "rgba(6, 10, 28, 0.74)";
      context.fillRect(0, 0, photoWidth, photoHeight);
      drawBulbs(context, factor, true);
      return;
    }

    // Roof planes from the partition, filled under the lines the way the reference app draws
    // them, so a glance shows which plane each edge belongs to.
    for (const plane of photoRef.current.interpreted?.planes ?? []) {
      if (!plane.polygon || plane.polygon.length < 3) continue;
      context.beginPath();
      plane.polygon.forEach((p, i) => (i ? context.lineTo(p[0], p[1]) : context.moveTo(p[0], p[1])));
      context.closePath();
      context.fillStyle = plane.color;
      context.globalAlpha = 0.2;
      context.fill();
      context.globalAlpha = 1;
    }

    // House tool: every roof plane detection found, green when it is on the house being quoted
    // and red, dashed, when it was left out as a neighbour, so it is plain what a tap will do.
    if (tool === "house") {
      for (const plane of photoRef.current.interpreted?.target?.planes ?? []) {
        for (const polygon of plane.outline) {
          if (polygon.length < 3) continue;
          context.beginPath();
          polygon.forEach((p, i) => (i ? context.lineTo(p[0], p[1]) : context.moveTo(p[0], p[1])));
          context.closePath();
          context.fillStyle = plane.keep ? "rgba(57, 255, 20, 0.22)" : "rgba(255, 59, 48, 0.2)";
          context.fill();
          context.lineWidth = 2 / factor;
          context.strokeStyle = plane.keep ? "#39ff14" : "rgba(255, 255, 255, 0.95)";
          context.setLineDash(plane.keep ? [] : [8 / factor, 6 / factor]);
          context.stroke();
          context.setLineDash([]);
        }
      }
    }

    // The house selected with Select: its roofs outlined, so it is plain what is picked.
    if (houseRef.current && tool !== "house") {
      for (const polygon of housePolygons()) {
        context.beginPath();
        polygon.forEach((p, i) => (i ? context.lineTo(p[0], p[1]) : context.moveTo(p[0], p[1])));
        context.closePath();
        context.fillStyle = "rgba(37, 99, 235, 0.16)";
        context.fill();
        context.lineWidth = 3 / factor;
        context.strokeStyle = "#ffffff";
        context.stroke();
        context.lineWidth = 1.5 / factor;
        context.strokeStyle = "#2563eb";
        context.stroke();
      }
    }

    const junctions: number[][] = [];
    for (const region of regionsRef.current) {
      // A roof line the instruction did not ask for is still drawn, dashed and dim: detected,
      // not lit. Every other hidden region stays off the canvas.
      const unlit = !region.visible && region.kind === "roof" && region.geometry === "line";
      if (!region.visible && !unlit) continue;
      if (quietObstacle(region)) continue;
      const isActive = region.id === activeRef.current;

      if (region.geometry === "area") {
        const layer = layers.current.get(region.id);
        if (!layer) continue;
        context.globalAlpha = isActive ? 0.55 : 0.3;
        context.drawImage(layer, 0, 0);
        context.globalAlpha = 1;
        continue;
      }

      const points =
        draftRef.current?.regionId === region.id
          ? draftRef.current.polyline
          : region.polyline;
      if (!points || points.length < 2) continue;

      const trace = (path: number[][]) => {
        context.beginPath();
        path.forEach((p, i) => (i ? context.lineTo(p[0], p[1]) : context.moveTo(p[0], p[1])));
      };
      context.lineJoin = "round";
      context.lineCap = "round";

      // An edge we could not identify gets a warning halo and a dotted line, so it reads as
      // "check this" at a glance rather than passing for a confirmed eave or rake.
      const unidentified = region.kind === "roof" && region.edge === "unknown";
      if (unidentified) {
        context.strokeStyle = UNKNOWN_HALO;
        context.lineWidth = (isActive ? 12 : 9) / factor;
        trace(points);
        context.stroke();
        context.setLineDash([0.5 / factor, 7 / factor]);
      }

      if (region.kind === "roof") {
        // a dark halo keeps a line crisp against bright sky and light shingles; an unlit edge
        // is dashed and a little quieter, but still legible on terracotta tile. A detected line
        // with little evidence behind it is dashed short, so it reads as "check this one".
        const doubtful =
          region.source !== "manual" && region.confidence != null && region.confidence < lowConfidence();
        context.globalAlpha = unlit ? 0.8 : 1;
        context.setLineDash(
          unlit
            ? [8 / factor, 6 / factor]
            : unidentified
              ? [0.5 / factor, 7 / factor]
              : doubtful
                ? [4 / factor, 4 / factor]
                : [],
        );
        context.strokeStyle = "rgba(0, 0, 0, 0.6)";
        context.lineWidth = (isActive ? 8 : 6) / factor;
        trace(points);
        context.stroke();
        context.strokeStyle = region.color;
        context.lineWidth = (isActive ? 4.5 : 3) / factor;
        trace(points);
        context.stroke();
        context.setLineDash([]);
        context.globalAlpha = 1;
        if (!unlit) junctions.push(points[0], points[points.length - 1]);
      } else {
        context.strokeStyle = region.color;
        context.lineWidth = (isActive ? 4.5 : 2.5) / factor;
        trace(points);
        context.stroke();
        context.setLineDash([]);
      }

      // Stretches that pass behind an object are dashed. They are measured along the saved
      // line, so they are skipped while that line is being dragged.
      const hidden = draftRef.current?.regionId === region.id ? [] : region.hidden ?? [];
      if (hidden.length) {
        context.strokeStyle = "rgba(8, 12, 17, 0.9)";
        context.lineWidth = (isActive ? 5.5 : 3.5) / factor;
        context.setLineDash([7 / factor, 7 / factor]);
        for (const [start, end] of hidden) {
          const part = slicePolyline(points, start, end);
          if (part.length < 2) continue;
          context.beginPath();
          part.forEach((p, i) => (i ? context.lineTo(p[0], p[1]) : context.moveTo(p[0], p[1])));
          context.stroke();
        }
        context.setLineDash([]);
      }

      if (isActive) {
        const radius = 5 / factor;
        context.lineWidth = 2 / factor;
        for (const point of points) {
          context.beginPath();
          context.arc(point[0], point[1], radius, 0, Math.PI * 2);
          context.fillStyle = "#ffffff";
          context.fill();
          context.strokeStyle = region.color;
          context.stroke();
        }
      }
    }

    // junctions: where roof lines meet, drawn once on top of every line
    if (junctions.length) {
      const radius = 4.5 / factor;
      context.lineWidth = 1.5 / factor;
      for (const [x, y] of junctions) {
        context.beginPath();
        context.arc(x, y, radius, 0, Math.PI * 2);
        context.fillStyle = "#ff3b30";
        context.fill();
        context.strokeStyle = "rgba(0, 0, 0, 0.6)";
        context.stroke();
      }
    }

    if (debugStage) drawDebugStage(context, factor);

    // The saved measurement, and the points of one being tapped. Amber, so it never reads as a
    // roof line; the length sits beside it.
    const ruler = (points: number[][], label: string | null) => {
      context.strokeStyle = "#f59e0b";
      context.lineWidth = 3 / factor;
      context.beginPath();
      points.forEach((p, i) => (i ? context.lineTo(p[0], p[1]) : context.moveTo(p[0], p[1])));
      context.stroke();
      for (const point of [points[0], points[points.length - 1]]) {
        context.beginPath();
        context.arc(point[0], point[1], 6 / factor, 0, Math.PI * 2);
        context.fillStyle = "#ffffff";
        context.fill();
        context.stroke();
      }
      if (label) {
        const [a, b] = [points[0], points[points.length - 1]];
        context.font = `bold ${14 / factor}px sans-serif`;
        context.lineWidth = 4 / factor;
        context.strokeStyle = "rgba(0, 0, 0, 0.75)";
        context.strokeText(label, (a[0] + b[0]) / 2 + 8 / factor, (a[1] + b[1]) / 2 - 8 / factor);
        context.fillStyle = "#fbbf24";
        context.fillText(label, (a[0] + b[0]) / 2 + 8 / factor, (a[1] + b[1]) / 2 - 8 / factor);
      }
    };
    const saved = photoRef.current.scale?.measurement;
    if (saved) {
      const along = saved.region_id ? regionsRef.current.find((r) => r.id === saved.region_id)?.polyline : saved.points;
      // With lengths shown, the reference's own pill carries its length (drawLengths).
      if (along && along.length >= 2) ruler(along, lengthsRef.current ? null : formatLength(saved.length_in));
    }
    if (tool === "measure" && measureRef.current.length) {
      const pending = measureRef.current.length === 1 && pointerRef.current
        ? [measureRef.current[0], pointerRef.current]
        : measureRef.current;
      if (pending.length >= 2) ruler(pending, null);
      else ruler([pending[0], pending[0]], null);
    }

    drawBulbs(context, factor, false);

    drawLengths(context, factor);
    drawNames(context, factor);

    if (boxRef.current) {
      const [x0, y0, x1, y1] = boxRef.current;
      context.strokeStyle = tool === "find" ? "#22d3ee" : "#f472b6";
      context.lineWidth = 2 / factor;
      context.setLineDash([8 / factor, 6 / factor]);
      context.strokeRect(x0, y0, x1 - x0, y1 - y0);
      context.setLineDash([]);
    }

    if (drawingColor && drawRef.current.length) {
      const points = pointerRef.current
        ? [...drawRef.current, pointerRef.current]
        : drawRef.current;
      context.strokeStyle = drawingColor;
      context.lineWidth = 4 / factor;
      context.lineJoin = "round";
      context.lineCap = "round";
      context.setLineDash([10 / factor, 6 / factor]);
      context.beginPath();
      points.forEach((p, i) => (i ? context.lineTo(p[0], p[1]) : context.moveTo(p[0], p[1])));
      context.stroke();
      context.setLineDash([]);
      for (const point of drawRef.current) {
        context.beginPath();
        context.arc(point[0], point[1], 5 / factor, 0, Math.PI * 2);
        context.fillStyle = "#ffffff";
        context.fill();
        context.strokeStyle = drawingColor;
        context.lineWidth = 2 / factor;
        context.stroke();
      }
    }

    const mid = handleAt();
    if (mid) {
      const size = 8 / factor;
      context.beginPath();
      context.moveTo(mid[0], mid[1] - size);
      context.lineTo(mid[0] + size, mid[1]);
      context.lineTo(mid[0], mid[1] + size);
      context.lineTo(mid[0] - size, mid[1]);
      context.closePath();
      context.fillStyle = "#ffffff";
      context.fill();
      context.strokeStyle = "#f59e0b";
      context.lineWidth = 2.5 / factor;
      context.stroke();
    }

    if (pointerRef.current && (tool === "brush" || tool === "erase")) {
      context.beginPath();
      context.arc(pointerRef.current[0], pointerRef.current[1], brushSize / 2, 0, Math.PI * 2);
      context.strokeStyle = tool === "erase" ? "#f87171" : "#ffffff";
      context.lineWidth = 1.5 / factor;
      context.stroke();
    }
  }, [brushSize, tool, drawingColor, debugStage, backdrop]);

  const requestDraw = useCallback(() => {
    if (frameRef.current) return;
    frameRef.current = requestAnimationFrame(() => {
      frameRef.current = 0;
      draw();
    });
  }, [draw]);

  useEffect(() => {
    backdropRef.current = null;
    if (!backdrop) {
      requestDraw();
      return;
    }
    let cancelled = false;
    loadImage(fileUrl(backdrop)).then((image) => {
      if (cancelled) return;
      backdropRef.current = image;
      requestDraw();
    });
    return () => {
      cancelled = true;
    };
  }, [backdrop, requestDraw]);

  useEffect(() => {
    let cancelled = false;
    loadImage(fileUrl(photo.image_url)).then((image) => {
      if (cancelled) return;
      imageRef.current = image;
      requestDraw();
    });
    return () => {
      cancelled = true;
    };
  }, [photo.image_url, requestDraw]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      for (const region of regions) {
        if (region.geometry !== "area" || !region.mask_url) continue;
        const key = `${region.mask_url}#${region.color}`;
        if (loadedKeys.current.get(region.id) === key) continue;

        const layer = layers.current.get(region.id) ?? createLayer(photo.width, photo.height);
        await paintMaskInto(layer, fileUrl(region.mask_url), region.color);
        if (cancelled) return;
        layers.current.set(region.id, layer);
        loadedKeys.current.set(region.id, key);
        requestDraw();
      }
      for (const id of [...layers.current.keys()]) {
        if (!regions.some((region) => region.id === id)) {
          layers.current.delete(id);
          loadedKeys.current.delete(id);
        }
      }
      requestDraw();
    })();
    return () => {
      cancelled = true;
    };
  }, [regions, photo.width, photo.height, requestDraw]);

  useEffect(requestDraw, [activeId, zoom, fitWidth, showObstacles, measurePoints, photo.scale, overrideLine, bulbs, geometry, showLengths, houseSelected, photo.interpreted, night, backing, requestDraw]);

  useEffect(() => {
    if (!drawingColor) {
      drawRef.current = [];
      return;
    }
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Enter" && drawRef.current.length >= 2) {
        event.preventDefault();
        const points = drawRef.current;
        drawRef.current = [];
        onDrawFinish(points);
      }
      if (event.key === "Escape") {
        drawRef.current = [];
        onDrawCancel();
      }
      if (event.key === "Backspace" && drawRef.current.length) {
        event.preventDefault();
        drawRef.current = drawRef.current.slice(0, -1);
        requestDraw();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [drawingColor, onDrawFinish, onDrawCancel, requestDraw]);

  const capture = (regionId: string, type: "mask" | "poly"): Snapshot | null => {
    if (type === "mask") {
      const layer = layers.current.get(regionId);
      return layer ? { type: "mask", regionId, data: layer.toDataURL("image/png") } : null;
    }
    const region = regionsRef.current.find((item) => item.id === regionId);
    return region?.polyline ? { type: "poly", regionId, data: region.polyline } : null;
  };

  const pushHistory = (regionId: string, type: "mask" | "poly") => {
    const snapshot = capture(regionId, type);
    if (snapshot) onRecord(snapshot);
  };

  const commitMask = async (regionId: string) => {
    const layer = layers.current.get(regionId);
    if (!layer) return;
    const region = await onMaskCommit(regionId, await layerToBlob(layer));
    loadedKeys.current.set(regionId, `${region.mask_url}#${region.color}`);
  };

  const restore = async (snapshot: Snapshot) => {
    if (snapshot.type === "poly") {
      onPolylineCommit(snapshot.regionId, snapshot.data);
      return;
    }
    const layer = layers.current.get(snapshot.regionId);
    if (!layer) return;
    const image = await loadImage(snapshot.data);
    const context = ctx2d(layer);
    context.clearRect(0, 0, layer.width, layer.height);
    context.drawImage(image, 0, 0);
    requestDraw();
    await commitMask(snapshot.regionId);
  };

  useImperativeHandle(ref, () => ({
    capture,
    restore,
    maskData: (regionId) => {
      const region = regionsRef.current.find((item) => item.id === regionId);
      const layer = layers.current.get(regionId);
      return region?.geometry === "area" && layer ? layer.toDataURL("image/png") : null;
    },
  }));

  // Wheel zoom about the pointer: the photo point under it is where it was after the zoom. The
  // listener is not passive, so the page does not scroll as well.
  const zoomRef = useRef(zoom);
  zoomRef.current = zoom;
  const onZoomRef = useRef(onZoom);
  onZoomRef.current = onZoom;
  const anchorRef = useRef<{ fx: number; fy: number; x: number; y: number } | null>(null);
  useEffect(() => {
    const bed = bedRef.current;
    if (!bed) return;
    const onWheel = (event: WheelEvent) => {
      const canvas = canvasRef.current;
      if (!canvas || !onZoomRef.current) return;
      event.preventDefault();
      const delta = event.deltaMode === 1 ? event.deltaY * 33 : event.deltaY;
      const next = Math.min(MAX_ZOOM, Math.max(1, zoomRef.current * Math.exp(-delta * WHEEL_ZOOM_RATE)));
      if (Math.abs(next - zoomRef.current) < 1e-3) return;
      const rect = canvas.getBoundingClientRect();
      anchorRef.current = {
        fx: (event.clientX - rect.left) / rect.width,
        fy: (event.clientY - rect.top) / rect.height,
        x: event.clientX,
        y: event.clientY,
      };
      onZoomRef.current(Math.abs(next - 1) < 0.02 ? 1 : next);
    };
    bed.addEventListener("wheel", onWheel, { passive: false });
    return () => bed.removeEventListener("wheel", onWheel);
  }, []);
  useLayoutEffect(() => {
    const anchor = anchorRef.current;
    const canvas = canvasRef.current;
    const bed = bedRef.current;
    anchorRef.current = null;
    if (!anchor || !canvas || !bed) return;
    const rect = canvas.getBoundingClientRect();
    bed.scrollLeft += rect.left + anchor.fx * rect.width - anchor.x;
    bed.scrollTop += rect.top + anchor.fy * rect.height - anchor.y;
  }, [zoom]);

  // Touch: every finger on the photo; a pinch in progress; a first finger waiting to act.
  const touches = useRef(new Map<number, number[]>());
  const pinchRef = useRef<{ distance: number; zoom: number; mid: number[] } | null>(null);
  const pendingRef = useRef<{ event: ReactPointerEvent<HTMLCanvasElement>; at: number[]; timer: number } | null>(null);
  // After a pinch, the finger left on the photo does nothing until it lifts.
  const afterPinch = useRef(false);

  // Panning a zoomed photo: drag with the middle button, or with Space held, whatever the tool.
  const panRef = useRef<{ x: number; y: number; left: number; top: number } | null>(null);
  const spaceRef = useRef(false);
  useEffect(() => {
    const down = (event: KeyboardEvent) => {
      const target = event.target;
      if (event.code !== "Space" || target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement
          || target instanceof HTMLSelectElement || target instanceof HTMLButtonElement) return;
      spaceRef.current = true;
      event.preventDefault();
    };
    const up = (event: KeyboardEvent) => {
      if (event.code === "Space") spaceRef.current = false;
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, []);

  const toImage = (event: ReactPointerEvent): number[] => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const factor = photoRef.current.width / rect.width;
    return [(event.clientX - rect.left) * factor, (event.clientY - rect.top) * factor];
  };

  const paint = (point: number[], erase: boolean) => {
    const layer = layers.current.get(activeRef.current!);
    const region = regionsRef.current.find((item) => item.id === activeRef.current);
    if (!layer || !region) return;
    const context = ctx2d(layer);
    context.globalCompositeOperation = erase ? "destination-out" : "source-over";
    context.fillStyle = region.color;
    context.beginPath();
    context.arc(point[0], point[1], brushSize / 2, 0, Math.PI * 2);
    context.fill();
    context.globalCompositeOperation = "source-over";
  };

  const activeRegion = () => regionsRef.current.find((item) => item.id === activeRef.current);

  const areaAt = (point: number[]) =>
    [...regionsRef.current].reverse().find((region) => {
      if (!region.visible || region.geometry !== "area" || quietObstacle(region)) return false;
      const layer = layers.current.get(region.id);
      if (!layer) return false;
      const x = Math.round(point[0]);
      const y = Math.round(point[1]);
      if (x < 0 || y < 0 || x >= layer.width || y >= layer.height) return false;
      return ctx2d(layer).getImageData(x, y, 1, 1).data[3] > 8;
    });

  /** The region a click lands on: the nearest line within reach, else the area painted under it.
   *  Lines win, because a line drawn over a mask is only a few pixels wide to aim at. */
  const regionAt = (point: number[]) => {
    const reach = 10 / scale();
    let nearest: { region: Region; distance: number } | undefined;
    for (const region of regionsRef.current) {
      if (!region.visible || region.geometry !== "line" || !region.polyline) continue;
      for (let i = 0; i < region.polyline.length - 1; i += 1) {
        const { distance } = distanceToSegment(point, region.polyline[i], region.polyline[i + 1]);
        if (distance <= reach && (!nearest || distance < nearest.distance)) {
          nearest = { region, distance };
        }
      }
    }
    return nearest?.region ?? areaAt(point);
  };

  /** Clicking away from the region being edited picks whatever was clicked instead. */
  const selectUnder = (point: number[]) => {
    const hit = regionAt(point);
    if (hit && hit.id !== activeRef.current) onSelect(hit.id);
  };

  /** The Select tool: a line or area under the tap, else the house when the tap is on its roof,
   *  else nothing (a tap on empty space clears the selection). */
  const pickUnder = (point: number[]) => {
    const hit = regionAt(point);
    if (hit) onSelect(hit.id);
    else if (onSelectHouse && onHouse(point)) onSelectHouse();
    else onSelect(null);
  };

  const onPointerDown = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const bed = bedRef.current;
    if (bed && (event.button === 1 || spaceRef.current)) {
      event.preventDefault();
      event.currentTarget.setPointerCapture(event.pointerId);
      panRef.current = { x: event.clientX, y: event.clientY, left: bed.scrollLeft, top: bed.scrollTop };
      return;
    }
    if (event.pointerType === "touch") {
      touches.current.set(event.pointerId, [event.clientX, event.clientY]);
      event.currentTarget.setPointerCapture(event.pointerId);
      if (touches.current.size >= 2) {
        startPinch();
        return;
      }
      if (afterPinch.current) return;
      // Kept for a moment: a second finger may be on its way.
      const held = {
        clientX: event.clientX,
        clientY: event.clientY,
        altKey: false,
        button: 0,
        pointerId: event.pointerId,
        pointerType: "touch",
        currentTarget: event.currentTarget,
        preventDefault: () => undefined,
      } as unknown as ReactPointerEvent<HTMLCanvasElement>;
      pendingRef.current = { event: held, at: [event.clientX, event.clientY], timer: window.setTimeout(firePending, TOUCH_HOLD_MS) };
      return;
    }
    begin(event);
  };

  // The held finger acts now (it moved, lifted, or waited long enough).
  const firePending = () => {
    const pending = pendingRef.current;
    if (!pending) return;
    window.clearTimeout(pending.timer);
    pendingRef.current = null;
    begin(pending.event);
  };

  const startPinch = () => {
    if (pendingRef.current) {
      window.clearTimeout(pendingRef.current.timer);
      pendingRef.current = null;
    }
    // Whatever one finger had started is dropped, not saved.
    dragRef.current = null;
    draftRef.current = null;
    boxRef.current = null;
    midDraftRef.current = null;
    const [a, b] = [...touches.current.values()];
    pinchRef.current = {
      distance: Math.hypot(a[0] - b[0], a[1] - b[1]) || 1,
      zoom: zoomRef.current,
      mid: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2],
    };
    afterPinch.current = true;
    requestDraw();
  };

  const movePinch = () => {
    const pinch = pinchRef.current;
    const canvas = canvasRef.current;
    const bed = bedRef.current;
    if (!pinch || !canvas || !bed || touches.current.size < 2) return;
    const [a, b] = [...touches.current.values()];
    const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    // Two fingers moving together move round the photo...
    bed.scrollLeft -= mid[0] - pinch.mid[0];
    bed.scrollTop -= mid[1] - pinch.mid[1];
    pinch.mid = mid;
    // ...and spreading or closing them zooms about the point between them.
    const next = Math.min(MAX_ZOOM, Math.max(1, (pinch.zoom * Math.hypot(a[0] - b[0], a[1] - b[1])) / pinch.distance));
    if (Math.abs(next - zoomRef.current) > 0.01 && onZoomRef.current) {
      const rect = canvas.getBoundingClientRect();
      anchorRef.current = { fx: (mid[0] - rect.left) / rect.width, fy: (mid[1] - rect.top) / rect.height, x: mid[0], y: mid[1] };
      onZoomRef.current(next);
    }
  };

  const begin = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const point = toImage(event);
    pointerRef.current = point;
    event.currentTarget.setPointerCapture(event.pointerId);

    if (drawingColor) {
      drawRef.current = [...drawRef.current, [Math.round(point[0]), Math.round(point[1])]];
      requestDraw();
      return;
    }

    const mid = handleAt();
    if (mid && overrideRef.current && Math.hypot(point[0] - mid[0], point[1] - mid[1]) <= 12 / scale()) {
      dragRef.current = { mode: "mid", regionId: overrideRef.current.regionId };
      midDraftRef.current = mid;
      return;
    }

    if (tool === "house") {
      onHouseTap([Math.round(point[0]), Math.round(point[1])]);
      return;
    }

    if (tool === "wreath") {
      onWreathTap?.([Math.round(point[0]), Math.round(point[1])]);
      return;
    }

    if (tool === "measure") {
      if (measureMode === "line") selectUnder(point);
      else onMeasureTap?.([Math.round(point[0]), Math.round(point[1])]);
      return;
    }

    if (tool === "select") {
      pickUnder(point);
      return;
    }

    if (tool === "click") {
      const hit = areaAt(point);
      if (event.altKey) {
        if (hit) onRemoveAt(hit.id);
      } else {
        onPoint([Math.round(point[0]), Math.round(point[1])]);
      }
      return;
    }

    if (tool === "find") {
      dragRef.current = { mode: "box", start: point };
      boxRef.current = [point[0], point[1], point[0], point[1]];
      requestDraw();
      return;
    }

    const region = activeRegion();
    if (!region) {
      selectUnder(point);
      return;
    }

    if (region.geometry === "area" && (tool === "brush" || tool === "erase")) {
      pushHistory(region.id, "mask");
      dragRef.current = { mode: "stroke", regionId: region.id, erase: tool === "erase" };
      paint(point, tool === "erase");
      requestDraw();
      return;
    }

    if (region.geometry === "line" && tool === "move" && region.polyline && region.polyline.length >= 2) {
      const hit = 12 / scale();
      const points = region.polyline.map((p) => [...p]);
      const last = points.length - 1;
      const end = [0, last].find(
        (i) => Math.hypot(points[i][0] - point[0], points[i][1] - point[1]) < hit * 1.6,
      );
      const onLine =
        end === undefined &&
        points.some(
          (p, i) => i < last && distanceToSegment(point, p, points[i + 1]).distance < hit,
        );
      if (end === undefined && !onLine) {
        selectUnder(point);
        return;
      }

      pushHistory(region.id, "poly");
      const origin = points.map((p) => [...p]);
      if (end !== undefined) {
        const neighbour = points[end === 0 ? 1 : last - 1];
        const away = [points[end][0] - neighbour[0], points[end][1] - neighbour[1]];
        const length = Math.hypot(away[0], away[1]) || 1;
        dragRef.current = {
          mode: "endpoint",
          regionId: region.id,
          index: end,
          origin,
          axis: [away[0] / length, away[1] / length],
          limit: -(length - 4),
        };
      } else {
        dragRef.current = { mode: "translate", regionId: region.id, start: point, origin };
      }
      draftRef.current = { regionId: region.id, polyline: points };
      requestDraw();
      return;
    }

    if (region.geometry === "line" && tool === "vertex" && region.polyline) {
      const hit = 10 / scale();
      const points = region.polyline.map((p) => [...p]);

      const vertex = points.findIndex((p) => Math.hypot(p[0] - point[0], p[1] - point[1]) < hit);
      if (vertex >= 0) {
        if (event.altKey && points.length > 2) {
          pushHistory(region.id, "poly");
          points.splice(vertex, 1);
          onPolylineCommit(region.id, points);
          return;
        }
        pushHistory(region.id, "poly");
        dragRef.current = { mode: "vertex", regionId: region.id, index: vertex };
        draftRef.current = { regionId: region.id, polyline: points };
        return;
      }

      for (let i = 0; i < points.length - 1; i += 1) {
        const { distance, point: projected } = distanceToSegment(point, points[i], points[i + 1]);
        if (distance < hit) {
          pushHistory(region.id, "poly");
          points.splice(i + 1, 0, [Math.round(projected[0]), Math.round(projected[1])]);
          dragRef.current = { mode: "vertex", regionId: region.id, index: i + 1 };
          draftRef.current = { regionId: region.id, polyline: points };
          requestDraw();
          return;
        }
      }
      selectUnder(point);
    }
  };

  const onPointerMove = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const pan = panRef.current;
    if (pan && bedRef.current) {
      bedRef.current.scrollLeft = pan.left - (event.clientX - pan.x);
      bedRef.current.scrollTop = pan.top - (event.clientY - pan.y);
      return;
    }
    if (event.pointerType === "touch" && touches.current.has(event.pointerId)) {
      touches.current.set(event.pointerId, [event.clientX, event.clientY]);
      if (pinchRef.current) {
        movePinch();
        return;
      }
      if (afterPinch.current) return;
      const pending = pendingRef.current;
      if (pending) {
        if (Math.hypot(event.clientX - pending.at[0], event.clientY - pending.at[1]) < TOUCH_SLOP_PX) return;
        firePending();
      }
    }
    const point = toImage(event);
    pointerRef.current = point;
    const drag = dragRef.current;
    // Named on the photo while the pointer rests on it (never mid-drag or while drawing a line).
    const hovering = drag || drawingColor || event.pointerType === "touch" ? null : regionAt(point)?.id ?? null;
    if (hovering !== hoverRef.current) hoverRef.current = hovering;

    if (drag?.mode === "stroke") paint(point, drag.erase);
    else if (drag?.mode === "vertex" && draftRef.current) {
      draftRef.current.polyline[drag.index] = [Math.round(point[0]), Math.round(point[1])];
    } else if (drag?.mode === "translate" && draftRef.current) {
      const dx = point[0] - drag.start[0];
      const dy = point[1] - drag.start[1];
      draftRef.current.polyline = drag.origin.map(([x, y]) => [
        Math.round(x + dx),
        Math.round(y + dy),
      ]);
    } else if (drag?.mode === "endpoint" && draftRef.current) {
      const anchor = drag.origin[drag.index];
      const travel = Math.max(
        (point[0] - anchor[0]) * drag.axis[0] + (point[1] - anchor[1]) * drag.axis[1],
        drag.limit,
      );
      const moved = drag.origin.map((p) => [...p]);
      moved[drag.index] = [
        Math.round(anchor[0] + drag.axis[0] * travel),
        Math.round(anchor[1] + drag.axis[1] * travel),
      ];
      draftRef.current.polyline = moved;
    } else if (drag?.mode === "mid") {
      midDraftRef.current = ontoLine(point);
    } else if (drag?.mode === "box") {
      boxRef.current = [
        Math.min(drag.start[0], point[0]),
        Math.min(drag.start[1], point[1]),
        Math.max(drag.start[0], point[0]),
        Math.max(drag.start[1], point[1]),
      ];
    }
    requestDraw();
  };

  const onPointerUp = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (panRef.current) {
      panRef.current = null;
      event.currentTarget.releasePointerCapture(event.pointerId);
      return;
    }
    if (event.pointerType === "touch") {
      touches.current.delete(event.pointerId);
      if (pinchRef.current || afterPinch.current) {
        if (touches.current.size < 2) pinchRef.current = null;
        if (touches.current.size === 0) afterPinch.current = false;
        return;
      }
      // A quick tap: the held finger acts, then lifts.
      if (pendingRef.current) firePending();
    }
    const drag = dragRef.current;
    dragRef.current = null;
    event.currentTarget.releasePointerCapture(event.pointerId);

    if (drag?.mode === "stroke") void commitMask(drag.regionId);
    else if (
      (drag?.mode === "vertex" || drag?.mode === "translate" || drag?.mode === "endpoint") &&
      draftRef.current
    ) {
      onPolylineCommit(drag.regionId, draftRef.current.polyline);
      draftRef.current = null;
    } else if (drag?.mode === "mid" && midDraftRef.current) {
      onOverrideMid?.(drag.regionId, midDraftRef.current.map(Math.round));
      midDraftRef.current = null;
    } else if (drag?.mode === "box" && boxRef.current) {
      const [x0, y0, x1, y1] = boxRef.current.map(Math.round);
      boxRef.current = null;
      if (x1 - x0 > 8 && y1 - y0 > 8) onBox([x0, y0, x1, y1]);
    }
    requestDraw();
  };

  return (
    <div className="canvas-scroll" ref={bedRef}>
      <canvas
        ref={canvasRef}
        width={backing.width}
        height={backing.height}
        style={{ width: fitWidth ? `${Math.round(fitWidth * zoom)}px` : `${zoom * 100}%` }}
        className={`canvas tool-${tool}${night ? " night" : ""}`}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={(event) => {
          // The browser took the gesture (or the finger left the screen): nothing is applied.
          touches.current.delete(event.pointerId);
          if (pendingRef.current) {
            window.clearTimeout(pendingRef.current.timer);
            pendingRef.current = null;
          }
          if (touches.current.size < 2) pinchRef.current = null;
          if (touches.current.size === 0) afterPinch.current = false;
        }}
        onPointerLeave={() => {
          pointerRef.current = null;
          hoverRef.current = null;
          requestDraw();
        }}
      />
    </div>
  );
});
