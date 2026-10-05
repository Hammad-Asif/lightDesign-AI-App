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
import { LOW_CONFIDENCE, type LineGeometry, type Photo, type Region, type RegionPlacement, type Tool } from "../types";
import {
  createLayer,
  ctx2d,
  distanceToSegment,
  layerToBlob,
  loadImage,
  paintMaskInto,
  slicePolyline,
} from "./layers";

const HISTORY_LIMIT = 20;
// Halo under a roof line whose edge type could not be identified.
const UNKNOWN_HALO = "rgba(244, 63, 94, 0.6)";
// Debug overlay: which source a line at a pipeline stage came from.
const DEBUG_COLOURS: Record<string, string> = {
  v3: "#22d3ee",
  partition: "#f59e0b",
};

export interface MaskCanvasHandle {
  undo: () => void;
  redo: () => void;
  // Drops the undo steps of regions that are gone, such as the roof lines a House tap redrew.
  forget: (regionIds: string[]) => void;
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
  onSelect: (regionId: string) => void;
  onBox: (box: number[]) => void;
  onPoint: (point: number[]) => void;
  // The House tool: a tap on a roof plane moves it into or out of the house being quoted.
  onHouseTap: (point: number[]) => void;
  onRemoveAt: (regionId: string) => void;
  onHistoryChange: (state: { canUndo: boolean; canRedo: boolean }) => void;
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
}

type Snapshot =
  | { type: "mask"; regionId: string; data: string }
  | { type: "poly"; regionId: string; data: number[][] };

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
    onBox,
    onPoint,
    onHouseTap,
    onRemoveAt,
    onHistoryChange,
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

  const past = useRef<Snapshot[]>([]);
  const future = useRef<Snapshot[]>([]);

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

  const quietObstacle = (region: Region) =>
    region.role === "blocker" && !obstaclesRef.current && region.id !== activeRef.current;

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

  const scale = () => {
    const canvas = canvasRef.current;
    if (!canvas) return 1;
    const rect = canvas.getBoundingClientRect();
    return rect.width / canvas.width || 1;
  };

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    if (!canvas || !image) return;

    const context = ctx2d(canvas);
    const factor = scale();
    context.clearRect(0, 0, canvas.width, canvas.height);
    if (backdrop && backdropRef.current) {
      context.drawImage(backdropRef.current, 0, 0, canvas.width, canvas.height);
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
    context.drawImage(image, 0, 0, canvas.width, canvas.height);

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

    // The bulbs: garland bands under their lights, icicle drops as thin wires, then each bulb at
    // its placed size and colour. Hidden ones (behind an object) are counted, never drawn.
    for (const placed of Object.values(bulbsRef.current ?? {})) {
      if (placed.ring) {
        const [x, y, diameter] = placed.ring;
        context.strokeStyle = "rgba(34, 92, 44, 0.9)";
        context.lineWidth = diameter * 0.16;
        context.beginPath();
        context.arc(x, y, diameter * 0.42, 0, Math.PI * 2);
        context.stroke();
      }
      if (placed.band && placed.band.length > 1) {
        context.strokeStyle = "rgba(34, 92, 44, 0.55)";
        context.lineCap = "round";
        context.lineWidth = placed.band.reduce((sum, b) => sum + b[2], 0) / placed.band.length;
        context.beginPath();
        placed.band.forEach((b, i) => (i ? context.lineTo(b[0], b[1]) : context.moveTo(b[0], b[1])));
        context.stroke();
      }
      context.strokeStyle = "rgba(255, 255, 255, 0.55)";
      context.lineWidth = 1 / factor;
      for (const [x0, y0, x1, y1] of placed.drops ?? []) {
        context.beginPath();
        context.moveTo(x0, y0);
        context.lineTo(x1, y1);
        context.stroke();
      }
      for (const [x, y, diameter, colour, hidden] of placed.bulbs ?? []) {
        if (hidden) continue;
        context.beginPath();
        context.arc(x, y, Math.max(1.2 / factor, diameter / 2), 0, Math.PI * 2);
        context.fillStyle = colour;
        context.fill();
        context.strokeStyle = "rgba(0, 0, 0, 0.55)";
        context.lineWidth = 0.8 / factor;
        context.stroke();
      }
    }

    drawLengths(context, factor);

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

  useEffect(requestDraw, [activeId, zoom, fitWidth, showObstacles, measurePoints, photo.scale, overrideLine, bulbs, geometry, showLengths, requestDraw]);

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

  const reportHistory = () =>
    onHistoryChange({ canUndo: past.current.length > 0, canRedo: future.current.length > 0 });

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
    if (!snapshot) return;
    past.current.push(snapshot);
    if (past.current.length > HISTORY_LIMIT) past.current.shift();
    future.current = [];
    reportHistory();
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
    undo: () => {
      const snapshot = past.current.pop();
      if (!snapshot) return;
      const current = capture(snapshot.regionId, snapshot.type);
      if (current) future.current.push(current);
      reportHistory();
      void restore(snapshot);
    },
    redo: () => {
      const snapshot = future.current.pop();
      if (!snapshot) return;
      const current = capture(snapshot.regionId, snapshot.type);
      if (current) past.current.push(current);
      reportHistory();
      void restore(snapshot);
    },
    forget: (regionIds) => {
      const gone = new Set(regionIds);
      past.current = past.current.filter((snapshot) => !gone.has(snapshot.regionId));
      future.current = future.current.filter((snapshot) => !gone.has(snapshot.regionId));
      reportHistory();
    },
  }));

  const toImage = (event: ReactPointerEvent): number[] => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    const factor = canvas.width / rect.width;
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

  const onPointerDown = (event: ReactPointerEvent<HTMLCanvasElement>) => {
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
      selectUnder(point);
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

    if (tool === "box" || tool === "find") {
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
    const point = toImage(event);
    pointerRef.current = point;
    const drag = dragRef.current;

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
        width={photo.width}
        height={photo.height}
        style={{ width: fitWidth ? `${Math.round(fitWidth * zoom)}px` : `${zoom * 100}%` }}
        className={`canvas tool-${tool}`}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={() => {
          pointerRef.current = null;
          requestDraw();
        }}
      />
    </div>
  );
});
