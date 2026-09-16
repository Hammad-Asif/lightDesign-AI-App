import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
  type PointerEvent as ReactPointerEvent,
} from "react";

import { fileUrl } from "../api";
import type { Photo, Region, Tool } from "../types";
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

export interface MaskCanvasHandle {
  undo: () => void;
  redo: () => void;
}

interface Props {
  photo: Photo;
  regions: Region[];
  activeId: string | null;
  tool: Tool;
  brushSize: number;
  zoom: number;
  onMaskCommit: (regionId: string, blob: Blob) => Promise<Region>;
  onPolylineCommit: (regionId: string, polyline: number[][]) => void;
  onBox: (box: number[]) => void;
  onPoint: (point: number[]) => void;
  onRemoveAt: (regionId: string) => void;
  onHistoryChange: (state: { canUndo: boolean; canRedo: boolean }) => void;
  drawingColor: string | null;
  onDrawFinish: (polyline: number[][]) => void;
  onDrawCancel: () => void;
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
  | null;

export const MaskCanvas = forwardRef<MaskCanvasHandle, Props>(function MaskCanvas(
  {
    photo,
    regions,
    activeId,
    tool,
    brushSize,
    zoom,
    onMaskCommit,
    onPolylineCommit,
    onBox,
    onPoint,
    onRemoveAt,
    onHistoryChange,
    drawingColor,
    onDrawFinish,
    onDrawCancel,
  },
  ref,
) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
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

  regionsRef.current = regions;
  activeRef.current = activeId;

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
    context.drawImage(image, 0, 0, canvas.width, canvas.height);

    for (const region of regionsRef.current) {
      if (!region.visible) continue;
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

      context.strokeStyle = region.color;
      context.lineWidth = (isActive ? 4.5 : 2.5) / factor;
      trace(points);
      context.stroke();
      context.setLineDash([]);

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

    if (boxRef.current) {
      const [x0, y0, x1, y1] = boxRef.current;
      context.strokeStyle = "#f472b6";
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

    if (pointerRef.current && (tool === "brush" || tool === "erase")) {
      context.beginPath();
      context.arc(pointerRef.current[0], pointerRef.current[1], brushSize / 2, 0, Math.PI * 2);
      context.strokeStyle = tool === "erase" ? "#f87171" : "#ffffff";
      context.lineWidth = 1.5 / factor;
      context.stroke();
    }
  }, [brushSize, tool, drawingColor]);

  const requestDraw = useCallback(() => {
    if (frameRef.current) return;
    frameRef.current = requestAnimationFrame(() => {
      frameRef.current = 0;
      draw();
    });
  }, [draw]);

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

  useEffect(requestDraw, [activeId, zoom, requestDraw]);

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

  const onPointerDown = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const point = toImage(event);
    pointerRef.current = point;
    event.currentTarget.setPointerCapture(event.pointerId);

    if (drawingColor) {
      drawRef.current = [...drawRef.current, [Math.round(point[0]), Math.round(point[1])]];
      requestDraw();
      return;
    }

    if (tool === "click") {
      const hit = [...regionsRef.current]
        .reverse()
        .find((region) => {
          if (!region.visible || region.geometry !== "area") return false;
          const layer = layers.current.get(region.id);
          if (!layer) return false;
          const x = Math.round(point[0]);
          const y = Math.round(point[1]);
          if (x < 0 || y < 0 || x >= layer.width || y >= layer.height) return false;
          return ctx2d(layer).getImageData(x, y, 1, 1).data[3] > 8;
        });
      if (event.altKey) {
        if (hit) onRemoveAt(hit.id);
      } else {
        onPoint([Math.round(point[0]), Math.round(point[1])]);
      }
      return;
    }

    if (tool === "box") {
      dragRef.current = { mode: "box", start: point };
      boxRef.current = [point[0], point[1], point[0], point[1]];
      requestDraw();
      return;
    }

    const region = activeRegion();
    if (!region) return;

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
      if (end === undefined && !onLine) return;

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
    } else if (drag?.mode === "box" && boxRef.current) {
      const [x0, y0, x1, y1] = boxRef.current.map(Math.round);
      boxRef.current = null;
      if (x1 - x0 > 8 && y1 - y0 > 8) onBox([x0, y0, x1, y1]);
    }
    requestDraw();
  };

  return (
    <div className="canvas-scroll">
      <canvas
        ref={canvasRef}
        width={photo.width}
        height={photo.height}
        style={{ width: `${zoom * 100}%` }}
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
