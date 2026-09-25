"use client";

import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from "react";
import { d2, nearestOnPolyline, type Region } from "@/lib/design";
import type { MaskStore } from "@/lib/masks";
import type { Point } from "@/lib/types";

export type DTool = "select" | "brush" | "erase" | "box" | "click";
export const ZOOM_STEPS = [1, 1.5, 2, 3, 4];

export interface DesignCanvasHandle { zoomStep(dir: 1 | -1): void; fit(): void }

interface Props {
  image: HTMLImageElement | null;
  width: number; height: number;
  regions: Region[];
  selectedId: string | null;
  tool: DTool;
  brushSize: number;
  masks: MaskStore | null;
  maskVersion: number;
  showObstacles: boolean;
  hidden: Record<string, [number, number][]>;
  drawing: { color: string; points: Point[] } | null;
  locked: boolean;
  busy: boolean;
  onSelect(id: string | null): void;
  onLine(id: string, pts: Point[], commit: boolean): void;
  onStroke(id: string, commit: boolean): void;
  onBox(box: [number, number, number, number]): void;
  onClickObject(p: Point, alt: boolean): void;
  onDrawPoint(p: Point): void;
  onDrawFinish(): void;
  onZoom(z: number): void;
}

type Drag =
  | { kind: "pan"; start: [number, number]; pan: [number, number]; moved: boolean }
  | { kind: "vertex"; id: string; index: number; base: Point[]; moved: boolean }
  | { kind: "translate"; id: string; start: Point; base: Point[]; moved: boolean }
  | { kind: "paint"; id: string; last: Point; erase: boolean }
  | { kind: "box"; start: Point; now: Point }
  | null;

const HIT_V = 10, HIT_L = 8;

const DesignCanvas = forwardRef<DesignCanvasHandle, Props>(function DesignCanvas(p, ref) {
  const { width: W, height: H } = p;
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [box, setBox] = useState({ w: 0, h: 0 });
  const [view, setView] = useState({ zoom: 1, panX: 0, panY: 0 });
  const [cursor, setCursor] = useState<Point | null>(null);
  const [boxRect, setBoxRect] = useState<[Point, Point] | null>(null);
  const drag = useRef<Drag>(null);
  const space = useRef(false);

  // fit the photo inside the frame
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setBox({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    setBox({ w: el.clientWidth, h: el.clientHeight });
    return () => ro.disconnect();
  }, []);
  const base = W && box.w ? Math.min(box.w / W, box.h / H) : 1;       // CSS px per image px at zoom 1
  const k = base * view.zoom;
  const offX = (box.w - W * k) / 2, offY = (box.h - H * k) / 2;       // centred when smaller than the frame

  const clamp = useCallback((v: { zoom: number; panX: number; panY: number }) => {
    const zoom = Math.min(8, Math.max(1, v.zoom));
    const kk = base * zoom, iw = W * kk, ih = H * kk;
    const lim = (pan: number, img: number, frame: number) => (img <= frame ? 0 : Math.min((img - frame) / 2, Math.max(-(img - frame) / 2, pan)));
    return { zoom, panX: lim(v.panX, iw, box.w), panY: lim(v.panY, ih, box.h) };
  }, [base, W, H, box]);

  const zoomTo = useCallback((zoom: number, cx = box.w / 2, cy = box.h / 2) => {
    setView((v) => {
      const z = Math.min(8, Math.max(1, zoom)), f = z / v.zoom;
      // keep the point under (cx, cy) still: centre-relative coordinates
      const rx = cx - box.w / 2 - v.panX, ry = cy - box.h / 2 - v.panY;
      return clamp({ zoom: z, panX: v.panX - rx * (f - 1), panY: v.panY - ry * (f - 1) });
    });
  }, [box, clamp]);

  useImperativeHandle(ref, () => ({
    zoomStep: (dir) => {
      const z = view.zoom;
      const next = dir > 0 ? ZOOM_STEPS.find((s) => s > z + 1e-6) ?? 8 : [...ZOOM_STEPS].reverse().find((s) => s < z - 1e-6) ?? 1;
      zoomTo(next);
    },
    fit: () => setView({ zoom: 1, panX: 0, panY: 0 }),
  }), [view.zoom, zoomTo]);
  useEffect(() => { p.onZoom(view.zoom); }, [view.zoom]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { setView({ zoom: 1, panX: 0, panY: 0 }); }, [p.image]);

  const originX = offX + view.panX, originY = offY + view.panY;
  const toImage = (cx: number, cy: number): Point => {
    const r = canvasRef.current!.getBoundingClientRect();
    return [(cx - r.left - originX) / k, (cy - r.top - originY) / k];
  };

  // ── draw ──
  useEffect(() => {
    const c = canvasRef.current;
    if (!c || !box.w) return;
    const dpr = window.devicePixelRatio || 1;
    if (c.width !== Math.round(box.w * dpr) || c.height !== Math.round(box.h * dpr)) { c.width = Math.round(box.w * dpr); c.height = Math.round(box.h * dpr); }
    const ctx = c.getContext("2d")!;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, c.width, c.height);
    if (!p.image) return;
    ctx.setTransform(dpr * k, 0, 0, dpr * k, dpr * originX, dpr * originY);
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(p.image, 0, 0, W, H);
    const px = 1 / k;


    // areas
    for (const r of p.regions) {
      if (r.geometry !== "area" || !p.masks) continue;
      const sel = r.id === p.selectedId;
      if (r.role === "blocker" && !p.showObstacles && !sel) continue;
      const t = p.masks.tinted(r.id, r.color);
      if (!t) continue;
      ctx.globalAlpha = sel ? 0.6 : r.visible ? (r.role === "blocker" ? 0.3 : 0.42) : 0.14;
      ctx.drawImage(t, 0, 0);
    }
    ctx.globalAlpha = 1;

    // lines
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    const path = (pts: Point[]) => { ctx.beginPath(); pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); };
    const ordered = [...p.regions.filter((r) => r.geometry === "line" && !r.visible), ...p.regions.filter((r) => r.geometry === "line" && r.visible)];
    for (const r of ordered) {
      const pts = r.polyline;
      if (!pts || pts.length < 2) continue;
      const sel = r.id === p.selectedId;
      const lw = (r.visible ? 3.2 : 1.8) * px * (sel ? 1.4 : 1);
      if (sel) { ctx.setLineDash([]); ctx.strokeStyle = "#ffffff"; ctx.lineWidth = lw + 4 * px; path(pts); ctx.stroke(); }
      ctx.setLineDash([]); ctx.strokeStyle = "rgba(0,0,0,.55)"; ctx.lineWidth = lw + 2 * px; path(pts); ctx.stroke();
      // every line is drawn solid, the same way, whatever it passes behind; a line switched off (○) is only fainter
      ctx.setLineDash([]); ctx.strokeStyle = r.color; ctx.lineWidth = lw; ctx.globalAlpha = r.visible ? 1 : 0.45; path(pts); ctx.stroke(); ctx.globalAlpha = 1;
      if (sel && !p.locked) {
        ctx.setLineDash([]);
        pts.forEach(([x, y], i) => {
          const end = i === 0 || i === pts.length - 1;
          ctx.beginPath(); ctx.arc(x, y, (end ? 6 : 5) * px, 0, Math.PI * 2);
          ctx.fillStyle = end ? "#dbe4ff" : "#ffffff"; ctx.fill();
          ctx.strokeStyle = "rgba(0,0,0,.8)"; ctx.lineWidth = 1.5 * px; ctx.stroke();
        });
      }
    }
    // unselected lines: small handles in Points tool, so you can see what can be dragged
    ctx.setLineDash([]);

    // a line being drawn
    if (p.drawing && p.drawing.points.length) {
      const pts = cursor ? [...p.drawing.points, cursor] : p.drawing.points;
      ctx.setLineDash([]); ctx.strokeStyle = p.drawing.color; ctx.lineWidth = 3 * px; path(pts); ctx.stroke();
      for (const [x, y] of p.drawing.points) { ctx.beginPath(); ctx.arc(x, y, 5 * px, 0, Math.PI * 2); ctx.fillStyle = "#fff"; ctx.fill(); ctx.strokeStyle = "#000"; ctx.lineWidth = 1.2 * px; ctx.stroke(); }
    }
    // box being dragged
    if (boxRect) {
      const [[x0, y0], [x1, y1]] = boxRect;
      ctx.setLineDash([6 * px, 4 * px]); ctx.strokeStyle = "#4f7cff"; ctx.lineWidth = 2 * px;
      ctx.strokeRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0)); ctx.setLineDash([]);
      ctx.fillStyle = "rgba(79,124,255,.12)"; ctx.fillRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
    }
    // brush cursor
    if (cursor && (p.tool === "brush" || p.tool === "erase") && !p.locked) {
      ctx.beginPath(); ctx.arc(cursor[0], cursor[1], p.brushSize / 2, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(0,0,0,.7)"; ctx.lineWidth = 2.5 * px; ctx.stroke();
      ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 1.2 * px; ctx.stroke();
    }
  });

  // ── hit tests ──
  function hitLine(pt: Point): { id: string; vertex: number; onBody: boolean } | null {
    const lines = p.regions.filter((r) => r.geometry === "line" && r.polyline && r.polyline.length >= 2);
    const sel = lines.find((r) => r.id === p.selectedId);
    const order = sel ? [sel, ...lines.filter((r) => r !== sel).reverse()] : [...lines].reverse();
    for (const r of order) {
      const pts = r.polyline!;
      const vi = pts.findIndex((q) => d2(q, pt) * k < HIT_V);
      if (vi >= 0) return { id: r.id, vertex: vi, onBody: false };
      if (nearestOnPolyline(pt, pts).d * k < HIT_L) return { id: r.id, vertex: -1, onBody: true };
    }
    return null;
  }
  function hitArea(pt: Point): string | null {
    if (!p.masks) return null;
    const areas = p.regions.filter((r) => r.geometry === "area" && (r.role !== "blocker" || p.showObstacles || r.id === p.selectedId));
    const sel = areas.find((r) => r.id === p.selectedId);
    for (const r of sel ? [sel, ...areas.filter((a) => a !== sel).reverse()] : [...areas].reverse()) if (p.masks.covers(r.id, pt[0], pt[1])) return r.id;
    return null;
  }

  // ── pointer ──
  function startPan(e: React.PointerEvent) {
    drag.current = { kind: "pan", start: [e.clientX, e.clientY], pan: [view.panX, view.panY], moved: false };
  }

  function onDown(e: React.PointerEvent<HTMLCanvasElement>) {
    if (!p.image) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    const pt = toImage(e.clientX, e.clientY);
    if (e.button === 1 || space.current) { startPan(e); return; }
    if (e.button !== 0) return;
    if (p.drawing) { p.onDrawPoint(pt); return; }
    if (p.locked) { const l = hitLine(pt); p.onSelect(l?.id ?? hitArea(pt)); startPan(e); return; }
    const sel = p.regions.find((r) => r.id === p.selectedId);

    if ((p.tool === "brush" || p.tool === "erase") && sel?.geometry === "area") {
      p.masks?.dab(sel.id, pt, pt, p.brushSize / 2, p.tool === "erase");
      drag.current = { kind: "paint", id: sel.id, last: pt, erase: p.tool === "erase" };
      p.onStroke(sel.id, false);
      return;
    }
    if (p.tool === "box") { drag.current = { kind: "box", start: pt, now: pt }; setBoxRect([pt, pt]); return; }
    if (p.tool === "click") { p.onClickObject(pt, e.altKey); return; }

    const line = hitLine(pt);
    if (p.tool === "select" && line) {                                  // one tool: drag a point, or drag the line
      const r = p.regions.find((x) => x.id === line.id)!;
      p.onSelect(r.id);
      const pts = r.polyline!;
      if (line.vertex >= 0) {
        if (e.altKey) { if (pts.length > 2) p.onLine(r.id, pts.filter((_, i) => i !== line.vertex), true); return; }
        drag.current = { kind: "vertex", id: r.id, index: line.vertex, base: pts, moved: false };   // an end point lengthens the line
      } else {
        drag.current = { kind: "translate", id: r.id, start: pt, base: pts, moved: false };
      }
      return;
    }
    const id = line?.id ?? hitArea(pt);
    p.onSelect(id);
    startPan(e);
  }

  function onMove(e: React.PointerEvent<HTMLCanvasElement>) {
    const pt = toImage(e.clientX, e.clientY);
    setCursor(pt);
    const d = drag.current;
    if (!d) return;
    if (d.kind === "pan") {
      const dx = e.clientX - d.start[0], dy = e.clientY - d.start[1];
      if (Math.hypot(dx, dy) > 3) d.moved = true;
      setView((v) => clamp({ ...v, panX: d.pan[0] + dx, panY: d.pan[1] + dy }));
    } else if (d.kind === "paint") {
      p.masks?.dab(d.id, d.last, pt, p.brushSize / 2, d.erase);
      d.last = pt;
      p.onStroke(d.id, false);
    } else if (d.kind === "box") {
      d.now = pt; setBoxRect([d.start, pt]);
    } else if (d.kind === "vertex") {
      d.moved = true;
      const next = d.base.map((q, i) => (i === d.index ? [Math.round(pt[0] * 10) / 10, Math.round(pt[1] * 10) / 10] : q)) as Point[];
      p.onLine(d.id, next, false);
    } else if (d.kind === "translate") {
      d.moved = true;
      const dx = pt[0] - d.start[0], dy = pt[1] - d.start[1];
      p.onLine(d.id, d.base.map(([x, y]) => [Math.round((x + dx) * 10) / 10, Math.round((y + dy) * 10) / 10]) as Point[], false);
    }
  }

  function onUp(e: React.PointerEvent<HTMLCanvasElement>) {
    const d = drag.current;
    drag.current = null;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId);
    if (!d) return;
    if (d.kind === "paint") p.onStroke(d.id, true);
    else if (d.kind === "box") {
      setBoxRect(null);
      const [x0, y0] = d.start, [x1, y1] = d.now;
      if (Math.abs(x1 - x0) * k > 6 && Math.abs(y1 - y0) * k > 6) p.onBox([Math.round(Math.min(x0, x1)), Math.round(Math.min(y0, y1)), Math.round(Math.max(x0, x1)), Math.round(Math.max(y0, y1))]);
    } else if ((d.kind === "vertex" || d.kind === "translate") && d.moved) {
      const r = p.regions.find((x) => x.id === d.id);
      if (r?.polyline) p.onLine(d.id, r.polyline, true);
    }
  }

  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    const wheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = c.getBoundingClientRect();
      zoomTo(view.zoom * Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0022)), e.clientX - r.left, e.clientY - r.top);
    };
    c.addEventListener("wheel", wheel, { passive: false });
    return () => c.removeEventListener("wheel", wheel);
  }, [view.zoom, zoomTo]);

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.code === "Space" && !(e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLSelectElement)) { space.current = true; e.preventDefault(); }
    };
    const up = (e: KeyboardEvent) => { if (e.code === "Space") space.current = false; };
    window.addEventListener("keydown", down); window.addEventListener("keyup", up);
    return () => { window.removeEventListener("keydown", down); window.removeEventListener("keyup", up); };
  }, []);

  const cur = drag.current?.kind === "pan" ? "grabbing"
    : p.drawing ? "crosshair"
    : p.locked ? "default"
    : p.tool === "brush" || p.tool === "erase" ? "none"
    : p.tool === "box" || p.tool === "click" ? "crosshair"
    : "default";

  return (
    <div ref={wrapRef} className={`relative h-full w-full overflow-hidden ${p.busy ? "busy" : ""}`}>
      <canvas ref={canvasRef} className="absolute inset-0 h-full w-full touch-none select-none" style={{ cursor: cur }}
        onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerCancel={onUp}
        onPointerLeave={() => setCursor(null)}
        onDoubleClick={(e) => {
          if (p.drawing) { p.onDrawFinish(); return; }
          if (p.locked || p.tool !== "select") return;
          const pt = toImage(e.clientX, e.clientY), hit = hitLine(pt);       // double-click a line: add a point there
          if (!hit || hit.vertex >= 0) return;
          const r = p.regions.find((x) => x.id === hit.id)!, near = nearestOnPolyline(pt, r.polyline!);
          p.onLine(r.id, [...r.polyline!.slice(0, near.seg + 1), near.point, ...r.polyline!.slice(near.seg + 1)] as Point[], true);
        }}
        onContextMenu={(e) => e.preventDefault()} />
    </div>
  );
});

export default DesignCanvas;

