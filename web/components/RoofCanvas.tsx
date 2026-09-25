"use client";

import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from "react";
import {
  applyOffset, dist, mergeRefs, moveTo, projectOnSegment, refsAt, snap, vertices,
  type Refs, type Selection,
} from "@/lib/editor";
import type { LineDoc, Point, Role, Segment } from "@/lib/types";

export type Tool = "select" | "draw";

export interface RoofCanvasHandle { zoomBy(f: number): void; fit(): void }

interface Props {
  image: HTMLImageElement | null;
  width: number;
  height: number;
  doc: LineDoc | null;
  roleColors: Record<string, string>;
  planeColors: Record<number, string>;
  editing: boolean;
  tool: Tool;
  drawRole: Role;
  selection: Selection;
  points: Point[];                     // point prompts (detection mode only)
  showPromptPlaceholder: boolean;
  busy?: boolean;
  onSelect(sel: Selection): void;
  onChange(doc: LineDoc, commit: boolean): void;
  onAdd(a: Point, b: Point): void;
  onSplit(index: number, at: Point): void;
  onPromptClick(p: Point, hitRadius: number): void;
  onHover(seg: Segment | null): void;
  onZoom?(zoom: number): void;
}

type Drag =
  | { kind: "vertex"; base: LineDoc; refs: Refs; from: Point; moved: boolean; pointerId: number }
  | { kind: "segment"; base: LineDoc; refs: Refs; start: Point; moved: boolean; pointerId: number }
  | { kind: "pan"; start: [number, number]; pan: [number, number]; moved: boolean; pointerId: number }
  | null;

const HANDLE = 6;           // junction handle radius, screen px
const HIT_VERTEX = 10;      // screen px
const HIT_LINE = 8;
const SNAP = 12;

const RoofCanvas = forwardRef<RoofCanvasHandle, Props>(function RoofCanvas(props, ref) {
  const { image, width: W, height: H, doc, editing, tool, selection } = props;
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [cssW, setCssW] = useState(0);
  const [view, setView] = useState({ zoom: 1, panX: 0, panY: 0 });
  const [hover, setHover] = useState<{ kind: "vertex"; point: Point } | { kind: "segment"; index: number } | null>(null);
  const [cursorPt, setCursorPt] = useState<Point | null>(null);
  const [drawStart, setDrawStart] = useState<Point | null>(null);
  const [snapMark, setSnapMark] = useState<{ point: Point; kind: "vertex" | "line" } | null>(null);
  const [tip, setTip] = useState<{ x: number; y: number; text: string[] } | null>(null);
  const drag = useRef<Drag>(null);
  const spaceDown = useRef(false);
  const propsRef = useRef(props);
  propsRef.current = props;

  const cssH = W ? (cssW * H) / W : 0;
  const base = W ? cssW / W : 1;                           // CSS px per image px at zoom 1
  const k = base * view.zoom;                              // CSS px per image px now

  // ── sizing ──
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setCssW(el.clientWidth));
    ro.observe(el);
    setCssW(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  const clampView = useCallback((v: { zoom: number; panX: number; panY: number }) => {
    const zoom = Math.min(16, Math.max(1, v.zoom));
    const minX = cssW - cssW * zoom, minY = cssH - cssH * zoom;
    return { zoom, panX: Math.min(0, Math.max(minX, v.panX)), panY: Math.min(0, Math.max(minY, v.panY)) };
  }, [cssW, cssH]);

  const zoomAt = useCallback((factor: number, cx: number, cy: number) => {
    setView((v) => {
      const zoom = Math.min(16, Math.max(1, v.zoom * factor));
      const f = zoom / v.zoom;
      return clampView({ zoom, panX: cx - (cx - v.panX) * f, panY: cy - (cy - v.panY) * f });
    });
  }, [clampView]);

  useImperativeHandle(ref, () => ({
    zoomBy: (f) => zoomAt(f, cssW / 2, cssH / 2),
    fit: () => setView({ zoom: 1, panX: 0, panY: 0 }),
  }), [zoomAt, cssW, cssH]);

  useEffect(() => { props.onZoom?.(view.zoom); }, [view.zoom]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { setView({ zoom: 1, panX: 0, panY: 0 }); }, [image]);
  useEffect(() => { if (!editing || tool !== "draw") { setDrawStart(null); setSnapMark(null); } }, [editing, tool]);

  // ── drawing ──
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !cssW || !W) return;
    const dpr = window.devicePixelRatio || 1;
    const bw = Math.round(cssW * dpr), bh = Math.round(cssH * dpr);
    if (canvas.width !== bw || canvas.height !== bh) { canvas.width = bw; canvas.height = bh; }
    const ctx = canvas.getContext("2d")!;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, bw, bh);
    ctx.setTransform(dpr * k, 0, 0, dpr * k, dpr * view.panX, dpr * view.panY);
    ctx.imageSmoothingQuality = "high";
    if (image) ctx.drawImage(image, 0, 0, W, H);
    const px = 1 / k;                                       // one screen px in image units
    const lw = Math.max(2.5, 2.5 * Math.min(1.6, cssW / 700)) * px;

    if (doc) {
      for (const p of doc.planes) {
        if (p.polygon.length < 3) continue;
        ctx.beginPath(); p.polygon.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.closePath();
        ctx.fillStyle = (props.planeColors[p.plane] || "#ffffff") + "33"; ctx.fill();
      }
      ctx.lineCap = "round"; ctx.lineJoin = "round";
      doc.segments.forEach((s, i) => {
        const [[x1, y1], [x2, y2]] = s.points;
        const selected = selection?.kind === "segment" && selection.index === i;
        const hot = hover?.kind === "segment" && hover.index === i;
        const w = selected || hot ? lw * 1.7 : lw;
        ctx.setLineDash([]); ctx.strokeStyle = "rgba(0,0,0,.6)"; ctx.lineWidth = w + lw * 1.2;
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
        if (selected) { ctx.strokeStyle = "#ffffff"; ctx.lineWidth = w + lw * 0.9; ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); }
        ctx.setLineDash(s.inferred ? [lw * 3, lw * 3] : []);
        ctx.strokeStyle = hot && !editing ? "#ffffff" : (s.inferred ? props.roleColors.inferred : props.roleColors[s.role]) || s.color || "#39ff14";
        ctx.lineWidth = w;
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
      });
      ctx.setLineDash([]);
      const r = (editing ? HANDLE : Math.max(4, cssW / 190 / 1.4)) * px;
      for (const v of vertices(doc)) {
        const sel = selection?.kind === "vertex" && dist(selection.point, v) < 0.75;
        const hot = hover?.kind === "vertex" && dist(hover.point, v) < 0.75;
        ctx.beginPath(); ctx.arc(v[0], v[1], (hot || sel ? r * 1.35 : r), 0, Math.PI * 2);
        ctx.fillStyle = editing ? (sel ? "#2451C9" : "#ffffff") : "#FF3B30"; ctx.fill();
        ctx.strokeStyle = editing ? (sel || hot ? "#2451C9" : "rgba(0,0,0,.75)") : "rgba(0,0,0,.6)";
        ctx.lineWidth = (editing ? 2 : 1) * px; ctx.stroke();
      }
    }

    if (editing && tool === "draw" && drawStart && cursorPt) {        // rubber band
      ctx.setLineDash([6 * px, 4 * px]); ctx.strokeStyle = props.roleColors[props.drawRole] || "#fff"; ctx.lineWidth = lw;
      ctx.beginPath(); ctx.moveTo(...drawStart); ctx.lineTo(...(snapMark?.point ?? cursorPt)); ctx.stroke(); ctx.setLineDash([]);
      ctx.beginPath(); ctx.arc(drawStart[0], drawStart[1], HANDLE * px, 0, Math.PI * 2); ctx.fillStyle = "#2451C9"; ctx.fill();
    }
    if (editing && snapMark) {
      ctx.strokeStyle = "#2451C9"; ctx.lineWidth = 2 * px;
      const s = 9 * px, [x, y] = snapMark.point;
      if (snapMark.kind === "vertex") { ctx.beginPath(); ctx.arc(x, y, s, 0, Math.PI * 2); ctx.stroke(); }
      else { ctx.strokeRect(x - s * 0.7, y - s * 0.7, s * 1.4, s * 1.4); }
    }

    if (!editing) {                                                    // numbered point prompts
      const marks = props.points.length ? props.points : props.showPromptPlaceholder ? [[Math.round(W / 2), Math.round(H * 0.45)] as Point] : [];
      const r = Math.max(4, cssW / 190 / 1.4) * px;
      ctx.font = `bold ${Math.round(r * 2.6)}px system-ui, sans-serif`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      marks.forEach(([x, y], i) => {
        ctx.fillStyle = "rgba(247,248,245,.92)"; ctx.beginPath(); ctx.arc(x, y, r * 2.2, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = "#2451C9"; ctx.lineWidth = 2 * px; ctx.stroke();
        ctx.fillStyle = "#2451C9"; ctx.fillText(props.points.length ? String(i + 1) : "+", x, y + r * 0.15);
      });
    }
  });

  // ── hit testing ──
  const toImage = useCallback((clientX: number, clientY: number): Point => {
    const rect = canvasRef.current!.getBoundingClientRect();
    return [(clientX - rect.left - view.panX) / k, (clientY - rect.top - view.panY) / k];
  }, [view, k]);

  const hitTest = useCallback((p: Point) => {
    if (!doc) return null;
    let bestV: Point | null = null, dv = HIT_VERTEX / k;
    for (const v of vertices(doc)) { const d = dist(p, v); if (d < dv) { dv = d; bestV = v; } }
    if (bestV) return { kind: "vertex" as const, point: bestV };
    let bestS = -1, ds = HIT_LINE / k;
    doc.segments.forEach((s, i) => { const { d } = projectOnSegment(p, s.points); if (d < ds) { ds = d; bestS = i; } });
    return bestS >= 0 ? { kind: "segment" as const, index: bestS } : null;
  }, [doc, k]);

  function showTip(e: React.PointerEvent, seg: Segment | null) {
    props.onHover(seg);
    if (!seg || drag.current) { setTip(null); return; }
    const rect = wrapRef.current!.getBoundingClientRect();
    setTip({ x: e.clientX - rect.left, y: e.clientY - rect.top, text: [
      `${seg.role}${seg.inferred ? " (inferred)" : ""}${seg.edited ? " · edited" : ""}${seg.planes?.length ? `, plane ${seg.planes.join(" + ")}` : ""}`,
      `pitch as seen ${seg.pitch_apparent || "steep"}, ${seg.angle_deg.toFixed(1)}°`,
      `${seg.length_px.toFixed(0)} px${seg.vp ? `, follows VP ${seg.vp}` : ""}`,
    ] });
  }

  // ── pointer handling ──
  function onPointerDown(e: React.PointerEvent<HTMLCanvasElement>) {
    if (!doc && !(!editing && image)) return;
    const p = toImage(e.clientX, e.clientY);
    const wantsPan = e.button === 1 || spaceDown.current;
    if (wantsPan) {
      e.preventDefault();
      drag.current = { kind: "pan", start: [e.clientX, e.clientY], pan: [view.panX, view.panY], moved: false, pointerId: e.pointerId };
      e.currentTarget.setPointerCapture(e.pointerId);
      return;
    }
    if (e.button !== 0) return;
    if (!editing) {
      if (view.zoom > 1) { drag.current = { kind: "pan", start: [e.clientX, e.clientY], pan: [view.panX, view.panY], moved: false, pointerId: e.pointerId }; e.currentTarget.setPointerCapture(e.pointerId); }
      return;                                                            // prompt clicks happen on pointerup
    }
    if (!doc) return;
    if (tool === "draw") {
      const s = e.shiftKey ? { point: p, kind: null } : snap(doc, p, SNAP / k);
      if (!drawStart) setDrawStart(s.point);
      else if (dist(drawStart, s.point) * k > 4) { props.onAdd(drawStart, s.point); setDrawStart(s.point); }
      return;
    }
    const hit = hitTest(p);
    if (hit?.kind === "vertex") {
      let refs = refsAt(doc, hit.point);
      if (e.altKey) {                                                     // detach: only one line's end moves
        const sel = selection?.kind === "segment" ? refs.ends.find((r) => r.seg === selection.index) : undefined;
        refs = { ends: [sel ?? refs.ends[0]], polys: [] };
      }
      props.onSelect({ kind: "vertex", point: hit.point });
      drag.current = { kind: "vertex", base: doc, refs, from: hit.point, moved: false, pointerId: e.pointerId };
      e.currentTarget.setPointerCapture(e.pointerId);
    } else if (hit?.kind === "segment") {
      const s = doc.segments[hit.index];
      const refs = mergeRefs(refsAt(doc, s.points[0]), refsAt(doc, s.points[1]));
      props.onSelect({ kind: "segment", index: hit.index });
      drag.current = { kind: "segment", base: doc, refs, start: p, moved: false, pointerId: e.pointerId };
      e.currentTarget.setPointerCapture(e.pointerId);
    } else {
      props.onSelect(null);
      drag.current = { kind: "pan", start: [e.clientX, e.clientY], pan: [view.panX, view.panY], moved: false, pointerId: e.pointerId };
      e.currentTarget.setPointerCapture(e.pointerId);
    }
  }

  function onPointerMove(e: React.PointerEvent<HTMLCanvasElement>) {
    const p = toImage(e.clientX, e.clientY);
    setCursorPt(p);
    const d = drag.current;
    if (d?.kind === "pan") {
      const dx = e.clientX - d.start[0], dy = e.clientY - d.start[1];
      if (Math.hypot(dx, dy) > 3) d.moved = true;
      setView((v) => clampView({ ...v, panX: d.pan[0] + dx, panY: d.pan[1] + dy }));
      return;
    }
    if (d?.kind === "vertex" && doc) {
      const s = e.shiftKey ? { point: p, kind: null } : snap(d.base, p, SNAP / k, d.refs);
      if (!d.moved && dist(p, d.from) * k < 2) return;
      d.moved = true;
      setSnapMark(s.kind ? { point: s.point, kind: s.kind } : null);
      props.onChange(moveTo(d.base, d.refs, d.from, s.point), false);
      return;
    }
    if (d?.kind === "segment" && doc) {
      const dx = p[0] - d.start[0], dy = p[1] - d.start[1];
      if (!d.moved && Math.hypot(dx, dy) * k < 2) return;
      d.moved = true;
      props.onChange(applyOffset(d.base, d.refs, dx, dy), false);
      return;
    }
    if (editing && tool === "draw" && doc) {
      const s = e.shiftKey ? { point: p, kind: null } : snap(doc, p, SNAP / k);
      setSnapMark(s.kind ? { point: s.point, kind: s.kind } : null);
    }
    const hit = hitTest(p);
    setHover(hit);
    showTip(e, hit?.kind === "segment" && doc ? doc.segments[hit.index] : null);
  }

  function onPointerUp(e: React.PointerEvent<HTMLCanvasElement>) {
    const d = drag.current;
    drag.current = null;
    setSnapMark(null);
    if (d && e.currentTarget.hasPointerCapture(d.pointerId)) e.currentTarget.releasePointerCapture(d.pointerId);
    if ((d?.kind === "vertex" || d?.kind === "segment") && d.moved && doc) {
      props.onChange(doc, true);
      if (d.kind === "vertex") {
        const moved = doc.segments[d.refs.ends[0].seg]?.points[d.refs.ends[0].end];
        if (moved) props.onSelect({ kind: "vertex", point: moved });
      }
      return;
    }
    if (!editing && e.button === 0 && !(d?.kind === "pan" && d.moved) && image) {
      props.onPromptClick(toImage(e.clientX, e.clientY), 14 / k);
    }
  }

  function onDoubleClick(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!editing || !doc) return;
    if (tool === "draw") { setDrawStart(null); return; }
    const p = toImage(e.clientX, e.clientY);
    const hit = hitTest(p);
    if (hit?.kind === "segment") props.onSplit(hit.index, p);
  }

  function onWheel(e: React.WheelEvent<HTMLCanvasElement>) {
    if (!editing && !e.ctrlKey && !e.metaKey && view.zoom === 1) return;  // let the page scroll (the native listener below blocks it otherwise)
    const rect = canvasRef.current!.getBoundingClientRect();
    zoomAt(Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0022)), e.clientX - rect.left, e.clientY - rect.top);
  }

  // wheel must be non-passive to preventDefault; attach natively
  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    const stop = (e: WheelEvent) => { if (propsRef.current.editing || e.ctrlKey || e.metaKey || view.zoom > 1) e.preventDefault(); };
    c.addEventListener("wheel", stop, { passive: false });
    return () => c.removeEventListener("wheel", stop);
  }, [view.zoom]);

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.code === "Space" && !(e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLSelectElement)) {
        if (!spaceDown.current && propsRef.current.editing) e.preventDefault();
        spaceDown.current = true;
      }
      if (e.key === "Escape") setDrawStart(null);
    };
    const up = (e: KeyboardEvent) => { if (e.code === "Space") spaceDown.current = false; };
    window.addEventListener("keydown", down); window.addEventListener("keyup", up);
    return () => { window.removeEventListener("keydown", down); window.removeEventListener("keyup", up); };
  }, []);

  const cursor = drag.current?.kind === "pan" ? "grabbing"
    : !editing ? (view.zoom > 1 ? "grab" : "crosshair")
    : tool === "draw" ? "crosshair"
    : hover?.kind === "vertex" ? "move" : hover?.kind === "segment" ? "grab" : view.zoom > 1 ? "grab" : "default";

  return (
    <div ref={wrapRef} className={`frame ${!image ? "empty" : ""} ${props.busy ? "busy" : ""}`} style={image ? { aspectRatio: `${W} / ${H}` } : undefined}>
      <canvas
        ref={canvasRef}
        className={image ? "block w-full h-full touch-none select-none" : "hidden"}
        style={{ cursor }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onPointerLeave={() => { setHover(null); setTip(null); props.onHover(null); setCursorPt(null); }}
        onDoubleClick={onDoubleClick}
        onWheel={onWheel}
        onContextMenu={(e) => { if (editing && drawStart) { e.preventDefault(); setDrawStart(null); } }}
      />
      {tip && <div className="tooltip" style={{ left: tip.x, top: tip.y }}>{tip.text.map((t, i) => <div key={i}>{t}</div>)}</div>}
      {view.zoom > 1 && <div className="absolute left-2 bottom-2 rounded bg-black/60 text-white text-xs px-2 py-0.5 pointer-events-none">{Math.round(view.zoom * 100)}%</div>}
    </div>
  );
});

export default RoofCanvas;
