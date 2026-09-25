"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import AppHeader, { useUser } from "@/components/AppHeader";
import DesignCanvas, { type DesignCanvasHandle, type DTool } from "@/components/DesignCanvas";
import { LightingPanel, RegionPanel } from "@/components/DesignPanels";
import { api, isSessionLost, post } from "@/lib/api";
import {
  EDGE_LABEL, BLOCKER_KINDS, defaultLabel, kindGeometry, nearestOnPolyline, kindLabel, lightingStatus, newId, regionColour,
  type Catalogue, type Design, type Region, type Target,
} from "@/lib/design";
import { hidden as hiddenAlong, isBlocker } from "@/lib/completion";
import { MaskStore } from "@/lib/masks";
import type { Point, Project, Role, SessionInfo } from "@/lib/types";

interface DetectResult {
  lines: { kind: string; edge: Role; polyline: Point[]; visible: boolean; planes: number[]; inferred: boolean; length_px: number; pitch_apparent: string | null }[];
  areas: { kind: string; label?: string; mask: string; visible: boolean; score: number | null; role: "target" | "blocker" }[];
  notes: string[]; elapsed_ms: number; support_mask: string | null;
}

const typing = (t: EventTarget | null) => t instanceof HTMLInputElement || t instanceof HTMLTextAreaElement || t instanceof HTMLSelectElement;
const loadImg = (src: string) => new Promise<HTMLImageElement>((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = src; });

export default function DesignEditor() {
  const user = useUser();
  const router = useRouter();
  const query = useSearchParams();
  const canvasRef = useRef<DesignCanvasHandle>(null);

  const [cat, setCat] = useState<Catalogue | null>(null);
  const [project, setProject] = useState<{ id: string; name: string } | null>(null);
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [design, setDesign] = useState<Design | null>(null);
  const [past, setPast] = useState<Region[][]>([]);
  const [future, setFuture] = useState<Region[][]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [tool, setTool] = useState<DTool>("select");
  const [brushSize, setBrushSize] = useState(40);
  const [zoom, setZoom] = useState(1);
  const masksRef = useRef<MaskStore | null>(null);
  const [maskVersion, setMaskVersion] = useState(0);          // repaint while painting
  const [maskCommitted, setMaskCommitted] = useState(0);      // recompute what depends on mask pixels
  const [showObstacles, setShowObstacles] = useState(true);
  const [tab, setTab] = useState<"lines" | "lighting">("lines");
  const [busy, setBusy] = useState("");
  const [message, setMessage] = useState<{ text: string; error?: boolean } | null>(null);
  const [drawing, setDrawing] = useState<{ kind: string; edge?: Role; color: string; points: Point[] } | null>(null);
  const [saveState, setSaveState] = useState<"saved" | "saving" | "unsaved" | "error">("saved");
  const dragBase = useRef<Region[] | null>(null);
  const loaded = useRef(false);

  const designRef = useRef(design);
  designRef.current = design;
  const sessionRef = useRef(session);
  sessionRef.current = session ?? sessionRef.current;
  const projectRef = useRef(project);
  projectRef.current = project ?? projectRef.current;
  /** Run a call against the working session; if the backend restarted and forgot it, reopen it from the project once. */
  async function withSession<T>(fn: (sid: string) => Promise<T>, sid = sessionRef.current?.session_id ?? ""): Promise<T> {
    try { return await fn(sid); }
    catch (e) {
      const pid = projectRef.current?.id;
      if (!isSessionLost(e) || !pid) throw e;
      const info = await post<SessionInfo>(`/api/projects/${pid}/session`);
      sessionRef.current = info; setSession(info);
      return fn(info.session_id);
    }
  }
  const catRef = useRef(cat);                                   // detection can run before the catalogue state has rendered
  catRef.current = cat;
  const locked = !!design?.approved;
  const say = (text: string, error = false) => setMessage({ text, error });

  // ───────────── load ─────────────
  useEffect(() => {
    if (!user || loaded.current) return;
    const id = query.get("project");
    if (!id) { router.replace("/new"); return; }
    loaded.current = true;
    (async () => {
      try {
        setBusy("Opening…");
        const [c, p] = await Promise.all([api<Catalogue>("/api/design/catalogue"), api<Project>(`/api/projects/${id}`)]);
        setCat(c); catRef.current = c;
        let d = (p.data as unknown as { design?: Design }).design;
        if (!d) {                                    // an older project saved from the (now hidden) detection studio
          const converted = await post<Project>(`/api/projects/${id}/design`);
          d = (converted.data as unknown as { design?: Design }).design;
          if (!d) throw new Error("This project has no lines to open.");
        }
        const info = await post<SessionInfo>(`/api/projects/${id}/session`);
        const img = await loadImg(info.image_url);
        masksRef.current = new MaskStore(info.width, info.height);
        await masksRef.current.sync(d.regions);
        await masksRef.current.setExtra("__support", d.support_mask);
        projectRef.current = { id: p.id, name: p.name }; sessionRef.current = info;
        setProject({ id: p.id, name: p.name }); setSession(info); setImage(img); setDesign(d);
        setMaskCommitted((v) => v + 1);
        setBusy("");
        if (d.mode === "automatic" && !d.detected) await detect(false, info, d);
      } catch (e) { setBusy(""); say(`Could not open: ${(e as Error).message}`, true); }
    })();
  }, [user]); // eslint-disable-line react-hooks/exhaustive-deps

  // ───────────── autosave ─────────────
  const firstSave = useRef(true);
  useEffect(() => {
    if (!design || !project) return;
    if (firstSave.current) { firstSave.current = false; return; }
    setSaveState("unsaved");
    const t = setTimeout(async () => {
      setSaveState("saving");
      try { await api(`/api/projects/${project.id}`, { method: "PUT", json: { data: { design: designRef.current } } }); setSaveState("saved"); }
      catch { setSaveState("error"); }
    }, 800);
    return () => clearTimeout(t);
  }, [design, project]);
  useEffect(() => {
    const warn = (e: BeforeUnloadEvent) => { if (saveState !== "saved") e.preventDefault(); };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [saveState]);

  // ───────────── history ─────────────
  const setRegions = useCallback((next: Region[], before: Region[] | null) => {
    if (before) { setPast((p) => [...p.slice(-80), before]); setFuture([]); }
    setDesign((d) => (d ? { ...d, regions: next } : d));
  }, []);
  const commit = useCallback((next: Region[]) => setRegions(next, designRef.current!.regions), [setRegions]);
  const patchDesign = useCallback((patch: Partial<Design>) => setDesign((d) => (d ? { ...d, ...patch } : d)), []);

  async function restore(regions: Region[]) {
    setDesign((d) => (d ? { ...d, regions } : d));
    if (await masksRef.current?.sync(regions)) { setMaskVersion((v) => v + 1); setMaskCommitted((v) => v + 1); }
    if (selectedId && !regions.some((r) => r.id === selectedId)) setSelectedId(null);
  }
  function undo() {
    if (!past.length || !design) return;
    setFuture((f) => [design.regions, ...f]); setPast((p) => p.slice(0, -1));
    restore(past[past.length - 1]);
  }
  function redo() {
    if (!future.length || !design) return;
    setPast((p) => [...p, design.regions]); setFuture((f) => f.slice(1));
    restore(future[0]);
  }

  // ───────────── detection ─────────────
  async function detect(all: boolean, s = session, d = designRef.current) {
    if (!s || !d) return;
    setBusy(all ? "Detecting everything…" : "Detecting…"); setMessage(null);
    try {
      const r = await withSession((sid) => post<DetectResult>("/api/design/detect", { session_id: sid, targets: d.targets, all_kinds: all }), s.session_id);
      const cur = designRef.current ?? d;
      const cat = catRef.current;
      const kept = cur.regions.filter((x) => x.source !== "auto");                // what you drew or refined stays
      const made: Region[] = [];
      for (const l of r.lines) {
        made.push({ id: newId(), kind: "roof", label: defaultLabel(cat, [...kept, ...made], "roof", l.edge), geometry: "line", role: "target", source: "auto",
                    color: regionColour(cat, "roof", l.edge), visible: l.visible, edge: l.edge, polyline: l.polyline, length_px: l.length_px,
                    pitch_apparent: l.pitch_apparent, planes: l.planes, inferred: l.inferred });
      }
      for (const a of r.areas) {
        made.push({ id: newId(), kind: a.kind, label: a.label ?? kindLabel(cat, a.kind), geometry: "area",
                    role: a.role, source: "auto", color: regionColour(cat, a.kind), visible: true, mask: a.mask, score: a.score });
      }
      await masksRef.current?.sync([...made, ...kept]);
      await masksRef.current?.setExtra("__support", r.support_mask);
      const regions = [...made, ...kept];
      setPast((p) => [...p.slice(-80), cur.regions]); setFuture([]);
      setDesign({ ...cur, regions, detected: true, notes: r.notes, support_mask: r.support_mask ?? cur.support_mask ?? null });
      setMaskVersion((v) => v + 1); setMaskCommitted((v) => v + 1);
      const lit = made.filter((x) => x.geometry === "line" && x.visible).length;
      say(`Found ${lit} line${lit === 1 ? "" : "s"} to light and ${r.areas.length} area${r.areas.length === 1 ? "" : "s"} in ${(r.elapsed_ms / 1000).toFixed(1)} s.`);
    } catch (e) { say(`Detection failed: ${(e as Error).message}`, true); }
    finally { setBusy(""); }
  }

  // ───────────── SAM prompts ─────────────
  /** Find in box with no area selected: roof the detection missed, as roofline (SAM 3 box prompt + the pipeline). */
  async function roofInBox(box: number[]) {
    const d = designRef.current;
    if (!d) return;
    setBusy("Looking for roof in the box…"); setMessage(null);
    try {
      const r = await withSession((sid) => post<{ lines: DetectResult["lines"]; support_mask: string | null }>("/api/design/roof-in-box",
        { session_id: sid, box }));
      const cur = designRef.current ?? d, cat = catRef.current;
      const existing = cur.regions.filter((x) => x.geometry === "line" && x.polyline && x.polyline.length >= 2);
      const retraces = (pts: Point[]) => {                                     // most of it within 6 px of a line we have
        let n = 0, near = 0;
        for (let i = 0; i < pts.length - 1; i++) {
          const steps = Math.max(1, Math.round(Math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) / 5));
          for (let k = 0; k <= steps; k++) {
            const t = k / steps, q: Point = [pts[i][0] + (pts[i + 1][0] - pts[i][0]) * t, pts[i][1] + (pts[i + 1][1] - pts[i][1]) * t];
            n++; if (existing.some((e) => nearestOnPolyline(q, e.polyline!).d <= 6)) near++;
          }
        }
        return n > 0 && near / n >= 0.7;
      };
      const made: Region[] = [];
      for (const l of r.lines) {
        if (retraces(l.polyline)) continue;
        made.push({ id: newId(), kind: "roof", label: defaultLabel(cat, [...cur.regions, ...made], "roof", l.edge), geometry: "line", role: "target",
                    source: "refined", color: regionColour(cat, "roof", l.edge), visible: l.visible, edge: l.edge, polyline: l.polyline,
                    length_px: l.length_px, pitch_apparent: l.pitch_apparent, planes: l.planes, inferred: l.inferred });
      }
      if (!made.length) { say("The roof in that box is already covered by your lines.", false); return; }
      await masksRef.current?.setExtra("__support", r.support_mask);
      commit([...cur.regions, ...made]);
      setDesign((x) => (x ? { ...x, support_mask: r.support_mask ?? x.support_mask ?? null } : x));
      setSelectedId(made[0].id); setTool("select");
      say(`Added ${made.length} roof line${made.length === 1 ? "" : "s"} from the box.`);
    } catch (e) { say((e as Error).message, true); }
    finally { setBusy(""); }
  }

  async function refine(body: { point?: number[]; box?: number[] }) {
    if (!session || !design) return;
    setBusy("Asking SAM 3…");
    try {
      const r = await withSession((sid) => post<{ mask: string; score: number | null }>("/api/design/refine", { session_id: sid, ...body }));
      const sel = design.regions.find((x) => x.id === selectedId && x.geometry === "area");
      const store = masksRef.current!;
      if (sel) {
        await store.merge(sel.id, r.mask);
        const mask = store.export(sel.id);
        const next = design.regions.map((x) => (x.id === sel.id ? { ...x, mask } : x));
        commit(next);
      } else {
        const id = newId();
        const region: Region = { id, kind: "custom_area", label: defaultLabel(cat, design.regions, "custom_area"), geometry: "area", role: "target",
          source: "refined", color: regionColour(cat, "custom_area"), visible: true, mask: r.mask, score: r.score };
        const regions = [...design.regions, region];
        await store.sync(regions);
        commit(regions);
        setSelectedId(id);
      }
      setMaskVersion((v) => v + 1); setMaskCommitted((v) => v + 1);
      setTool("brush");
    } catch (e) { say((e as Error).message, true); }
    finally { setBusy(""); }
  }

  function onClickObject(pt: Point, alt: boolean) {
    if (!design) return;
    if (alt) {
      const hit = [...design.regions].reverse().find((r) => r.geometry === "area" && masksRef.current?.covers(r.id, pt[0], pt[1]));
      if (hit) { deleteRegion(hit.id); say(`Removed ${hit.label}.`); }
      return;
    }
    refine({ point: [Math.round(pt[0]), Math.round(pt[1])] });
  }

  // ───────────── canvas edits ─────────────
  function onLine(id: string, pts: Point[], isCommit: boolean) {
    const d = designRef.current;
    if (!d) return;
    if (!isCommit) {
      if (!dragBase.current) dragBase.current = d.regions;
      setDesign({ ...d, regions: d.regions.map((r) => (r.id === id ? { ...r, polyline: pts } : r)) });
      return;
    }
    const before = dragBase.current ?? d.regions;
    dragBase.current = null;
    const next = d.regions.map((r) => (r.id === id ? { ...r, polyline: pts, source: r.source === "auto" ? ("auto" as const) : r.source } : r));
    setRegions(next, before);
  }

  function onStroke(id: string, isCommit: boolean) {
    setMaskVersion((v) => v + 1);
    if (!isCommit) return;
    const d = designRef.current!;
    const mask = masksRef.current!.export(id);
    const next = d.regions.map((r) => (r.id === id ? { ...r, mask } : r));
    commit(next);
    setMaskCommitted((v) => v + 1);
  }

  function updateRegion(id: string, patch: Partial<Region>) {
    const d = designRef.current!;
    const next = d.regions.map((r) => (r.id === id ? { ...r, ...patch } : r));
    commit(next);
    if ("role" in patch || "visible" in patch || "kind" in patch) setMaskCommitted((v) => v + 1);
  }
  function deleteRegion(id: string) {
    const d = designRef.current!;
    commit(d.regions.filter((r) => r.id !== id));
    if (selectedId === id) setSelectedId(null);
    setMaskCommitted((v) => v + 1);
  }

  function addItem(kind: string, edge?: Role) {
    if (!design) return;
    if (kindGeometry(cat, kind) === "line") {
      setDrawing({ kind, edge, color: regionColour(cat, kind, edge), points: [] });
      setSelectedId(null);
      say(`Click along the photo to draw the ${kind === "roof" && edge ? EDGE_LABEL[edge].toLowerCase() : kindLabel(cat, kind).toLowerCase()}. Double-click or Enter to finish, Esc to cancel.`);
      return;
    }
    const id = newId();
    const region: Region = { id, kind, label: defaultLabel(cat, design.regions, kind), geometry: "area",
      role: kind === "obstruction" ? "blocker" : "target", source: "manual", color: regionColour(cat, kind), visible: true, mask: null };
    commit([...design.regions, region]);
    masksRef.current?.entry(id);
    setSelectedId(id); setTool("brush");
    say(`Paint the ${region.label.toLowerCase()} with the brush, or drag a Box / use Click object to let SAM find it.`);
  }

  function drawPoint(pt: Point) {
    setDrawing((dr) => {
      if (!dr) return dr;
      const last = dr.points[dr.points.length - 1];
      if (last && Math.hypot(last[0] - pt[0], last[1] - pt[1]) < 3) return dr;         // the second click of a double-click
      return { ...dr, points: [...dr.points, [Math.round(pt[0] * 10) / 10, Math.round(pt[1] * 10) / 10]] };
    });
  }
  function finishDrawing() {
    const dr = drawing, d = designRef.current;
    setDrawing(null);
    if (!dr || !d) return;
    if (dr.points.length < 2) { say("A line needs at least two points."); return; }
    const id = newId();
    const region: Region = { id, kind: dr.kind, label: defaultLabel(cat, d.regions, dr.kind, dr.edge ?? null), geometry: "line", role: "target",
      source: "manual", color: dr.color, visible: true, edge: dr.kind === "roof" ? (dr.edge ?? null) : null, polyline: dr.points, side: "front" };
    commit([...d.regions, region]);
    setSelectedId(id); setTool("select");
    setMessage(null);
  }

  // ───────────── lighting / approve ─────────────
  function setTargets(targets: Target[]) { patchDesign({ targets, lighting_confirmed: false }); }
  function confirmLighting() { patchDesign({ lighting_confirmed: true }); say("Lighting confirmed."); }
  function approve() {
    if (!design?.regions.length) { say("Add at least one line or area before approving.", true); return; }
    patchDesign({ approved: true }); setDrawing(null); setTool("select"); say("Approved. The lines and areas are locked.");
  }

  // ───────────── derived ─────────────
  const hidden = useMemo(() => {
    const out: Record<string, [number, number][]> = {}, share: Record<string, number> = {};
    const store = masksRef.current;
    if (!design || !store) return { runs: out, share };
    const blockers = design.regions.filter(isBlocker);
    if (!blockers.length || !session) return { runs: out, share };
    const covered = (x: number, y: number) => blockers.some((b) => store.covers(b.id, x, y));
    for (const r of design.regions) {
      if (r.geometry !== "line" || !r.visible || !r.polyline || r.polyline.length < 2) continue;
      const h = hiddenAlong(r.polyline, covered, session.width);
      out[r.id] = h.runs; share[r.id] = h.share;
    }
    return { runs: out, share };
  }, [design, maskCommitted]); // eslint-disable-line react-hooks/exhaustive-deps

  const selected = design?.regions.find((r) => r.id === selectedId) ?? null;
  const areaSelected = selected?.geometry === "area";

  // ───────────── keyboard ─────────────
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (typing(e.target) || !design) return;
      const mod = e.metaKey || e.ctrlKey, k = e.key.toLowerCase();
      if (mod && k === "z") { e.preventDefault(); if (!locked) { if (e.shiftKey) redo(); else undo(); } return; }
      if (mod) return;
      if (drawing) {
        if (k === "enter") { e.preventDefault(); finishDrawing(); }
        else if (k === "escape") { setDrawing(null); setMessage(null); }
        else if (k === "backspace") setDrawing((dr) => (dr ? { ...dr, points: dr.points.slice(0, -1) } : dr));
        return;
      }
      if (k === "=" || k === "+") canvasRef.current?.zoomStep(1);
      else if (k === "-") canvasRef.current?.zoomStep(-1);
      else if (k === "0") canvasRef.current?.fit();
      if (locked) return;
      if (k === "s") setTool("select");
      else if (k === "b" && areaSelected) setTool("brush");
      else if (k === "e" && areaSelected) setTool("erase");
      else if (k === "[") setBrushSize((s) => Math.max(4, s - 6));
      else if (k === "]") setBrushSize((s) => Math.min(200, s + 6));
      else if ((k === "delete" || k === "backspace") && selectedId) { e.preventDefault(); deleteRegion(selectedId); }
      else if (k === "escape") setSelectedId(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  // ───────────── render ─────────────
  const T = (t: DTool, label: string, opts: { disabled?: boolean; title?: string } = {}) => (
    <button type="button" className="tool-btn" aria-pressed={tool === t} disabled={opts.disabled} title={opts.title} onClick={() => setTool(t)}>{label}</button>
  );
  const hint = locked ? "Reopen for correction to change lines and areas."
    : drawing ? "Click along the photo to place points. Double-click or Enter finishes, Backspace removes the last point, Esc cancels."
    : tool === "brush" ? `Painting ${selected?.label ?? "the area"}. [ and ] change the brush size.`
    : tool === "erase" ? `Erasing from ${selected?.label ?? "the area"}. [ and ] change the brush size.`
    : tool === "box" ? (areaSelected ? `Drag a box around an object: SAM 3 finds it and adds it to ${selected!.label}.`
        : "Drag a box around roof the detection missed: SAM 3 finds it and adds its roofline. (Select an area first to add to that area instead.)")
    : tool === "click" ? `Click an object: SAM 3 finds it and adds it to ${areaSelected ? selected!.label : "a new area"}. Alt-click an area to remove it.`
    : selected?.geometry === "line" ? "Drag a point to move it (an end point lengthens the line) · drag the line to move it · double-click the line to add a point · Alt-click a point to remove it · Delete removes the line."
    : areaSelected ? "Use Brush or Erase to shape this area, or Find in box / Find by click to let SAM 3 add to it. Delete removes it."
    : "Click a line or an area to edit it. Scroll to zoom, hold Space and drag to pan. Add lines and areas from the panel on the right.";
  const saveText = saveState === "saved" ? "Saved" : saveState === "saving" ? "Saving…" : saveState === "unsaved" ? "Unsaved" : "Save failed";

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      <AppHeader user={user} actions={<Link href="/new" className="tool-btn">New photo</Link>}
        status={<span className={`text-xs ${saveState === "error" ? "text-red-500" : "text-muted"}`}>{design ? saveText : ""}</span>}>
        <nav className="flex items-center gap-3 min-w-0 text-sm">
          <Link href="/" className="text-muted hover:text-ink">Projects</Link>
          {project && <span className="font-medium truncate max-w-56">{project.name}</span>}
          {design && <span className={`text-xs rounded border px-2 py-0.5 ${locked ? "border-emerald-600 text-emerald-500" : "border-rule text-muted"}`}>{locked ? "Locked" : "Draft"}</span>}
          {design && <button type="button" className="text-draft hover:underline truncate" onClick={() => setTab("lighting")}>{lightingStatus(design)}</button>}
        </nav>
      </AppHeader>

      {/* ───────────── toolbar: tools left, view + history + actions right, one hint line under it ───────────── */}
      <div className="border-b border-rule bg-panel px-4 py-2 space-y-1.5">
        <div className="flex flex-wrap items-center gap-1.5">
          {locked ? (
            <>
              <span className="text-sm text-muted">Approved: lines and areas are locked. Lighting can still be changed.</span>
              <button type="button" className="tool-btn" onClick={() => { patchDesign({ approved: false }); say("Reopened for correction."); }}>Reopen for correction</button>
            </>
          ) : (
            <>
              {T("select", "Select", { title: "Select, move and reshape lines and areas (S)" })}
              <span className="tool-sep" />
              {T("brush", "Brush", { disabled: !areaSelected, title: areaSelected ? "Paint the selected area (B)" : "Select an area first" })}
              {T("erase", "Erase", { disabled: !areaSelected, title: areaSelected ? "Erase from the selected area (E)" : "Select an area first" })}
              {(tool === "brush" || tool === "erase") && (
                <label className="flex items-center gap-2 mx-1 text-sm text-muted">Size
                  <input type="range" min={4} max={200} step={1} value={brushSize} onChange={(e) => setBrushSize(Number(e.target.value))} className="!w-28" />
                  <span className="w-7 text-ink">{brushSize}</span>
                </label>
              )}
              {cat?.sam_api && (
                <>
                  <span className="tool-sep" />
                  {T("box", "Find in box", { title: "Drag a box around missing roof (adds its roofline), or around an object with an area selected" })}
                  {T("click", "Find by click", { title: "Click an object: SAM 3 finds it" })}
                </>
              )}
            </>
          )}
          <div className="ml-auto flex items-center gap-1.5">
            <button type="button" className="tool-btn" onClick={() => canvasRef.current?.zoomStep(-1)} disabled={zoom <= 1} aria-label="Zoom out" title="Zoom out (−)">−</button>
            <button type="button" className="tool-btn min-w-14" onClick={() => canvasRef.current?.fit()} title="Fit the photo (0)">{zoom === 1 ? "Fit" : `${Math.round(zoom * 100)}%`}</button>
            <button type="button" className="tool-btn" onClick={() => canvasRef.current?.zoomStep(1)} aria-label="Zoom in" title="Zoom in (+)">+</button>
            {!locked && (
              <>
                <span className="tool-sep" />
                <button type="button" className="tool-btn" onClick={undo} disabled={!past.length} title="Undo (⌘Z)">Undo</button>
                <button type="button" className="tool-btn" onClick={redo} disabled={!future.length} title="Redo (⇧⌘Z)">Redo</button>
                <span className="tool-sep" />
                <button type="button" className="tool-btn" disabled={!!busy} onClick={() => detect(false)}
                  title="Detect again what the instruction names; lines and areas you added are kept">Re-detect</button>
                <button type="button" className="tool-btn primary" onClick={approve} disabled={!!busy || !!drawing} title="Lock the lines and areas">Approve</button>
              </>
            )}
          </div>
        </div>
        <p className={`text-xs truncate ${message?.error ? "text-red-500" : busy ? "text-draft" : "text-muted"}`} aria-live="polite">
          {busy || message?.text || hint}
        </p>
      </div>

      {/* ───────────── canvas + panel ───────────── */}
      <div className="flex-1 min-h-0 grid lg:grid-cols-[minmax(0,1fr)_30rem]">
        <div className="relative min-h-[50vh] bg-lightbox">
          {design && session ? (
            <DesignCanvas ref={canvasRef} image={image} width={session.width} height={session.height} regions={design.regions}
              selectedId={selectedId} tool={tool} brushSize={brushSize} masks={masksRef.current} maskVersion={maskVersion}
              showObstacles={showObstacles} hidden={hidden.runs}
              drawing={drawing} locked={locked} busy={!!busy}
              onSelect={(id) => { setSelectedId(id); const r = design.regions.find((x) => x.id === id); if (r?.geometry === "line" && (tool === "brush" || tool === "erase")) setTool("select"); }}
              onLine={onLine} onStroke={onStroke} onBox={(b) => (areaSelected ? refine({ box: b }) : roofInBox(b))} onClickObject={onClickObject}
              onDrawPoint={drawPoint} onDrawFinish={finishDrawing} onZoom={setZoom} />
          ) : <div className="absolute inset-0 grid place-items-center text-muted">{busy || message?.text || "Loading…"}</div>}
        </div>
        <aside className="border-l border-rule bg-panel flex flex-col min-h-0">
          <div role="tablist" className="grid grid-cols-2 border-b border-rule">
            {(["lines", "lighting"] as const).map((t) => (
              <button key={t} role="tab" aria-selected={tab === t} type="button" onClick={() => setTab(t)}
                className={`py-3 font-medium border-b-2 ${tab === t ? "border-draft" : "border-transparent text-muted hover:text-ink"}`}>
                {t === "lines" ? "Lines & areas" : <span className="inline-flex items-center gap-2">Lighting{design && !design.lighting_confirmed && <span className="h-2 w-2 rounded-full bg-draft" />}</span>}
              </button>
            ))}
          </div>
          <div className="flex-1 overflow-y-auto p-5">
            {design && (tab === "lines" ? (
              <RegionPanel design={design} cat={cat} selectedId={selectedId} locked={locked} hiddenShare={hidden.share}
                showObstacles={showObstacles} setShowObstacles={setShowObstacles}
                onSelect={(id) => { setSelectedId(id); const r = design.regions.find((x) => x.id === id); if (r?.geometry === "line" && (tool === "brush" || tool === "erase")) setTool("select"); }}
                onUpdate={updateRegion} onDelete={deleteRegion} onAdd={addItem} drawing={!!drawing}
                onDismissNotes={() => patchDesign({ notes: [] })} />
            ) : (
              <LightingPanel design={design} cat={cat} onTargets={setTargets} onConfirm={confirmLighting} />
            ))}
          </div>
        </aside>
      </div>

    </div>
  );
}
