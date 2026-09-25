"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import AppHeader, { useUser } from "@/components/AppHeader";
import RoofCanvas, { type RoofCanvasHandle } from "@/components/RoofCanvas";
import { SegmentTable, PlaneTable } from "@/components/Tables";
import { api, isSessionLost, post } from "@/lib/api";
import { dist } from "@/lib/editor";
import {
  type Config, type Geometry, type GeometryParams, type LineDoc, type Point, type Project, type ProjectData,
  type Segment, type Segmentation, type SessionInfo,
} from "@/lib/types";

type Mode = "sam3" | "mock" | "threshold";
type Layer = "partition" | "instances" | "arcs" | "lines";

const LAYER_NOTES: Record<Layer, string> = {
  partition: "Every pixel belongs to exactly one plane. The smallest instance wins a contested pixel, so sub-planes beat their parents. Borders between two colours are where ridges, hips and valleys come from.",
  instances: "What SAM 3 returned, before anything is resolved. Overlaps show as blended colours.",
  arcs: "Green: free boundary. Yellow: border with another plane. Red: untrusted, because something stands in front (a tree, a palm, a car), so it takes no part in any line fit; the lines on either side are extended through it. Purple tint: foliage. Blue tint: solid obstacles (cars, trunks, poles).",
  lines: "Every line segment in the photo, coloured by the vanishing point it agrees with: pink vertical, yellow and blue the two horizontal directions. These fix the scene's directions; none of them is ever output.",
};
const DEFAULT_GP: GeometryParams = { epsilon: 0.01, mergeAngle: 5, snap: 0.008, wdir: 3, stage: 7, focal: null };

const typing = (t: EventTarget | null) => t instanceof HTMLInputElement || t instanceof HTMLTextAreaElement || t instanceof HTMLSelectElement;

export default function Studio() {
  const user = useUser();
  const router = useRouter();
  const query = useSearchParams();
  const canvasRef = useRef<RoofCanvasHandle>(null);

  // ── detection state (as in the original single-page app) ──
  const [config, setConfig] = useState<Config>({ sam_api: false, role_colors: {}, defaults: {} });
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [useSampleMasks, setUseSampleMasks] = useState(false);
  const [mode, setModeState] = useState<Mode>("mock");
  const [textPrompt, setTextPrompt] = useState("roof");
  const [threshold, setThreshold] = useState(80);
  const [invert, setInvert] = useState(true);
  const [treePass, setTreePass] = useState(false);
  const [refineSmall, setRefineSmall] = useState(false);
  const [obstacles, setObstacles] = useState(true);            // SAM obstacle pass: lines traced behind trees, palms, cars
  const [samples, setSamples] = useState<{ name: string; label: string }[]>([]);
  const [sampleName, setSampleName] = useState("synth_cross_high");
  const [subtract, setSubtract] = useState("");
  const [points, setPoints] = useState<Point[]>([]);
  const [gp, setGp] = useState<GeometryParams>(DEFAULT_GP);
  const [segmentation, setSegmentation] = useState<Segmentation | null>(null);
  const [detected, setDetected] = useState<Geometry | null>(null);
  const [hasPrep, setHasPrep] = useState(false);            // the backend holds a segmentation for this session
  const [segBusy, setSegBusy] = useState(false);
  const [segTime, setSegTime] = useState("");
  const [geoTime, setGeoTime] = useState("");
  const [status, setStatusState] = useState<{ text: string; error?: boolean }>({ text: "Ready" });
  const [showControls, setShowControls] = useState(false);     // detection settings drawer
  const [showInspect, setShowInspect] = useState(false);       // view A beside the photo
  const [showDetails, setShowDetails] = useState(false);       // plane and segment tables
  const [fullscreen, setFullscreen] = useState(false);         // the photo alone, filling the window
  const [layer, setLayer] = useState<Layer>("partition");
  const [hoverSeg, setHoverSeg] = useState<Segment | null>(null);

  // Lines saved by the Studio's former line editor stay visible (read-only); editing happens in the one editor at /design.
  const [edited, setEdited] = useState<LineDoc | null>(null);
  const [opening, setOpening] = useState(false);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey;
      if (mod && e.key.toLowerCase() === "s") { e.preventDefault(); save(); return; }
      if (typing(e.target) || mod) return;
      if (e.key === "f" || e.key === "F") setFullscreen((v) => !v);
      else if (e.key === "Escape") setFullscreen(false);
      else if (e.key === "=" || e.key === "+") canvasRef.current?.zoomBy(1.25);
      else if (e.key === "-") canvasRef.current?.zoomBy(0.8);
      else if (e.key === "0") canvasRef.current?.fit();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  // ── project state ──
  const [project, setProject] = useState<{ id: string; name: string } | null>(null);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);

  const setStatus = (text: string, error = false) => setStatusState({ text, error });

  // every async runner reads the newest values from here
  const latest = useRef({ session, useSampleMasks, mode, textPrompt, threshold, invert, treePass, refineSmall, obstacles, subtract, points, gp, segmentation, edited, projectId: "" });
  latest.current = { session, useSampleMasks, mode, textPrompt, threshold, invert, treePass, refineSmall, obstacles, subtract, points, gp, segmentation, edited, projectId: project?.id ?? "" };

  /** Run a call against the working session; if the backend restarted and forgot it, reopen it from the saved project once. */
  const withSession = useCallback(async <T,>(fn: (sid: string) => Promise<T>): Promise<T> => {
    const sid = latest.current.session?.session_id ?? "";
    try { return await fn(sid); }
    catch (e) {
      if (!isSessionLost(e) || !latest.current.projectId) throw e;
      const info = await post<SessionInfo>(`/api/projects/${latest.current.projectId}/session`);
      latest.current.session = info; setSession(info); setHasPrep(false);
      return fn(info.session_id);
    }
  }, []);

  const shown: LineDoc | null = edited ?? (detected ? { segments: detected.segments, planes: detected.planes, pitch: detected.info.pitch } : null);

  const planeColors = useMemo(() => Object.fromEntries((segmentation?.instances || []).filter((i) => i.plane).map((i) => [i.plane as number, i.color as string])), [segmentation]);

  // ───────────────────────────── Phases 3-7 ─────────────────────────────
  const geoTicket = useRef(0);
  const runGeometry = useCallback(async (overrideGp?: GeometryParams) => {
    const L = latest.current;
    if (!L.session) return;
    const g = overrideGp ?? L.gp, mine = ++geoTicket.current;
    try {
      const result = await post<Geometry>("/api/geometry", {
        session_id: L.session.session_id, epsilon: g.epsilon, merge_angle_deg: g.mergeAngle, snap_radius_frac: g.snap, w_dir: g.wdir,
        stop_after: g.stage, focal_35mm: g.focal && g.focal !== L.session.exif_focal_35mm ? g.focal : null,
      });
      if (mine !== geoTicket.current) return;
      setDetected(result);
      setGeoTime(`${result.timings.interactive_ms} ms`);
      setDirty(true);
    } catch (err) { if (mine === geoTicket.current) setStatus(`Geometry failed: ${(err as Error).message}`, true); }
  }, []);

  // ───────────────────────────── Phases 1-2 ─────────────────────────────
  const segTicket = useRef(0);
  const runSegmentation = useCallback(async (over: Partial<typeof latest.current> = {}) => {
    const L = { ...latest.current, ...over };
    if (!L.session) return;
    const mine = ++segTicket.current;
    const m = L.useSampleMasks ? "sample" : L.mode;
    setSegBusy(true);
    setStatus(m === "sam3" ? `Calling SAM 3 for "${L.textPrompt}"${L.points.length ? ` and ${L.points.length} point${L.points.length > 1 ? "s" : ""}` : ""}…` : "Finding roof planes…");
    try {
      const body = { session_id: L.session.session_id, mode: m, text_prompt: L.textPrompt, points: L.points, threshold: L.threshold, invert: L.invert,
        foliage: L.treePass && m === "sam3" ? "sam" : "color", refine_small: L.refineSmall && m === "sam3", obstacles: L.obstacles,
        subtract: m === "sam3" ? L.subtract.split(",").map((t) => t.trim()).filter(Boolean) : [] };
      const result = await withSession((sid) => post<Omit<Segmentation, "params">>("/api/segment", { ...body, session_id: sid }));
      body.session_id = latest.current.session?.session_id ?? body.session_id;
      if (mine !== segTicket.current) return;
      setSegmentation({ params: body, ...result });
      setHasPrep(true);
      setSegTime(result.elapsed_ms >= 1000 ? `${(result.elapsed_ms / 1000).toFixed(2)} s` : `${result.elapsed_ms} ms`);
      setStatus("Planes ready");
      await runGeometry(L.gp);
    } catch (err) { if (mine === segTicket.current) setStatus(`Segmentation failed: ${(err as Error).message}`, true); }
    finally { if (mine === segTicket.current) setSegBusy(false); }
  }, [runGeometry]);

  // sliders: live geometry, debounced (only once the backend holds a segmentation for this photo)
  const prepRef = useRef(false);
  prepRef.current = hasPrep;
  useEffect(() => {
    if (!prepRef.current) return;
    const t = setTimeout(() => runGeometry(), 30);
    return () => clearTimeout(t);
  }, [gp, runGeometry]);

  // threshold slider re-segments (local method, cheap)
  const thresholdTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  function onThreshold(v: number) {
    setThreshold(v);
    if (thresholdTimer.current) clearTimeout(thresholdTimer.current);
    thresholdTimer.current = setTimeout(() => runSegmentation({ threshold: v }), 120);
  }

  // ───────────────────────────── Images and projects ─────────────────────────────
  async function showSession(info: SessionInfo) {
    setSession(info);
    latest.current.session = info;
    const img = new Image();
    img.src = info.image_url;
    await img.decode();
    setImage(img);
  }

  function resetWork() {
    setSegmentation(null); setDetected(null); setHasPrep(false); setEdited(null); setPoints([]); setSegTime(""); setGeoTime("");
  }

  async function startSession(info: SessionInfo) {
    resetWork();
    setProject(null);
    window.history.replaceState(null, "", info.is_sample ? `/studio?sample=${info.sample_name ?? "1"}` : "/studio");
    const focal = info.exif_focal_35mm || null;
    const nextGp = { ...latest.current.gp, focal };
    setGp(nextGp);
    setUseSampleMasks(info.is_sample);
    await showSession(info);
    await runSegmentation({ session: info, useSampleMasks: info.is_sample, points: [], gp: nextGp });
  }

  async function uploadFile(file?: File | null) {
    if (!file) return;
    if (!file.type.startsWith("image/")) { setStatus("Choose an image file (JPG, PNG or WebP).", true); return; }
    if (dirty && !window.confirm("Start a new project from this photo? Unsaved changes to the current one are lost.")) return;
    setStatus(`Uploading ${file.name}…`);
    try {
      const form = new FormData(); form.append("file", file);
      await startSession(await api<SessionInfo>("/api/upload", { method: "POST", body: form }));
    } catch (err) { setStatus(`Upload failed: ${(err as Error).message}`, true); }
  }

  async function loadSample(name = sampleName) {
    if (dirty && !window.confirm("Load the sample house? Unsaved changes are lost.")) return;
    setStatus("Loading sample house…");
    try { await startSession(await post<SessionInfo>(`/api/sample?name=${encodeURIComponent(name)}`)); } catch (err) { setStatus(`Sample failed: ${(err as Error).message}`, true); }
  }

  async function openProject(id: string) {
    setStatus("Opening project…");
    try {
      const p = await api<Project>(`/api/projects/${id}`);
      const info = await post<SessionInfo>(`/api/projects/${id}/session`);
      resetWork();
      const d = p.data;
      const sp = (d.segmentation?.params || {}) as Record<string, unknown>;
      const savedMode = String(sp.mode || "");
      setUseSampleMasks(savedMode === "sample");
      if (savedMode === "sam3" || savedMode === "mock" || savedMode === "threshold") setModeState(savedMode);
      if (typeof sp.text_prompt === "string") setTextPrompt(sp.text_prompt);
      if (typeof sp.threshold === "number") setThreshold(sp.threshold);
      if (typeof sp.invert === "boolean") setInvert(sp.invert);
      if (Array.isArray(sp.points)) setPoints(sp.points as Point[]);
      setGp({ ...DEFAULT_GP, ...d.geometry_params });
      setSegmentation(d.segmentation ? { ...d.segmentation } : null);
      setDetected(d.detected);
      setEdited(d.edited);
      setProject({ id: p.id, name: p.name });
      await showSession(info);
      setDirty(false);
      setStatus(`Opened ${p.name}`);
    } catch (err) { setStatus(`Could not open the project: ${(err as Error).message}`, true); }
  }

  async function save() {
    const L = latest.current;
    if (!session || (!detected && !edited)) return;
    const segData = segmentation ? (({ layers, ...rest }) => rest)(segmentation) : null; // eslint-disable-line @typescript-eslint/no-unused-vars
    const data: ProjectData = { detected, edited: L.edited, segmentation: segData, geometry_params: gp };
    setSaving(true);
    try {
      if (!project) {
        const stem = (session.filename || "Roof").replace(/\.[^.]+$/, "");
        const name = window.prompt("Name this project", stem === "sample-house" ? "Sample house" : stem)?.trim();
        if (!name) { setSaving(false); return; }
        const created = await post<Project>("/api/projects", { session_id: session.session_id, name, data });
        setProject({ id: created.id, name: created.name });
        window.history.replaceState(null, "", `/studio?project=${created.id}`);
      } else {
        await api(`/api/projects/${project.id}`, { method: "PUT", json: { data } });
      }
      setDirty(false);
      setStatus("Saved");
    } catch (err) { setStatus(`Save failed: ${(err as Error).message}`, true); }
    finally { setSaving(false); }
  }

  async function renameProject() {
    if (!project) return;
    const name = window.prompt("Project name", project.name)?.trim();
    if (!name || name === project.name) return;
    try { await api(`/api/projects/${project.id}`, { method: "PUT", json: { name } }); setProject({ ...project, name }); }
    catch (err) { setStatus(`Rename failed: ${(err as Error).message}`, true); }
  }

  // start-up: config, then ?project= / ?sample=1, and URL presets as before (?eps=, ?stage=, ?layer=, ?mode=, ?prompt=)
  const started = useRef(false);
  useEffect(() => {
    if (!user || started.current) return;
    started.current = true;
    (async () => {
      let c = config;
      try { c = await api<Config>("/api/config"); setConfig(c); } catch { /* keep defaults */ }
      api<{ samples: { name: string; label: string }[] }>("/api/samples").then((r) => setSamples(r.samples)).catch(() => {});
      const d = c.defaults || {};
      const num = (key: string, lo: number, hi: number) => { const v = Number(query.get(key)); return query.has(key) && Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : null; };
      const g: GeometryParams = {
        ...DEFAULT_GP,
        epsilon: num("eps", 0.001, 0.05) ?? d.epsilon ?? 0.01, mergeAngle: num("merge", 1, 15) ?? d.merge_angle_deg ?? 5,
        snap: num("snap", 0.002, 0.03) ?? d.snap_radius_frac ?? 0.008, wdir: num("wdir", 0, 30) ?? d.w_dir ?? 3,
        stage: Math.round(num("stage", 3, 7) ?? 7),
      };
      setGp(g); latest.current.gp = g;
      const l = query.get("layer"); if (l && l in LAYER_NOTES) setLayer(l as Layer);
      const prompt = query.get("prompt") || c.text_prompt || "roof";
      setTextPrompt(prompt); latest.current.textPrompt = prompt;
      const qm = query.get("mode");
      const m: Mode = qm === "sam3" || qm === "mock" || qm === "threshold" ? qm : c.sam_api ? "sam3" : "mock";
      const mm = m === "sam3" && !c.sam_api ? "mock" : m;
      setModeState(mm); latest.current.mode = mm;
      const pid = query.get("project");
      if (pid) await openProject(pid);
      else if (query.has("sample")) {
        const q = query.get("sample");
        const name = q && q !== "1" ? q : "synth_cross_high";
        setSampleName(name);
        await loadSample(name);
      }
    })();
  }, [user]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const warn = (e: BeforeUnloadEvent) => { if (dirty && (detected || edited)) { e.preventDefault(); } };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty, detected, edited]);

  // ───────────────────────────── Modes ─────────────────────────────
  function setMode(m: Mode) {
    if (m === "sam3" && !config.sam_api) m = "mock";
    setModeState(m); setUseSampleMasks(false);
    runSegmentation({ mode: m, useSampleMasks: false });
  }

  function onPromptClick(p: Point, hitRadius: number) {
    if (mode === "threshold" || !session || useSampleMasks) return;
    const hit = points.findIndex((q) => dist(q, p) <= hitRadius);
    const next = hit >= 0 ? points.filter((_, i) => i !== hit) : [...points, [Math.round(p[0]), Math.round(p[1])] as Point];
    setPoints(next);
    runSegmentation({ points: next });
  }

  // ───────────────────────────── Open in the editor ─────────────────────────────
  /** The one line editor is /design: hand it these detected lines (grouped per edge, as in automatic mode). */
  async function openInEditor() {
    if (!session || !shown) return;
    setOpening(true);
    try {
      const stem = (project?.name || session.filename || "Roof").replace(/\.[^.]+$/, "");
      const created = await withSession((sid) => post<{ id: string }>("/api/design/from-studio", { session_id: sid, name: stem, segments: shown.segments }));
      router.push(`/design?project=${created.id}`);
    } catch (err) { setStatus(`Could not open the editor: ${(err as Error).message}`, true); setOpening(false); }
  }

  // ───────────────────────────── Export ─────────────────────────────
  function exportJson() {
    const g = detected, s = session, doc = shown;
    if (!s || !doc) return;
    const payload = {
      exported_at: new Date().toISOString(),
      source: { filename: s.filename, width: s.width, height: s.height, source_width: s.source_width, source_height: s.source_height, scale_from_source: s.scale_from_source, watermark: s.watermark },
      parameters: { segmentation: segmentation?.params ?? null, geometry: g?.parameters ?? null },
      edited_by_hand: !!edited,
      conventions: {
        points: "pixel coordinates in the working image; origin top-left, y increases downward",
        role: "from topology: ridge / hip / valley are shared by two planes; eave / rake are free edges (a free sloped edge of a plane that narrows upward is a hip). Lines marked edited carry the role a person chose.",
        slope: "rise over run measured against the facade's own horizontal at that edge (from the vanishing points), so camera roll does not leak in; null when vertical",
        slope_pixel: "raw (y2 - y1) / (x2 - x1) in image coordinates",
        pitch_apparent: "rise:12 as seen in the photo. It is NOT the true roof pitch: perspective changes it.",
        pitch_calibrated: "true pitch from the focal length, the vertical vanishing point and the plane's eave direction. Trust it when the focal length comes from EXIF or was entered by hand.",
        inferred: "the segment or one of its ends was recovered through an occlusion (lines extended through a tree) rather than observed",
        edited: "the line was moved, drawn or relabelled by hand in the editor; its numbers are re-measured with the same formulas",
      },
      pitch: doc.pitch ?? g?.info.pitch ?? null, vanishing_points: segmentation?.vanishing_points ?? null, graph: g?.info.graph ?? null, hidden_creases: g?.info.creases || [],
      instances: segmentation?.instances ?? [],
      planes: doc.planes,
      roof_lines: doc.segments.map((seg) => ({ id: seg.id, role: seg.role, planes: seg.planes, points: seg.points, slope: seg.slope, slope_pixel: seg.slope_pixel,
        angle_deg: seg.angle_deg, pitch_apparent: seg.pitch_apparent, length_px: seg.length_px, follows_vanishing_point: seg.vp || null,
        inferred: !!seg.inferred, hidden_crease: !!seg.crease, edited: !!seg.edited })),
      junctions: [...new Map(doc.segments.flatMap((sg) => sg.points).map((p) => [`${p[0]},${p[1]}`, p])).values()],
    };
    const stem = (project?.name || s.filename || "image").replace(/\.[^.]+$/, "").replace(/[^\w-]+/g, "_");
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `roof-geometry-${stem}.json`;
    document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(a.href);
    setStatus(`Exported ${a.download}`);
  }

  // ───────────────────────────── Render ─────────────────────────────
  const roleColors = config.role_colors;
  const editedCount = edited ? edited.segments.filter((s) => s.edited).length : 0;
  const sampleLocked = useSampleMasks;
  const [dragOver, setDragOver] = useState(false);
  const dropProps = {
    onDragOver: (e: React.DragEvent) => { e.preventDefault(); setDragOver(true); },
    onDragLeave: () => setDragOver(false),
    onDrop: (e: React.DragEvent) => { e.preventDefault(); setDragOver(false); uploadFile(e.dataTransfer.files[0]); },
  };

  return (
    <>
      <AppHeader user={user} status={<p className={`text-sm hidden md:block ${status.error ? "text-red-700" : "text-muted"}`} aria-live="polite">{status.text}</p>}>
        <nav className="flex items-center gap-2 text-sm min-w-0">
          <span className="text-rule">/</span>
          <Link href="/" className="text-muted hover:text-ink">Projects</Link>
          <span className="text-rule">/</span>
          {project ? (
            <button type="button" onClick={renameProject} title="Rename" className="font-medium truncate max-w-56 hover:underline">{project.name}</button>
          ) : <span className="text-muted">{session ? "Unsaved project" : "New project"}</span>}
          {dirty && (detected || edited) && <span className="text-xs rounded bg-amber-100 text-amber-900 px-1.5 py-0.5 shrink-0">unsaved</span>}
        </nav>
      </AppHeader>

      <div className="min-h-[calc(100vh-3.5rem)]">
        {/* ─────────────── Controls: a drawer, closed until "Settings" is clicked ─────────────── */}
        {showControls && <div className="fixed inset-0 top-14 z-30 bg-black/40" onClick={() => setShowControls(false)} />}
        <aside aria-hidden={!showControls}
          className={`fixed left-0 top-14 bottom-0 z-40 w-[340px] max-w-[90vw] bg-panel border-r border-rule p-5 space-y-7 overflow-y-auto shadow-2xl transition-transform duration-200 ${showControls ? "translate-x-0" : "-translate-x-full"}`}>
          <section className="space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold">Image</h2>
              <button type="button" onClick={() => setShowControls(false)} className="text-muted hover:text-ink px-1" aria-label="Close settings">✕</button>
            </div>
            <label htmlFor="file" tabIndex={0} {...dropProps}
              onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); document.getElementById("file")?.click(); } }}
              className={`block cursor-pointer rounded-md border border-dashed bg-paper px-4 py-5 text-center text-sm text-muted hover:border-draft focus-visible:outline-2 focus-visible:outline-draft ${dragOver ? "border-draft" : "border-rule"}`}>
              <span className="block font-medium text-ink">Drop a house photo here</span>
              <span className="block mt-1">or click to choose a JPG, PNG or WebP</span>
            </label>
            <input id="file" type="file" accept="image/*" className="sr-only" onChange={(e) => { uploadFile(e.target.files?.[0]); e.target.value = ""; }} />
            <p className="text-sm text-muted truncate" title={session && session.source_width !== session.width ? `Downscaled from ${session.source_width} × ${session.source_height}; coordinates refer to the working size.` : ""}>
              {session ? `${session.filename}, ${session.width} × ${session.height}` : "No image loaded"}
            </p>
            {session?.watermark.flagged && (
              <p className="text-sm rounded-md border border-red-300 bg-red-50 text-red-800 px-3 py-2 leading-snug">
                This photo is watermarked ({session.watermark.reason}). Watermarks add false lines; use the clean original for real measurements.
              </p>
            )}
            <div className="space-y-1.5 text-sm">
              <label htmlFor="sample" className="block">No photo handy? Try a sample house</label>
              <div className="flex gap-2">
                <select id="sample" className="field min-w-0 flex-1 !py-1" value={sampleName} onChange={(e) => setSampleName(e.target.value)}>
                  {(samples.length ? samples : [{ name: "synth_cross_high", label: "Plain house (no obstacles)" }]).map((x) => <option key={x.name} value={x.name}>{x.label}</option>)}
                </select>
                <button type="button" onClick={() => loadSample()} className="tool-btn !py-1">Load</button>
              </div>
            </div>
          </section>

          <section className="space-y-4">
            <div className="flex items-baseline justify-between"><h2 className="text-sm font-semibold">Roof planes</h2><span className="text-xs text-muted">{segTime}</span></div>
            <div role="group" aria-label="Segmentation method" className="grid grid-cols-3 rounded-md border border-rule overflow-hidden text-sm">
              {(["sam3", "mock", "threshold"] as Mode[]).map((m) => (
                <button key={m} type="button" className="seg-btn" aria-pressed={!sampleLocked && mode === m} disabled={(m === "sam3" && !config.sam_api) || !session}
                  title={m === "sam3" && !config.sam_api ? "Put FAL_KEY in .env to enable" : ""} onClick={() => setMode(m)}>
                  {m === "sam3" ? "SAM 3" : m === "mock" ? "GrabCut" : "Threshold"}
                </button>
              ))}
            </div>
            {!config.sam_api && <p className="text-xs text-muted leading-snug">SAM 3 (fal.ai) is off until FAL_KEY is set in .env and the server restarted.</p>}
            {sampleLocked && <p className="text-xs text-muted leading-snug">The sample uses its own SAM-like masks (overlaps, holes, a parent instance), so nothing is sent to the API.{session?.sample_name && session.sample_name !== "synth_cross_high" ? " It also carries the obstacle masks a SAM pass would give, so the roof lines are traced behind its palms, trees and cars (Inspect → Trusted arcs shows them)." : ""} Pick a method above to segment it yourself.</p>}

            {!sampleLocked && mode === "sam3" && (
              <div className="space-y-2 text-sm">
                <label htmlFor="prompt" className="block">What to segment</label>
                <div className="flex gap-2">
                  <input id="prompt" className="field min-w-0 flex-1" value={textPrompt} autoComplete="off" spellCheck={false}
                    onChange={(e) => setTextPrompt(e.target.value.trim())} onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); runSegmentation(); } }} />
                  <button type="button" onClick={() => runSegmentation()} disabled={!session} className="shrink-0 rounded-md bg-ink text-panel px-3 py-1.5 font-medium hover:bg-black disabled:opacity-50">Run</button>
                </div>
                {config.sam_api_host && <p className="text-muted leading-snug">Each run calls {config.sam_model} on {config.sam_api_host}, about half a cent. Results are cached per photo and prompt.</p>}
                <details>
                  <summary className="cursor-pointer text-draft font-medium">Extra SAM passes</summary>
                  <div className="mt-2 space-y-2">
                    <label className="flex items-start gap-2 cursor-pointer"><input type="checkbox" className="mt-1 h-4 w-4 accent-draft" checked={obstacles} onChange={(e) => setObstacles(e.target.checked)} /><span>Find solid obstacles (cars, poles, palm trunks) so roof lines are traced behind them <span className="text-muted">(three calls, cached)</span></span></label>
                    <label className="flex items-start gap-2 cursor-pointer"><input type="checkbox" className="mt-1 h-4 w-4 accent-draft" checked={treePass} onChange={(e) => setTreePass(e.target.checked)} /><span>Find trees with SAM instead of by colour <span className="text-muted">(one more call)</span></span></label>
                    <label className="flex items-start gap-2 cursor-pointer"><input type="checkbox" className="mt-1 h-4 w-4 accent-draft" checked={refineSmall} onChange={(e) => setRefineSmall(e.target.checked)} /><span>Sharpen small planes on a zoomed crop <span className="text-muted">(one call each)</span></span></label>
                    <label className="block" htmlFor="subtract">Cut these out of the planes</label>
                    <input id="subtract" className="field w-full" placeholder="chimney, skylight" autoComplete="off" spellCheck={false} value={subtract} onChange={(e) => setSubtract(e.target.value)} />
                  </div>
                </details>
              </div>
            )}

            {!sampleLocked && mode !== "threshold" && (
              <div className="space-y-2 text-sm">
                <p className="text-muted leading-snug">
                  {mode === "sam3"
                    ? "The text prompt returns one instance per roof plane. If a plane is missing, click it to add it as a point prompt; click a marker to remove it."
                    : "GrabCut runs locally and stands in for SAM 3. Click each roof plane to segment it from that point."}
                </p>
                <div className="flex items-center justify-between">
                  <span>Point prompts <span className="text-muted">{points.length ? `${points.length} point${points.length > 1 ? "s" : ""}` : "none"}</span></span>
                  <button type="button" disabled={!points.length} onClick={() => { setPoints([]); runSegmentation({ points: [] }); }}
                    className="text-draft font-medium hover:underline disabled:text-muted disabled:no-underline">Clear</button>
                </div>
              </div>
            )}

            {!sampleLocked && mode === "threshold" && (
              <div className="space-y-3">
                <Slider label="Gray threshold" min={0} max={255} step={1} value={threshold} format={String} onChange={onThreshold} />
                <label className="flex items-center gap-2 text-sm cursor-pointer">
                  <input type="checkbox" className="h-4 w-4 accent-draft" checked={invert} onChange={(e) => { setInvert(e.target.checked); runSegmentation({ invert: e.target.checked }); }} />
                  Dark pixels are the roof
                </label>
              </div>
            )}
            {session && !hasPrep && (detected || edited) && (
              <div className="rounded-md border border-rule bg-paper p-3 text-sm space-y-2">
                <p className="text-muted leading-snug">These lines were loaded from the saved project. Run detection again to inspect the planes or retune the geometry.{mode === "sam3" && !sampleLocked ? " It calls SAM 3 once more." : ""}</p>
                <button type="button" onClick={() => runSegmentation()} className="rounded-md bg-ink text-panel px-3 py-1.5 font-medium hover:bg-black">Detect again</button>
              </div>
            )}
          </section>

          <section className="space-y-4">
            <div className="flex items-baseline justify-between"><h2 className="text-sm font-semibold">Geometry</h2><span className="text-xs text-muted">{geoTime}</span></div>
            {edited && <p className="text-xs rounded-md bg-amber-50 border border-amber-200 text-amber-900 px-2.5 py-1.5 leading-snug">This project has lines edited in the former Studio editor; they are shown as saved. Edit lines opens them in the editor.</p>}
            <Slider label="Corner tolerance (ε)" min={0.001} max={0.05} step={0.001} value={gp.epsilon} disabled={!hasPrep} format={(v) => v.toFixed(3)}
              onChange={(v) => setGp((g) => ({ ...g, epsilon: v }))} note="Douglas-Peucker only seeds corners. Higher means fewer seeds, so longer fitted lines." />
            <Slider label="Merge angle" min={1} max={15} step={0.5} value={gp.mergeAngle} disabled={!hasPrep} format={(v) => `${v}°`}
              onChange={(v) => setGp((g) => ({ ...g, mergeAngle: v }))} note="Neighbouring edges closer than this in direction become one line." />
            <Slider label="Junction snap radius" min={0.002} max={0.03} step={0.001} value={gp.snap} disabled={!hasPrep} format={(v) => `${(v * 100).toFixed(1)}% of the long edge`}
              onChange={(v) => setGp((g) => ({ ...g, snap: v }))} note="Corners of different planes closer than this share one junction." />
            <Slider label="Pull toward vanishing points" min={0} max={30} step={0.5} value={gp.wdir} disabled={!hasPrep} format={(v) => (v === 0 ? "off" : `${v}×`)}
              onChange={(v) => setGp((g) => ({ ...g, wdir: v }))} note="How strongly eaves and ridges follow the scene's horizontal directions. 0 turns it off." />
            <div className="text-sm">
              <label htmlFor="stage" className="block mb-1">Show the result after</label>
              <select id="stage" className="field w-full" value={gp.stage} disabled={!hasPrep} onChange={(e) => setGp((g) => ({ ...g, stage: Number(e.target.value) }))}>
                <option value={3}>3 · line fitting per plane</option>
                <option value={4}>4 · vanishing-point directions</option>
                <option value={5}>5 · planar graph and energy loop</option>
                <option value={6}>6 · hidden creases</option>
                <option value={7}>7 · roles and pitch (final)</option>
              </select>
            </div>
          </section>

          <section className="space-y-3 text-sm">
            <h2 className="text-sm font-semibold">Pitch calibration</h2>
            <div className="flex items-center gap-2">
              <label htmlFor="focal" className="shrink-0">Focal length</label>
              <input id="focal" type="number" min={8} max={400} step={1} placeholder="26" className="field w-20" value={gp.focal ?? ""}
                onChange={(e) => { const f = Number(e.target.value) || null; setGp((g) => ({ ...g, focal: f })); }} />
              <span className="text-muted">mm, 35 mm equivalent</span>
            </div>
            <p className="text-xs text-muted leading-snug">
              {session?.exif_focal_35mm
                ? `Read from ${session.is_sample ? "the sample's known camera" : "the photo's EXIF"}: ${session.exif_focal_35mm} mm. Change it only if you know better.`
                : "True pitch needs the focal length. It is read from EXIF when the photo has it. A phone's main camera is about 26 mm, its ultrawide about 13 mm."}
            </p>
            <PitchReadout doc={shown} />
          </section>

          <section className="text-xs text-muted space-y-1.5">
            {Object.entries(roleColors).map(([role, color]) => (
              <div key={role} className="flex items-center gap-2"><span className="inline-block w-5 h-0.5" style={{ background: color }} />{role}</div>
            ))}
            <div className="flex items-center gap-2"><span className="inline-block w-5 border-t-2 border-dashed border-neutral-400" />inferred through an occlusion</div>
            <div className="flex items-center gap-2"><span className="inline-block w-2.5 h-2.5 rounded-full" style={{ background: "#FF3B30" }} />junction</div>
          </section>
        </aside>

        {/* ─────────────── Viewports ─────────────── */}
        <main className={fullscreen ? "fixed inset-0 z-50 overflow-auto bg-paper px-4 py-3 space-y-3" : "p-5 space-y-4 min-w-0"}>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-3">
              {!fullscreen && (
                <>
                  <button type="button" className="tool-btn" aria-pressed={showControls} onClick={() => setShowControls((v) => !v)} title="Photo, segmentation method, geometry sliders, focal length">Settings</button>
                  <button type="button" className="tool-btn" aria-pressed={showInspect} onClick={() => setShowInspect((v) => !v)} title="Show the inspection layers (A) beside the photo">Inspect</button>
                  <button type="button" className="tool-btn" aria-pressed={showDetails} onClick={() => setShowDetails((v) => !v)} disabled={!shown} title="Plane and segment tables">Details</button>
                </>
              )}
              <button type="button" className="tool-btn" aria-pressed={fullscreen} onClick={() => setFullscreen((v) => !v)} disabled={!session} title="The photo alone, filling the window (F)">
                {fullscreen ? "Exit full screen" : "Full screen"}
              </button>
              {shown && (
                <button type="button" onClick={openInEditor} disabled={opening} className="tool-btn primary"
                  title="Open these lines in the editor, where lines and areas are edited">
                  {opening ? "Opening the editor…" : "Edit lines"}
                </button>
              )}
            </div>
            <div className="flex items-center gap-2">
              <button type="button" onClick={save} disabled={!session || !shown || saving} className="tool-btn">
                {saving ? "Saving…" : project ? (dirty ? "Save changes" : "Saved") : "Save project"}
              </button>
              <button type="button" onClick={exportJson} disabled={!shown || !shown.segments.length} className="tool-btn">Export roof geometry</button>
            </div>
            <p className={`text-sm text-muted truncate basis-full ${fullscreen ? "hidden" : ""}`}>
              {hoverSeg ? `Segment ${hoverSeg.id + 1}: ${hoverSeg.role}${hoverSeg.planes?.length ? ` of plane ${hoverSeg.planes.join(" + ")}` : ""}, pitch as seen ${hoverSeg.pitch_apparent || "steep"}, ${hoverSeg.length_px.toFixed(0)} px${hoverSeg.inferred ? ", inferred through an occlusion" : ""}` : ""}
            </p>
          </div>

          <div className={`grid gap-4 ${showInspect && !fullscreen ? "xl:grid-cols-2" : ""}`}>
            <figure className={`space-y-2 min-w-0 ${showInspect && !fullscreen ? "" : "hidden"}`}>
              <figcaption className="text-sm flex flex-wrap items-center gap-x-2 gap-y-1">
                <span className="font-medium">A</span>
                <select className="field text-sm !py-1" aria-label="Inspection layer" value={layer} onChange={(e) => setLayer(e.target.value as Layer)}>
                  <option value="partition">Hard partition (phase 1)</option>
                  <option value="instances">Raw SAM instances</option>
                  <option value="arcs">Trusted arcs (phase 2)</option>
                  <option value="lines">All lines, by vanishing point (phase 4)</option>
                </select>
                <span className="text-muted">{segmentation ? coverageText(segmentation) : ""}</span>
              </figcaption>
              <div className={`frame ${!session ? "empty" : ""} ${segBusy ? "busy" : ""}`} style={session ? { aspectRatio: `${session.width} / ${session.height}` } : undefined} {...dropProps}>
                {hasPrep && segmentation?.layers ? (
                  /* eslint-disable-next-line @next/next/no-img-element -- a backend-rendered inspection layer behind the session cookie */
                  <img src={segmentation.layers[layer]} alt="Inspection layer of the segmentation" className="block w-full h-auto" />
                ) : (
                  <p className="text-sm text-neutral-400 px-6 text-center">{session && !hasPrep && (detected || edited) ? "Run detection again to see the planes for this saved project." : "The partition of the roof into planes appears here."}</p>
                )}
              </div>
              <p className="text-xs text-muted leading-snug">{LAYER_NOTES[layer]}</p>
            </figure>

            <figure className="space-y-2 min-w-0 mx-auto w-full" {...dropProps}
              style={session && fullscreen ? { maxWidth: `calc((100vh - 5rem) * ${session.width / session.height})` } : undefined}>
              {showInspect && !fullscreen && <figcaption className="text-sm"><span className="font-medium">B</span> <span className="text-muted">planes and lines over the photo</span></figcaption>}
              {edited && <p className="text-xs"><span className="rounded bg-amber-100 text-amber-900 px-1.5 py-0.5">edited by hand</span></p>}
              {session ? (
                <RoofCanvas
                  ref={canvasRef} image={image} width={session.width} height={session.height} doc={shown}
                  roleColors={roleColors} planeColors={planeColors} editing={false} tool="select" drawRole="ridge"
                  selection={null} points={points} busy={segBusy}
                  showPromptPlaceholder={mode === "mock" && !useSampleMasks && !points.length}
                  onSelect={() => {}} onChange={() => {}}
                  onAdd={() => {}}
                  onSplit={() => {}}
                  onPromptClick={onPromptClick} onHover={setHoverSeg}
                />
              ) : (
                <div className="frame empty">
                  <div className="text-sm text-neutral-400 px-6 text-center space-y-3">
                    <p>No image yet. Upload a house photo, or try the sample.</p>
                    <div className="flex justify-center gap-2">
                      <button type="button" onClick={() => document.getElementById("file")?.click()} className="tool-btn primary">Upload a photo</button>
                      <button type="button" onClick={() => loadSample()} className="tool-btn">Load sample house</button>
                    </div>
                  </div>
                </div>
              )}
            </figure>
          </div>

          {showDetails && !fullscreen && shown && <p className="text-sm text-muted">{describeResult(detected, shown, edited, editedCount)}</p>}
          {showDetails && !fullscreen && shown && <PlaneTable doc={shown} focalKnown={!!shown.pitch?.focal_px} edited={!!edited} />}
          {showDetails && !fullscreen && shown && shown.segments.length > 0 && (
            <SegmentTable doc={shown} roleColors={roleColors} editing={false} hover={hoverSeg?.id ?? -1}
              selected={-1} onSelect={() => {}} onRole={() => {}} />
          )}
        </main>
      </div>
    </>
  );
}

// ───────────────────────────── Small pieces ─────────────────────────────

function Slider({ label, min, max, step, value, onChange, format, note, disabled }: {
  label: string; min: number; max: number; step: number; value: number; onChange(v: number): void; format(v: number): string; note?: string; disabled?: boolean;
}) {
  return (
    <div>
      <div className="flex items-baseline justify-between text-sm"><label>{label}</label><output className="text-muted">{format(value)}</output></div>
      <input type="range" min={min} max={max} step={step} value={value} disabled={disabled} aria-label={label} onChange={(e) => onChange(Number(e.target.value))} />
      {note && <p className="text-xs text-muted leading-snug">{note}</p>}
    </div>
  );
}

function PitchReadout({ doc }: { doc: LineDoc | null }) {
  const p = doc?.pitch;
  if (!p) return null;
  if (p.roof) {
    const trusted = p.focal_source === "EXIF" || p.focal_source === "entered by hand";
    return (
      <p className="leading-snug">
        <span className="font-medium">Roof pitch {p.roof.pitch}</span> ({p.roof.pitch_deg}°), nearest standard {p.roof.standard.pitch}.{" "}
        <span className="text-muted">{trusted ? "Calibrated" : "Low confidence"}: focal length {Math.round(p.focal_px || 0)} px, {p.focal_source}.</span>
      </p>
    );
  }
  return <p className="leading-snug text-muted">{p.focal_px ? "This view does not constrain the pitch (no plane shows both an eave on a vanishing point and a consistent slope)." : "No focal length, so only the pitch as seen in the photo is available."}</p>;
}

function coverageText(s: Segmentation) {
  const dropped = s.instances.filter((i) => i.status !== "plane").length;
  return `${s.planes} plane${s.planes === 1 ? "" : "s"} from ${s.instances.length} instance${s.instances.length === 1 ? "" : "s"}`
    + (dropped ? `, ${dropped} dropped as duplicates` : "") + `. ${s.adjacency_percent}% of the boundary is shared, ${s.untrusted_percent}% untrusted.`;
}

function describeResult(g: Geometry | null, doc: LineDoc | null, edited: LineDoc | null, editedCount: number) {
  if (!doc) return "Export writes a JSON file with every plane and segment: endpoints, role, slope, pitch and length.";
  if (!doc.segments.length) return edited ? "No lines. Draw some, or discard the edits." : "No lines. Check the partition in view A, or lower the corner tolerance.";
  const counts: Record<string, number> = {};
  for (const s of doc.segments) counts[s.role] = (counts[s.role] || 0) + 1;
  const parts = Object.entries(counts).map(([r, n]) => `${n} ${r}${n > 1 ? "s" : ""}`);
  const junctions = new Set(doc.segments.flatMap((s) => s.points.map((p) => `${p[0]},${p[1]}`))).size;
  let text = `${doc.planes.length} plane${doc.planes.length === 1 ? "" : "s"}, ${doc.segments.length} segments (${parts.join(", ")}), ${junctions} junctions.`;
  if (edited) return text + ` Edited by hand in the former Studio editor: ${editedCount} line${editedCount === 1 ? "" : "s"} changed or drawn.`;
  const gr = g?.info.graph;
  if (gr) text += ` Energy ${gr.energy_start} to ${gr.energy_end} in ${gr.rounds} round${gr.rounds === 1 ? "" : "s"}; polygons and partition disagree on ${(gr.coverage.after.total * 100).toFixed(1)}% of the roof.`;
  if (g?.info.creases?.length) text += ` ${g.info.creases.length} hidden crease${g.info.creases.length > 1 ? "s" : ""} confirmed by image lines.`;
  if (g && g.stage < 7) text += ` Showing the result after phase ${g.stage}.`;
  return text;
}
