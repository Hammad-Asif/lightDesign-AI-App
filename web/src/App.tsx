import { useCallback, useEffect, useRef, useState } from "react";

import * as api from "./api";
import { MaskCanvas, type MaskCanvasHandle, type Snapshot } from "./canvas/MaskCanvas";
import { AccountMenu } from "./components/AccountMenu";
import { CanvasLegend } from "./components/CanvasLegend";
import { ConfirmDialog, type Confirmation } from "./components/ConfirmDialog";
import { DetectionProgress } from "./components/DetectionProgress";
import { ProjectName } from "./components/ProjectName";
import { ShortcutSheet } from "./components/ShortcutSheet";
import { StepBar, type Step as ProgressStep } from "./components/StepBar";
import { Toasts, type Notice } from "./components/Toasts";
import { Activity } from "./components/Activity";
import { Auth } from "./components/Auth";
import { ModeChooser } from "./components/ModeChooser";
import { Products } from "./components/Products";
import { forgetProjects, Projects, rememberProject } from "./components/Projects";
import { ThemeToggle, type Theme } from "./components/ThemeToggle";
import { EdgePicker } from "./components/EdgePicker";
import { EDGE_COLORS, needsOverride, RegionPanel } from "./components/RegionPanel";
import { HistoryButtons, ToolDock, ViewportControls } from "./components/Toolbar";
import { Upload } from "./components/Upload";
import { Specification } from "./components/Specification";
import { formatLength, MeasurePanel, remeasured, type MeasureMode } from "./components/MeasurePanel";
import { ScaleDebugPanel, stepFile } from "./components/ScaleDebugPanel";
import { guideSeen, hasGuide, markGuideSeen, ToolGuide } from "./components/ToolGuide";
import { forgetThumbnails } from "./thumbnails";
import { BLOCKER_KINDS, geometryOf, KIND_COLORS } from "./types";
import type {
  Design,
  DesignRequest,
  MaskingMode,
  PhotoGeometry,
  Placement,
  Measurement,
  PaletteColour,
  PhotoDetail,
  Product,
  Region,
  RegionKind,
  ScaleCheckInput,
  ScaleOverride,
  Tool,
  User,
} from "./types";

// One undo step: a stroke or a line edit on the canvas (the state before it), a change made in the
// list (the values before and after), or a line or area added or deleted (`present`: whether it
// exists after the change; `region` and `mask` are what to bring back).
type Step =
  | { type: "canvas"; snapshot: Snapshot }
  | { type: "patch"; id: string; before: Partial<Region>; after: Partial<Region> }
  | { type: "exists"; region: Region; mask: string | null; present: boolean };

const HISTORY_LIMIT = 30;

// Where the app is, in the address bar after "#/": the projects list, a new upload, the product
// library, or one photo. A hash, so the API's own paths (/photos/...) never clash with a page's, and
// the browser's Back and Forward move between these.
type Route = { name: "projects" } | { name: "new" } | { name: "products" } | { name: "photo"; id: string };

function readRoute(): Route {
  const [name, id] = window.location.hash.replace(/^#\/?/, "").split("/");
  if (name === "photo" && id) return { name: "photo", id };
  if (name === "new") return { name: "new" };
  if (name === "products") return { name: "products" };
  return { name: "projects" };
}

const stepRegion = (step: Step) =>
  step.type === "canvas" ? step.snapshot.regionId : step.type === "patch" ? step.id : step.region.id;

export default function App() {
  const [user, setUser] = useState<User | null>(null);
  const [checkingSession, setCheckingSession] = useState(true);
  // index.html picks the first theme before paint; this keeps it in step after that.
  const [theme, setTheme] = useState<Theme>(
    () => (document.documentElement.dataset.theme as Theme) || "light",
  );
  const [photo, setPhoto] = useState<PhotoDetail | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  // The whole house, picked by tapping its roof with Select (a line or area pick clears it).
  const [houseSelected, setHouseSelected] = useState(false);
  const [tool, setTool] = useState<Tool>("select");
  const [brushSize, setBrushSize] = useState(40);
  const [zoom, setZoom] = useState(1);
  const [history, setHistory] = useState({ canUndo: false, canRedo: false });
  // Undo and redo stacks, newest last. A step being applied blocks the next one.
  const past = useRef<Step[]>([]);
  const future = useRef<Step[]>([]);
  const travelling = useRef(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // A neutral message, e.g. that Find in Box found nothing new. Not an error.
  const [notice, setNotice] = useState<Notice | null>(null);
  const say = (text: string, action?: Notice["action"]) => setNotice({ text, action });
  // An action asked first (detect again, approve, delete a project).
  const [confirmation, setConfirmation] = useState<Confirmation | null>(null);
  // Detection running: the whole app waits behind its progress card.
  const [detecting, setDetecting] = useState<{ label: string; staged: boolean } | null>(null);
  // Night preview: the photo darkened, only the lights.
  const [night, setNight] = useState(false);
  const [shortcuts, setShortcuts] = useState(false);
  // Narrow screens: the panel is a sheet under the photo that folds down to its handle.
  const [sheetOpen, setSheetOpen] = useState(true);
  // A roof line just drawn, waiting for its type to be confirmed; `picked` once someone chose.
  const [typing, setTyping] = useState<{ id: string; picked: boolean } | null>(null);
  // Development (ROOFLINE_DEBUG): the pipeline stage whose roof lines are drawn over the photo.
  const [debugStage, setDebugStage] = useState<string | null>(null);
  const [drawingKind, setDrawingKind] = useState<RegionKind | null>(null);
  const [products, setProducts] = useState<Product[]>([]);
  const [palette, setPalette] = useState<PaletteColour[]>([]);
  // The photo's confirmed lighting, if any; null until the user confirms it.
  const [design, setDesign] = useState<Design | null>(null);
  // The sidebar shows either the lines and areas, or the lighting for them.
  const [sidebarTab, setSidebarTab] = useState<"regions" | "lighting">("regions");
  const [showProducts, setShowProducts] = useState(false);
  // With no photo open: the projects list, or (after "New photo") the upload form.
  const [showUpload, setShowUpload] = useState(false);
  const [showObstacles, setShowObstacles] = useState(false);
  // Measure tool: two points tapped on a feature, or a line picked on the photo.
  const [measureMode, setMeasureMode] = useState<MeasureMode>("points");
  const [measurePoints, setMeasurePoints] = useState<number[][]>([]);
  // Every roof line's real length and how it was placed; recomputed on the server after any change.
  const [geometry, setGeometry] = useState<PhotoGeometry | null>(null);
  // Development (SCALE_DEBUG): the scale solve's step whose overlay is drawn over the photo.
  const [scaleStep, setScaleStep] = useState<string | null>(null);
  // Real lengths drawn on the photo's lines, once it has a measurement.
  const [showLengths, setShowLengths] = useState(true);
  // A tool's how-to card: shown the first time the tool is picked, and from the dock's "?".
  const [guide, setGuide] = useState<Tool | null>(null);
  const closeGuide = useCallback(() => {
    setGuide((open) => {
      if (open) markGuideSeen(open);
      return null;
    });
  }, []);
  // The confirmed lighting's bulbs, placed on the server from the camera and the measurement.
  const [placement, setPlacement] = useState<Placement | null>(null);

  useEffect(() => {
    // Wreaths are placed before any lighting is confirmed, so this runs for every photo.
    if (!photo) {
      setPlacement(null);
      return;
    }
    let current = true;
    api.getPlacement(photo.id).then(
      (found) => current && setPlacement(found),
      () => current && setPlacement(null),
    );
    return () => {
      current = false;
    };
  }, [photo, design]);

  useEffect(() => {
    if (!photo) {
      setGeometry(null);
      return;
    }
    let current = true;
    api.getGeometry(photo.id).then(
      (found) => current && setGeometry(found),
      () => current && setGeometry(null),
    );
    return () => {
      current = false;
    };
  }, [photo]);

  const canvasRef = useRef<MaskCanvasHandle>(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("ld-theme", theme);
    } catch {
      // A browser with storage blocked still gets the theme, just not remembered.
    }
  }, [theme]);

  useEffect(() => {
    api
      .me()
      .then(setUser)
      .catch(() => setUser(null))
      .finally(() => setCheckingSession(false));
  }, []);

  // The catalogue and palette are read once per sign-in; the specification is shown against them.
  useEffect(() => {
    if (!user) return;
    Promise.all([api.listProducts(), api.palette()])
      .then(([items, colours]) => {
        setProducts(items);
        setPalette(colours);
      })
      .catch(fail);
  }, [user?.id]);

  const resetWorkspace = () => {
    setPhoto(null);
    setActiveId(null);
    setHouseSelected(false);
    setDrawingKind(null);
    setDesign(null);
    setSidebarTab("regions");
    setShowObstacles(false);
  };

  // A 401 anywhere means the session ended: return to sign-in instead of showing an error.
  const fail = (cause: unknown) => {
    if (cause instanceof api.UnauthorizedError) {
      setUser(null);
      resetWorkspace();
      forgetProjects();
      return;
    }
    setError(cause instanceof Error ? cause.message : String(cause));
  };

  const guard = async <T,>(work: () => Promise<T>): Promise<T | undefined> => {
    setBusy(true);
    setError(null);
    try {
      return await work();
    } catch (cause) {
      fail(cause);
      return undefined;
    } finally {
      setBusy(false);
    }
  };

  const handleSignOut = async () => {
    await api.logout().catch(() => undefined);
    setUser(null);
    resetWorkspace();
    // The next account on this browser sees none of this one's projects.
    forgetProjects();
    void forgetThumbnails();
  };

  const reportHistory = () => setHistory({ canUndo: past.current.length > 0, canRedo: future.current.length > 0 });

  const record = (step: Step) => {
    past.current.push(step);
    if (past.current.length > HISTORY_LIMIT) past.current.shift();
    future.current = [];
    reportHistory();
  };

  const clearHistory = () => {
    past.current = [];
    future.current = [];
    reportHistory();
  };

  // After lines were redrawn or joined on the server: steps about a line or area that is gone go,
  // except one that brings back what was deleted on purpose. With `redrawn` (a House tap), detected
  // lines are not brought back either: the tap drew their replacements.
  const pruneHistory = (regions: Region[], redrawn = false) => {
    const ids = new Set(regions.map((region) => region.id));
    const keep = (step: Step, absentAfter: boolean) => {
      if (step.type === "exists" && absentAfter) {
        return !ids.has(step.region.id) && !(redrawn && step.region.geometry === "line" && step.region.source !== "manual");
      }
      return ids.has(stepRegion(step));
    };
    past.current = past.current.filter((step) => keep(step, step.type === "exists" && !step.present));
    future.current = future.current.filter((step) => keep(step, step.type === "exists" && step.present));
    reportHistory();
  };

  const mergeRegion = (region: Region) =>
    setPhoto((current) =>
      current
        ? {
            ...current,
            regions: current.regions.map((item) => (item.id === region.id ? region : item)),
          }
        : current,
    );

  const handleUpload = (file: File, prompt: string, name: string) =>
    guard(async () => {
      const uploaded = await api.uploadPhoto(file, prompt, name);
      rememberProject(uploaded);
      setDesign(null);
      setPhoto(await api.getPhoto(uploaded.id));
      setShowUpload(false);
    });

  // A project from the list opens where it was left: the mode choice, or the editor.
  const handleOpenProject = (id: string) =>
    guard(async () => {
      const detail = await api.getPhoto(id);
      setDesign(null);
      setActiveId(null);
      setSidebarTab("regions");
      setShowUpload(false);
      setNight(false);
      setPhoto(detail);
      return true;
    });

  const handleRename = (name: string) =>
    guard(async () => {
      if (!photo) return;
      const saved = await api.renamePhoto(photo.id, name);
      setPhoto((current) => (current && current.id === saved.id ? { ...current, name: saved.name } : current));
      rememberProject({ ...photo, name: saved.name });
    });

  // Leave the open photo for the projects list, or for a new upload.
  const goToProjects = (upload = false) => {
    // Its card shows where it was left (mode, locked), without reading the list again.
    if (photo) rememberProject(photo);
    setPhoto(null);
    setActiveId(null);
    setDrawingKind(null);
    setDesign(null);
    setNight(false);
    setShowUpload(upload);
  };

  // The library changed under us. Refresh it, and read the instruction again so the
  // confirmation screen resolves against the new list without a reload.
  const handleLibraryChanged = () =>
    guard(async () => {
      setProducts(await api.listProducts());
      if (photo) {
        await api.interpret(photo.id);
        setPhoto(await api.getPhoto(photo.id));
      }
    });

  const handleInterpret = (prompt: string) =>
    guard(async () => {
      if (!photo) return;
      await api.interpret(photo.id, prompt);
      setPhoto(await api.getPhoto(photo.id));
    });

  const handleMode = (mode: MaskingMode) =>
    guard(async () => {
      if (!photo) return;
      await api.setMode(photo.id, mode);
      if (mode === "automatic") {
        setDetecting({ label: "Detecting the house", staged: true });
        try {
          await api.detectRegions(photo.id);
        } finally {
          setDetecting(null);
        }
      }
      const detail = await api.getPhoto(photo.id);
      setPhoto(detail);
      clearHistory();
      // Start on the first thing the instruction asked for, never on an obstacle.
      const first = detail.regions.find((region) => region.role !== "blocker");
      setActiveId(first?.id ?? null);
      setTool(first?.geometry === "line" ? "vertex" : "brush");
    });

  const handleConfirmLighting = (requests: DesignRequest[]) =>
    guard(async () => {
      if (!photo) return;
      setDesign(await api.putDesign(photo.id, requests));
    });

  const handleRedetect = (allKinds = false) =>
    guard(async () => {
      if (!photo) return;
      setDetecting({ label: allKinds ? "Detecting everything" : "Detecting again", staged: true });
      try {
        await api.detectRegions(photo.id, { allKinds });
      } finally {
        setDetecting(null);
      }
      setPhoto(await api.getPhoto(photo.id));
      // Every detected line and area was replaced: nothing to step back through.
      clearHistory();
    });

  const handleMaskCommit = useCallback(
    async (regionId: string, blob: Blob) => {
      const region = await api.putMask(photo!.id, regionId, blob);
      // Repainting something in front of the house changes what the lines behind it hide, and
      // can join two pieces of a roofline into one — so the whole photo is re-read, and a
      // selection that pointed at the absorbed piece falls back to nothing rather than dangling.
      if (BLOCKER_KINDS.includes(region.kind)) {
        const detail = await api.getPhoto(photo!.id);
        setPhoto(detail);
        pruneHistory(detail.regions);
        setActiveId((current) => (current && detail.regions.some((r) => r.id === current) ? current : region.id));
      } else mergeRegion(region);
      return region;
    },
    [photo?.id],
  );

  const handlePolylineCommit = useCallback(
    (regionId: string, polyline: number[][]) => {
      setPhoto((current) =>
        current
          ? {
              ...current,
              regions: current.regions.map((item) =>
                item.id === regionId ? { ...item, polyline } : item,
              ),
            }
          : current,
      );
      api
        .patchRegion(photo!.id, regionId, { polyline })
        .then((saved) =>
          // Take only the recomputed hidden stretches: a newer drag may already be on screen.
          setPhoto((current) =>
            current
              ? {
                  ...current,
                  regions: current.regions.map((item) =>
                    item.id === saved.id ? { ...item, hidden: saved.hidden } : item,
                  ),
                }
              : current,
          ),
        )
        .catch(fail);
    },
    [photo?.id],
  );

  // A line or area picked, on the photo or in the list: it is what is selected now, not the house,
  // and the list shows it (a pick on the photo while Lighting is open goes back to the list).
  const selectRegion = (id: string | null, fromPhoto = false) => {
    setActiveId(id);
    setHouseSelected(false);
    if (id && fromPhoto) setSidebarTab("regions");
  };

  const selectHouse = () => {
    setActiveId(null);
    setHouseSelected(true);
    setSidebarTab("regions");
  };

  // Find lines: detection runs inside the box and only adds lines; the new ones are selected.
  const handleBox = (box: number[]) =>
    guard(async () => {
      if (!photo) return;
      setNotice(null);
      setDetecting({ label: "Finding lines in the box", staged: false });
      let found: Region[];
      try {
        found = await api.findLinesInBox(photo.id, box);
      } finally {
        setDetecting(null);
      }
      setPhoto(await api.getPhoto(photo.id));
      if (found.length) selectRegion(found[0].id);
      else say("No new roof lines found in that box. Try a box closer around the missing roof edge.");
    });

  const handlePoint = (point: number[]) =>
    guard(async () => {
      if (!photo) return;
      const region = await api.refineRegion(photo.id, { point, kind: "custom" });
      const detail = await api.getPhoto(photo.id);
      setPhoto(detail);
      setActiveId(region.id);
      record({ type: "exists", region: detail.regions.find((r) => r.id === region.id) ?? region, mask: null, present: true });
    });

  // A tap with the House tool redraws the detected roof lines, so a selected line, and the undo
  // steps of the old lines, may be gone; confirmed lighting moves to the new lines on the server.
  // One tap at a time: each reads the house the last one left. A locked photo's lines stay put.
  const handleHouseTap = (point: number[]) => {
    if (!photo || photo.approved || busy) return;
    void guard(async () => {
      const detail = await api.tapHousePlane(photo.id, point);
      pruneHistory(detail.regions, true);
      setPhoto(detail);
      setActiveId((current) => (current && detail.regions.some((r) => r.id === current) ? current : null));
      if (design) setDesign(await api.getDesign(photo.id));
    });
  };

  // A third tap starts a new pair: the first two were not the ones wanted.
  const handleMeasureTap = (point: number[]) =>
    setMeasurePoints((points) => (points.length >= 2 ? [point] : [...points, point]));

  const handleMeasureSave = (measurement: Measurement | null) => {
    if (!photo) return;
    void guard(async () => {
      const saved = await api.putScale(photo.id, measurement);
      setPhoto({ ...photo, ...saved, regions: photo.regions });
      setMeasurePoints([]);
      if (!measurement) return;
      // Back to the house, now labelled with real lengths, and say what the measurement gave.
      chooseTool("select");
      setShowLengths(true);
      const found = await api.getGeometry(photo.id);
      setGeometry(found);
      const lines = Object.values(found.lines);
      const measured = lines.filter((line) => line.length_in != null).length;
      const waiting = lines.length - measured;
      const side = found.reference?.facade ?? saved.scale?.measurement?.facade ?? "front";
      say(
        `Scale set from ${formatLength(measurement.length_in)} on the ${side} side. ` +
          `${measured} ${measured === 1 ? "line" : "lines"} measured` +
          (waiting ? `; ${waiting} ${waiting === 1 ? "needs" : "need"} attention (marked ? on the photo).` : ".") +
          (found.warning ? ` ${found.warning}` : ""),
      );
    });
  };

  // Measure again with the last saved values: the same points or line, length and side (a side
  // read from the photo is read again, with the current rules).
  const handleRemeasure = (saved: Measurement) => {
    if (saved.region_id) {
      setMeasureMode("line");
      setActiveId(saved.region_id);
    } else {
      setMeasureMode("points");
      setMeasurePoints(saved.points ?? []);
    }
    handleMeasureSave(remeasured(saved));
  };

  // A second known length: saved with the photo, and the geometry (refetched with it) reports how
  // far the photo's length for that line is from it.
  const handleCheck = (check: ScaleCheckInput | null) => {
    if (!photo) return;
    void guard(async () => {
      const saved = await api.putScaleCheck(photo.id, check);
      setPhoto({ ...photo, ...saved, regions: photo.regions });
    });
  };

  const handleOverride = (regionId: string, override: ScaleOverride | null) => {
    if (!photo) return;
    void guard(async () => {
      const kept = photo.scale?.overrides?.[regionId];
      // A new middle keeps a typed length and the other way round; null clears both.
      const saved = await api.putOverride(photo.id, regionId, override && { ...kept, ...override });
      setPhoto({ ...photo, ...saved, regions: photo.regions });
    });
  };

  const handleWreathTap = (point: number[]) => {
    if (!photo || photo.approved || busy) return;
    void guard(async () => {
      const detail = await api.addWreath(photo.id, point);
      setPhoto(detail);
      const added = detail.regions[detail.regions.length - 1];
      setActiveId(added?.id ?? null);
      if (added) record({ type: "exists", region: added, mask: null, present: true });
    });
  };

  // A change to a line or area, shown at once and then saved; the server's answer replaces it.
  const savePatch = async (id: string, patch: Partial<Region>) => {
    if (!photo) return;
    const current = photo.regions.find((item) => item.id === id);
    if (current) mergeRegion({ ...current, ...patch } as Region);
    const saved = await api.patchRegion(photo.id, id, patch);
    // A kind change can turn an area into, or out of, something lines pass behind.
    if (patch.kind) setPhoto(await api.getPhoto(photo.id));
    else mergeRegion(saved);
    // A line's type, side or visibility decides which confirmed requests light it.
    if (design && ("edge" in patch || "side" in patch || "visible" in patch || "kind" in patch))
      setDesign(await api.getDesign(photo.id));
  };

  const handlePatch = (id: string, patch: Partial<Region>) =>
    guard(async () => {
      if (!photo) return;
      const current = photo.regions.find((item) => item.id === id);
      if (current) {
        const before = Object.fromEntries(
          Object.keys(patch).map((key) => [key, current[key as keyof Region] ?? null]),
        ) as Partial<Region>;
        record({ type: "patch", id, before, after: patch });
      }
      await savePatch(id, patch);
    });

  // The × in the list and the Delete key do the same: the line or area goes, and nothing is
  // selected after it (selecting the next row made a second Delete remove something unseen).
  const handleDelete = (id: string) =>
    guard(async () => {
      if (!photo) return;
      const region = photo.regions.find((item) => item.id === id);
      // An area's painted mask, kept so Undo brings it back as it was.
      const mask = canvasRef.current?.maskData(id) ?? null;
      await api.deleteRegion(photo.id, id);
      const detail = await api.getPhoto(photo.id);
      setPhoto(detail);
      if (activeId === id) setActiveId(null);
      if (region) {
        record({ type: "exists", region, mask, present: false });
        say(`Deleted ${region.label}.`, { label: "Undo", run: () => void travelRef.current("undo") });
      }
    });

  // Puts one step back (undo) or forward again (redo); returns the step for the other stack.
  const applyStep = async (step: Step, direction: "undo" | "redo"): Promise<Step | null> => {
    if (!photo) return null;
    if (step.type === "canvas") {
      const canvas = canvasRef.current;
      if (!canvas) return null;
      const now = canvas.capture(step.snapshot.regionId, step.snapshot.type);
      await canvas.restore(step.snapshot);
      return now ? { type: "canvas", snapshot: now } : null;
    }
    if (step.type === "patch") {
      await savePatch(step.id, direction === "undo" ? step.before : step.after);
      return step;
    }
    const wanted = direction === "undo" ? !step.present : step.present;
    if (wanted) {
      await api.restoreRegion(photo.id, step.region, step.mask);
      setPhoto(await api.getPhoto(photo.id));
      setActiveId(step.region.id);
      if (design) setDesign(await api.getDesign(photo.id));
      return step;
    }
    // Taken away again: as it is now, edits since included, so a redo brings that back.
    const region = photo.regions.find((item) => item.id === step.region.id) ?? step.region;
    const mask = canvasRef.current?.maskData(region.id) ?? step.mask;
    await api.deleteRegion(photo.id, region.id);
    setPhoto(await api.getPhoto(photo.id));
    setActiveId((current) => (current === region.id ? null : current));
    if (design) setDesign(await api.getDesign(photo.id));
    return { ...step, region, mask };
  };

  // The toast's Undo runs after later renders: it calls the newest travel, with the newest photo.
  const travelRef = useRef<(direction: "undo" | "redo") => Promise<void>>(async () => undefined);
  const travel = async (direction: "undo" | "redo") => {
    const from = direction === "undo" ? past : future;
    const to = direction === "undo" ? future : past;
    if (travelling.current || busy || !photo || photo.approved || !from.current.length) return;
    const step = from.current.pop()!;
    travelling.current = true;
    reportHistory();
    try {
      const done = await guard(() => applyStep(step, direction));
      if (done === undefined) from.current.push(step);
      else if (done) to.current.push(done);
    } finally {
      travelling.current = false;
      reportHistory();
    }
  };
  travelRef.current = travel;

  // Asked first: detecting again replaces every detected line and area, and approving locks.
  const askRedetect = (allKinds: boolean) =>
    setConfirmation({
      title: allKinds ? "Detect every kind of area?" : "Detect again?",
      body:
        "Every line and area found automatically is replaced, so your changes to them are lost and undo starts over. " +
        "Lines and areas you drew stay." + (allKinds ? " Areas the instruction did not name are found too." : ""),
      confirm: allKinds ? "Detect everything" : "Detect again",
      danger: true,
      run: () => void handleRedetect(allKinds),
    });

  const askApprove = () =>
    setConfirmation({
      title: "Approve and lock this design?",
      body: "The lines, areas and lighting are locked so the quote cannot change by accident. You can reopen it from your account menu to correct it.",
      confirm: "Approve",
      run: () => void handleApprove(),
    });

  const toggleNight = (on: boolean) => {
    setNight(on);
    const lit = Object.values(placement?.regions ?? {}).some((placed) => placed.bulbs?.length);
    if (on && !lit) say("No lights are placed yet: measure the house and confirm the lighting to see them at night.");
  };

  const handleAdd = (kind: RegionKind) => {
    setTyping(null);
    if (geometryOf(kind) === "line") {
      setDrawingKind((current) => (current === kind ? null : kind));
      setTool("line");
      return;
    }
    setDrawingKind(null);
    void guard(async () => {
      if (!photo) return;
      const region = await api.createRegion(photo.id, { kind, geometry: "area" });
      setPhoto(await api.getPhoto(photo.id));
      setActiveId(region.id);
      setTool("brush");
      record({ type: "exists", region, mask: null, present: true });
    });
  };

  const handleDrawFinish = (polyline: number[][]) =>
    guard(async () => {
      if (!photo || !drawingKind) return;
      // No side sent: the server reads it from the house's vanishing points, snaps the ends onto
      // the lines they meet, and suggests the roof line's type for the picker.
      const region = await api.createRegion(photo.id, { kind: drawingKind, geometry: "line", polyline });
      setDrawingKind(null);
      setPhoto(await api.getPhoto(photo.id));
      record({ type: "exists", region, mask: null, present: true });
      setActiveId(region.id);
      setTool("vertex");
      if (region.kind === "roof") setTyping({ id: region.id, picked: false });
      if (design) setDesign(await api.getDesign(photo.id));
    });

  const chooseTool = (next: Tool) => {
    setDrawingKind(null);
    // Half-tapped measure points belong to the tool being left.
    if (next !== "measure") setMeasurePoints([]);
    if (next !== tool && hasGuide(next) && !guideSeen(next)) setGuide(next);
    setTool(next);
  };

  const handleApprove = () =>
    guard(async () => photo && setPhoto(await api.approvePhoto(photo.id)));

  const handleReopen = () =>
    guard(async () => photo && setPhoto(await api.reopenPhoto(photo.id)));

  // The key handler reads this render's state through a ref, so Delete removes what is selected
  // now, not what was selected when the listener was added.
  const keyRef = useRef<(event: KeyboardEvent) => void>(() => undefined);
  keyRef.current = (event: KeyboardEvent) => {
    // A dialog or detection in progress has the keyboard to itself.
    if (confirmation || shortcuts || detecting) return;
    const target = event.target;
    if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement
        || (target instanceof HTMLElement && target.isContentEditable)) return;
    if ((event.metaKey || event.ctrlKey) && (event.key.toLowerCase() === "z" || event.key.toLowerCase() === "y")) {
      event.preventDefault();
      void travel(event.shiftKey || event.key.toLowerCase() === "y" ? "redo" : "undo");
      return;
    }
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    if (event.key === "?") {
      event.preventDefault();
      setShortcuts(true);
      return;
    }
    if (event.key === "n" && photo?.mode) {
      toggleNight(!night);
      return;
    }
    if (event.key === "Delete" || event.key === "Backspace") {
      // The selected line or area goes, as with its × in the list. Never while drawing a line,
      // on a locked photo, or with the whole house selected (that is not one thing to delete).
      if (!photo || photo.approved || drawingKind || busy) return;
      if (activeId) {
        event.preventDefault();
        void handleDelete(activeId);
      } else if (houseSelected) {
        say("Select one line or area to delete it. The house itself stays; tap roofs out of it with the House tool.");
      }
      return;
    }
    if (event.key === "Escape" && !drawingKind) {
      selectRegion(null);
      return;
    }
    // chooseTool, not setTool: it also clears a half-drawn line, which setTool left stranded.
    if (event.key === "b") chooseTool("brush");
    if (event.key === "s") chooseTool("select");
    if (event.key === "e") chooseTool("erase");
    if (event.key === "v") chooseTool("vertex");
    if (event.key === "m") chooseTool("move");
    if (event.key === "h" && houseAvailable) chooseTool("house");
    if (event.key === "a") chooseTool("click");
    if (event.key === "f") chooseTool("find");
    if (event.key === "r") chooseTool("measure");
    if (event.key === "w") chooseTool("wreath");
    if (event.key === "[") setBrushSize((size) => Math.max(4, size - 6));
    if (event.key === "]") setBrushSize((size) => Math.min(200, size + 6));
  };
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => keyRef.current(event);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const active = photo?.regions.find((item) => item.id === activeId);
  // Every detection stores the roof planes; a photo detected before that has none to tap.
  const housePlanes = photo?.interpreted?.target?.planes ?? [];
  const houseAvailable = housePlanes.length > 0;
  const houseRoofs = houseAvailable ? housePlanes.filter((plane) => plane.keep).length : (photo?.interpreted?.planes?.length ?? 0);

  // Measure's panel sits above the list: opening Measure, or picking the line to measure, brings
  // it into view rather than leaving it scrolled away.
  const measurePanelRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (tool !== "measure") return;
    const frame = requestAnimationFrame(() => measurePanelRef.current?.scrollIntoView({ block: "start", behavior: "smooth" }));
    return () => cancelAnimationFrame(frame);
  }, [tool, activeId, measureMode]);
  // The picker belongs to the photo the line was drawn on.
  useEffect(() => setTyping(null), [photo?.id]);
  const typingRegion = (typing && !photo?.approved && photo?.regions.find((region) => region.id === typing.id)) || null;
  const requests = photo?.interpreted?.requests ?? [];
  const attention = requests.filter((r) => r.problems.length > 0 || !r.settings).length;

  // Whatever was confirmed for this photo before, e.g. when it is opened again.
  useEffect(() => {
    if (!photo?.id || !photo.mode) return;
    api.getDesign(photo.id).then(setDesign).catch(fail);
  }, [photo?.id, photo?.mode]);

  useEffect(() => {
    // Picking a region switches to a tool that can edit it, but never interrupts selecting.
    if (!active || drawingKind || tool === "select") return;
    if (active.geometry === "line" && tool !== "vertex" && tool !== "move" && tool !== "find")
      setTool("vertex");
    if (active.geometry === "area" && (tool === "vertex" || tool === "move")) setTool("brush");
  }, [activeId]);

  // A stage picked on one photo means nothing on a photo without that debug stage (another photo,
  // or one detected again with ROOFLINE_DEBUG off): the picker hides, so the choice goes too.
  const debugStages = photo?.interpreted?.roofline_debug?.stages;
  useEffect(() => {
    if (debugStage && !debugStages?.[debugStage]) setDebugStage(null);
  }, [debugStage, debugStages]);

  // Likewise a scale step, when the server stops returning debug data (SCALE_DEBUG off, another photo).
  const scaleDebug = geometry?.debug ?? null;
  useEffect(() => {
    if (scaleStep && !scaleDebug?.steps?.[scaleStep]) setScaleStep(null);
  }, [scaleStep, scaleDebug]);

  // The House tool's button is disabled when the photo has no planes to tap (detected before they
  // were stored); the tool goes with it rather than lingering unusable.
  useEffect(() => {
    if (tool === "house" && !houseAvailable) setTool("select");
  }, [tool, houseAvailable]);

  // A house selection and the undo history belong to the photo they were made on; the house gives
  // way to any line or area selected after it (a new object, a drawn line, a find).
  useEffect(() => {
    setHouseSelected(false);
    clearHistory();
  }, [photo?.id]);
  useEffect(() => {
    if (activeId) setHouseSelected(false);
  }, [activeId]);

  const routeNow = !user ? null : showProducts ? "products" : photo ? `photo/${photo.id}` : showUpload ? "new" : "projects";
  // Declared before the effect that reads the address at sign-in, so it never pushes over it.
  const routeReady = useRef(false);
  useEffect(() => {
    if (!routeNow || !routeReady.current) return;
    if (window.location.hash !== `#/${routeNow}`) window.history.pushState(null, "", `#/${routeNow}`);
  }, [routeNow]);
  const applyRoute = useRef<(route: Route) => void>(() => undefined);
  applyRoute.current = (route: Route) => {
    if (route.name === "products") {
      setShowProducts(true);
      return;
    }
    setShowProducts(false);
    if (route.name === "photo") {
      if (photo?.id !== route.id) {
        void handleOpenProject(route.id).then((opened) => {
          if (!opened) window.history.replaceState(null, "", "#/projects");
        });
      }
      return;
    }
    if (photo) rememberProject(photo);
    setPhoto(null);
    setActiveId(null);
    setDrawingKind(null);
    setDesign(null);
    setShowUpload(route.name === "new");
  };
  useEffect(() => {
    if (!user) {
      routeReady.current = false;
      return;
    }
    if (routeReady.current) return;
    routeReady.current = true;
    applyRoute.current(readRoute());
  }, [user?.id]);
  useEffect(() => {
    const onPop = () => routeReady.current && applyRoute.current(readRoute());
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  // Where the photo stands, step by step, for the header's progress bar.
  const roofLineRegions = photo?.regions.filter((r) => r.geometry === "line" && r.kind === "roof") ?? [];
  const untyped = roofLineRegions.filter((r) => r.edge === "unknown" && r.visible);
  const steps: ProgressStep[] = photo
    ? [
        {
          id: "detect",
          label: "Detect",
          state: photo.regions.length ? "done" : "todo",
          detail: photo.regions.length ? "Lines and areas found. Detect again from the panel to start over." : "Nothing found yet: detect the photo.",
          run: photo.approved ? undefined : () => askRedetect(false),
        },
        {
          id: "lines",
          label: "Check lines",
          state: roofLineRegions.length && !untyped.length ? "done" : "todo",
          detail: untyped.length
            ? `${untyped.length} roof ${untyped.length === 1 ? "line needs" : "lines need"} a type`
            : roofLineRegions.length
              ? "Every roof line has a type. Take a neighbour's roof out with House if its lines show."
              : "No roof lines yet: draw one with + Add, or detect again.",
          run: photo.approved
            ? undefined
            : () => {
                setSidebarTab("regions");
                setNight(false);
                chooseTool("select");
                if (untyped.length) selectRegion(untyped[0].id, true);
                else say("Tap a line to check it, Delete to remove it, and House to take a neighbour's roof out.");
              },
        },
        {
          id: "measure",
          label: "Measure",
          state: photo.scale?.measurement ? "done" : "todo",
          detail: photo.scale?.measurement ? "Scale set. Measure again to change it." : "Give the photo real scale from one known length.",
          run: photo.approved
            ? undefined
            : () => {
                setNight(false);
                chooseTool("measure");
              },
        },
        {
          id: "lighting",
          label: "Lighting",
          state: !requests.length ? "skip" : design ? "done" : "todo",
          detail: !requests.length
            ? "The instruction names no lighting."
            : design
              ? `Confirmed for ${design.regions.length} ${design.regions.length === 1 ? "line or area" : "lines and areas"}`
              : attention
                ? `${attention} to check before confirming`
                : "Confirm the lighting",
          run: () => {
            setSheetOpen(true);
            setSidebarTab("lighting");
          },
        },
        {
          id: "approve",
          label: photo.approved ? "Locked" : "Approve",
          state: photo.approved ? "done" : "todo",
          detail: photo.approved ? "Locked. Reopen it from your account menu to change it." : "Lock the design for the quote.",
          run: photo.approved ? undefined : askApprove,
        },
      ]
    : [];

  if (checkingSession) return <div className="app" />;

  if (!user) {
    return (
      <div className="app">
        <div className="topbar">
          <span className="brand">Light Design</span>
          <div className="spacer" />
          <ThemeToggle theme={theme} onChange={setTheme} />
        </div>
        <main className="entry">
          <Auth onSignedIn={setUser} />
        </main>
        <Activity />
      </div>
    );
  }

  return (
    <div className="app">
      <Activity />
      <Toasts notice={notice} error={error} onDismissNotice={() => setNotice(null)} onDismissError={() => setError(null)} />
      {confirmation && <ConfirmDialog confirmation={confirmation} onClose={() => setConfirmation(null)} />}
      {detecting && <DetectionProgress label={detecting.label} staged={detecting.staged} />}
      {shortcuts && <ShortcutSheet onClose={() => setShortcuts(false)} />}
      {typingRegion && (
        <EdgePicker
          region={typingRegion}
          suggested={!typing?.picked}
          onPick={(edge) => {
            setTyping({ id: typingRegion.id, picked: true });
            void handlePatch(typingRegion.id, { edge, ...(EDGE_COLORS[edge] ? { color: EDGE_COLORS[edge] } : {}) });
          }}
          onDone={() => setTyping(null)}
        />
      )}

      {(!(photo && photo.mode) || showProducts) && (
        <>
          <div className="account-bar">
            <span className="brand">Light Design</span>
            {(photo || showUpload) && !showProducts && (
              <button className="link" disabled={busy} onClick={() => goToProjects(false)}>
                ← Your projects
              </button>
            )}
            <div className="spacer" />
            <ThemeToggle theme={theme} onChange={setTheme} />
            <AccountMenu user={user} onOpenProducts={() => setShowProducts(true)} onSignOut={handleSignOut} />
          </div>
          <main className="entry">
            {showProducts ? (
              <Products
                palette={palette}
                onClose={() => setShowProducts(false)}
                onChanged={handleLibraryChanged}
              />
            ) : photo ? (
              <ModeChooser
                photo={photo}
                products={products}
                palette={palette}
                busy={busy}
                onInterpret={handleInterpret}
                onChoose={handleMode}
              />
            ) : showUpload ? (
              <Upload busy={busy} onSubmit={handleUpload} onCancel={() => setShowUpload(false)} />
            ) : (
              <Projects
                userId={user.id}
                busy={busy}
                onOpen={(id) => void handleOpenProject(id)}
                onNew={() => setShowUpload(true)}
                onConfirm={setConfirmation}
                onError={setError}
              />
            )}
          </main>
        </>
      )}

      {photo && photo.mode && !showProducts && (
        <div className="workspace">
          <header className="header">
            <nav className="breadcrumb" aria-label="Breadcrumb">
              <button className="crumb" disabled={busy} onClick={() => goToProjects(false)} title="All your projects">
                Projects
              </button>
              <span className="crumb-sep" aria-hidden="true">/</span>
              <ProjectName key={photo.id} photo={photo} busy={busy} onRename={(name) => void handleRename(name)} />
            </nav>
            <StepBar steps={steps} busy={busy} />
            <div className="header-actions">
              {!photo.approved && (
                <HistoryButtons
                  canUndo={history.canUndo && !busy}
                  canRedo={history.canRedo && !busy}
                  onUndo={() => void travel("undo")}
                  onRedo={() => void travel("redo")}
                />
              )}
              <ThemeToggle theme={theme} onChange={setTheme} />
              <AccountMenu
                user={user}
                onOpenProducts={() => setShowProducts(true)}
                onSignOut={handleSignOut}
                onReopen={photo.approved ? handleReopen : undefined}
              />
            </div>
          </header>

          <div className="stage">
            <div className="canvas-area">
              <MaskCanvas
                ref={canvasRef}
                photo={photo}
                regions={photo.regions}
                activeId={photo.approved ? null : activeId}
                tool={tool}
                brushSize={brushSize}
                zoom={zoom}
                showObstacles={showObstacles}
                onMaskCommit={handleMaskCommit}
                onPolylineCommit={handlePolylineCommit}
                onSelect={(id) => selectRegion(id, true)}
                houseSelected={houseSelected}
                onSelectHouse={selectHouse}
                onBox={handleBox}
                onPoint={handlePoint}
                onHouseTap={handleHouseTap}
                onRemoveAt={handleDelete}
                onRecord={(snapshot) => record({ type: "canvas", snapshot })}
                onZoom={setZoom}
                drawingColor={drawingKind ? KIND_COLORS[drawingKind] : null}
                onDrawFinish={handleDrawFinish}
                onDrawCancel={() => setDrawingKind(null)}
                debugStage={debugStage}
                measureMode={measureMode}
                measurePoints={measurePoints}
                onMeasureTap={handleMeasureTap}
                overrideLine={
                  active?.geometry === "line" && !photo.approved && needsOverride(geometry?.lines[active.id])
                    ? { regionId: active.id, mid: photo.scale?.overrides?.[active.id]?.mid ?? null }
                    : null
                }
                onOverrideMid={(regionId, mid) => handleOverride(regionId, { mid })}
                bulbs={placement?.regions}
                onWreathTap={handleWreathTap}
                geometry={geometry?.lines}
                showLengths={showLengths && Boolean(photo.scale?.measurement)}
                backdrop={scaleDebug && scaleStep ? stepFile(scaleDebug, scaleStep) : null}
                night={night}
              />
              {!night && <CanvasLegend regions={photo.regions} />}
              {photo.interpreted?.roofline_debug?.stages && (
                <label className="debug-stage" title="Development: the roof lines at one stage of detection">
                  <span>Stage</span>
                  <select value={debugStage ?? ""} onChange={(event) => setDebugStage(event.target.value || null)}>
                    <option value="">Off</option>
                    {Object.keys(photo.interpreted.roofline_debug.stages).map((stage) => (
                      <option key={stage} value={stage}>
                        {stage.replace(/_/g, " ")}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              {!photo.approved && !night && (
                <ToolDock tool={tool} setTool={chooseTool} brushSize={brushSize} setBrushSize={setBrushSize}
                          active={active} houseAvailable={houseAvailable}
                          onHelp={hasGuide(tool) ? () => setGuide(tool) : undefined}
                          onShortcuts={() => setShortcuts(true)} />
              )}
              {(
                <>
                  <ViewportControls
                    zoom={zoom}
                    setZoom={setZoom}
                    night={{ on: night, set: toggleNight }}
                    lengths={photo.scale?.measurement ? { on: showLengths, set: setShowLengths } : null}
                  />
                </>
              )}
            </div>
            <aside className={sheetOpen ? "sidebar" : "sidebar collapsed"}>
              <button
                className="sheet-handle"
                aria-expanded={sheetOpen}
                onClick={() => setSheetOpen(!sheetOpen)}
                title={sheetOpen ? "Fold the panel down to see more of the photo" : "Open the panel"}
              >
                <span className="sheet-grip" aria-hidden="true" />
                {sheetOpen ? "Hide panel" : "Lines, areas & lighting"}
              </button>
              <div className="tabs" role="tablist">
                <button
                  role="tab"
                  aria-selected={sidebarTab === "regions"}
                  className={sidebarTab === "regions" ? "tab active" : "tab"}
                  onClick={() => setSidebarTab("regions")}
                >
                  Lines & areas
                </button>
                <button
                  role="tab"
                  aria-selected={sidebarTab === "lighting"}
                  className={sidebarTab === "lighting" ? "tab active" : "tab"}
                  onClick={() => setSidebarTab("lighting")}
                >
                  Lighting
                  {!design && requests.length > 0 && <span className="tab-dot" title="Not confirmed yet" />}
                </button>
              </div>
              <div className="sidebar-body">
                {scaleDebug && (
                  <ScaleDebugPanel
                    debug={scaleDebug}
                    step={scaleStep}
                    onStep={setScaleStep}
                    activeId={activeId}
                    onSelect={setActiveId}
                  />
                )}
                {tool === "measure" && !photo.approved && (
                  <div ref={measurePanelRef}>
                  <MeasurePanel
                    photo={photo}
                    mode={measureMode}
                    setMode={(mode) => {
                      setMeasureMode(mode);
                      setMeasurePoints([]);
                    }}
                    points={measurePoints}
                    onResetPoints={() => setMeasurePoints([])}
                    line={active?.geometry === "line" ? active : null}
                    busy={busy}
                    onSave={handleMeasureSave}
                    onClear={() => handleMeasureSave(null)}
                    onRemeasure={handleRemeasure}
                    onCheck={handleCheck}
                    checkResult={geometry?.reference_check ?? null}
                    referenceQuality={geometry?.reference_quality ?? null}
                    scaleCheck={geometry?.scale_check ?? null}
                    nextMeasurement={geometry?.next_measurement ?? null}
                    onSelectLine={(id) => {
                      setMeasureMode("line");
                      setMeasurePoints([]);
                      selectRegion(id);
                    }}
                  />
                  </div>
                )}
                {sidebarTab === "regions" ? (
                  <RegionPanel
                    regions={photo.regions}
                    activeId={activeId}
                    drawingKind={drawingKind}
                    showObstacles={showObstacles}
                    onShowObstacles={setShowObstacles}
                    onSelect={(id) => selectRegion(id)}
                    house={
                      houseSelected
                        ? {
                            roofs: houseRoofs,
                            onChangeRoofs: houseAvailable && !photo.approved ? () => chooseTool("house") : undefined,
                            onClear: () => setHouseSelected(false),
                          }
                        : null
                    }
                    follow={tool !== "measure"}
                    onPatch={handlePatch}
                    onDelete={handleDelete}
                    onAdd={(kind) => {
                      handleAdd(kind);
                    }}
                    geometry={geometry?.lines}
                    placement={placement?.regions}
                    measurement={photo.scale?.measurement ?? null}
                    referenceEdited={Boolean(geometry?.reference_edited)}
                    scaleWarning={geometry?.warning ?? null}
                    scaleChecks={geometry?.checks ?? []}
                    referenceSide={geometry?.reference?.facade ?? null}
                    lens={geometry?.camera.focal_source ?? photo.interpreted?.camera?.focal_source ?? null}
                    onMeasure={photo.approved ? undefined : () => chooseTool("measure")}
                    onRemeasure={
                      photo.approved || !photo.scale?.measurement
                        ? undefined
                        : () => handleRemeasure(photo.scale!.measurement!)
                    }
                    onOverride={photo.approved ? undefined : handleOverride}
                  />
                ) : requests.length > 0 ? (
                  <Specification
                    requests={requests}
                    products={products}
                    palette={palette}
                    regions={photo.regions}
                    design={design}
                    busy={busy}
                    onConfirm={handleConfirmLighting}
                  />
                ) : (
                  <p className="muted small">
                    The instruction names no lighting to set. Describe it on the start screen, for
                    example "C9 warm white on the roofline".
                  </p>
                )}
              </div>
              {!photo.approved && (
                <div className="sidebar-footer">
                  <button
                    title="Detect again from the instruction. Replaces every detected line and area; ones you drew stay."
                    disabled={busy}
                    onClick={() => askRedetect(false)}
                  >
                    Re-detect
                  </button>
                  <button
                    title="Detect every kind of area, not only the ones the instruction names"
                    disabled={busy}
                    onClick={() => askRedetect(true)}
                  >
                    Detect everything
                  </button>
                  <button className="primary" title="Lock these lines and areas" disabled={busy} onClick={askApprove}>
                    Approve
                  </button>
                </div>
              )}
            </aside>
          </div>
        </div>
      )}
      {guide && <ToolGuide tool={guide} onClose={closeGuide} />}
    </div>
  );
}
