import { useCallback, useEffect, useRef, useState } from "react";

import * as api from "./api";
import { MaskCanvas, type MaskCanvasHandle } from "./canvas/MaskCanvas";
import { AccountMenu } from "./components/AccountMenu";
import { Auth } from "./components/Auth";
import { ModeChooser } from "./components/ModeChooser";
import { Products } from "./components/Products";
import { ThemeToggle, type Theme } from "./components/ThemeToggle";
import { needsOverride, RegionPanel } from "./components/RegionPanel";
import { ToolDock, ViewportControls } from "./components/Toolbar";
import { Upload } from "./components/Upload";
import { Specification } from "./components/Specification";
import { formatLength, MeasurePanel, type MeasureMode } from "./components/MeasurePanel";
import { guideSeen, hasGuide, markGuideSeen, ToolGuide } from "./components/ToolGuide";
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
  ScaleOverride,
  Tool,
  User,
} from "./types";

export default function App() {
  const [user, setUser] = useState<User | null>(null);
  const [checkingSession, setCheckingSession] = useState(true);
  // index.html picks the first theme before paint; this keeps it in step after that.
  const [theme, setTheme] = useState<Theme>(
    () => (document.documentElement.dataset.theme as Theme) || "light",
  );
  const [photo, setPhoto] = useState<PhotoDetail | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [tool, setTool] = useState<Tool>("select");
  const [brushSize, setBrushSize] = useState(40);
  const [zoom, setZoom] = useState(1);
  const [history, setHistory] = useState({ canUndo: false, canRedo: false });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // A neutral message, e.g. that Find in Box found nothing new. Not an error.
  const [notice, setNotice] = useState<string | null>(null);
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
  const [showObstacles, setShowObstacles] = useState(false);
  // Measure tool: two points tapped on a feature, or a line picked on the photo.
  const [measureMode, setMeasureMode] = useState<MeasureMode>("points");
  const [measurePoints, setMeasurePoints] = useState<number[][]>([]);
  // Every roof line's real length and how it was placed; recomputed on the server after any change.
  const [geometry, setGeometry] = useState<PhotoGeometry | null>(null);
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

  const handleUpload = (file: File, prompt: string) =>
    guard(async () => {
      const uploaded = await api.uploadPhoto(file, prompt);
      setDesign(null);
      setPhoto(await api.getPhoto(uploaded.id));
    });

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
      if (mode === "automatic") await api.detectRegions(photo.id);
      const detail = await api.getPhoto(photo.id);
      setPhoto(detail);
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
      await api.detectRegions(photo.id, { allKinds });
      setPhoto(await api.getPhoto(photo.id));
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

  // Find in Box: detection runs inside the box and only adds lines; the new ones are selected.
  const handleFindInBox = (box: number[]) =>
    guard(async () => {
      if (!photo) return;
      setNotice(null);
      const found = await api.findLinesInBox(photo.id, box);
      setPhoto(await api.getPhoto(photo.id));
      if (found.length) setActiveId(found[0].id);
      else setNotice("No new roof lines found in that box. Try a box closer around the missing roof edge.");
    });

  const handleBox = (box: number[]) =>
    tool === "find"
      ? handleFindInBox(box)
      : guard(async () => {
          if (!photo) return;
          const active = photo.regions.find((item) => item.id === activeId);
          const region = await api.refineRegion(photo.id, {
            box,
            region_id: active?.geometry === "area" ? active.id : undefined,
            kind: active?.geometry === "area" ? active.kind : "custom",
          });
          setPhoto(await api.getPhoto(photo.id));
          setActiveId(region.id);
          setTool("brush");
        });

  const handlePoint = (point: number[]) =>
    guard(async () => {
      if (!photo) return;
      const region = await api.refineRegion(photo.id, { point, kind: "custom" });
      setPhoto(await api.getPhoto(photo.id));
      setActiveId(region.id);
    });

  // A tap with the House tool redraws the detected roof lines, so a selected line, and the undo
  // steps of the old lines, may be gone; confirmed lighting moves to the new lines on the server.
  // One tap at a time: each reads the house the last one left. A locked photo's lines stay put.
  const handleHouseTap = (point: number[]) => {
    if (!photo || photo.approved || busy) return;
    void guard(async () => {
      const detail = await api.tapHousePlane(photo.id, point);
      canvasRef.current?.forget(
        photo.regions.filter((r) => !detail.regions.some((d) => d.id === r.id)).map((r) => r.id),
      );
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
      const side = saved.scale?.measurement?.facade ?? "front";
      setNotice(
        `Scale set from ${formatLength(measurement.length_in)} on the ${side} side. ` +
          `${measured} ${measured === 1 ? "line" : "lines"} measured` +
          (waiting ? `; ${waiting} ${waiting === 1 ? "needs" : "need"} attention (marked ? on the photo).` : "."),
      );
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
      setActiveId(detail.regions[detail.regions.length - 1]?.id ?? null);
    });
  };

  const handlePatch = (id: string, patch: Partial<Region>) =>
    guard(async () => {
      if (!photo) return;
      mergeRegion({ ...photo.regions.find((item) => item.id === id)!, ...patch } as Region);
      const saved = await api.patchRegion(photo.id, id, patch);
      // A kind change can turn an area into, or out of, something lines pass behind.
      if (patch.kind) setPhoto(await api.getPhoto(photo.id));
      else mergeRegion(saved);
    });

  const handleDelete = (id: string) =>
    guard(async () => {
      if (!photo) return;
      await api.deleteRegion(photo.id, id);
      const detail = await api.getPhoto(photo.id);
      setPhoto(detail);
      if (activeId === id) setActiveId(detail.regions[0]?.id ?? null);
    });

  const handleAdd = (kind: RegionKind) => {
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
    });
  };

  const handleDrawFinish = (polyline: number[][]) =>
    guard(async () => {
      if (!photo || !drawingKind) return;
      const region = await api.createRegion(photo.id, {
        kind: drawingKind,
        geometry: "line",
        side: "front",
        polyline,
      });
      setDrawingKind(null);
      setPhoto(await api.getPhoto(photo.id));
      setActiveId(region.id);
      setTool("vertex");
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

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) return;
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "z") {
        event.preventDefault();
        event.shiftKey ? canvasRef.current?.redo() : canvasRef.current?.undo();
        return;
      }
      // chooseTool, not setTool: it also clears a half-drawn line, which setTool left stranded.
      if (event.key === "b") chooseTool("brush");
      if (event.key === "s") chooseTool("select");
      if (event.key === "e") chooseTool("erase");
      if (event.key === "v") chooseTool("vertex");
      if (event.key === "m") chooseTool("move");
      if (event.key === "[") setBrushSize((size) => Math.max(4, size - 6));
      if (event.key === "]") setBrushSize((size) => Math.min(200, size + 6));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const active = photo?.regions.find((item) => item.id === activeId);
  // Only a photo detected with TARGET_HOUSE_ONLY on has roof planes to tap.
  const houseAvailable = Boolean(photo?.interpreted?.target?.planes?.length);
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
    if (active.geometry === "line" && tool !== "vertex" && tool !== "move" && tool !== "box" && tool !== "find")
      setTool("vertex");
    if (active.geometry === "area" && (tool === "vertex" || tool === "move")) setTool("brush");
  }, [activeId]);

  // A stage picked on one photo means nothing on a photo without that debug stage (another photo,
  // or one detected again with ROOFLINE_DEBUG off): the picker hides, so the choice goes too.
  const debugStages = photo?.interpreted?.roofline_debug?.stages;
  useEffect(() => {
    if (debugStage && !debugStages?.[debugStage]) setDebugStage(null);
  }, [debugStage, debugStages]);

  // The House tool's button goes when the photo has no planes to tap (another photo, or one
  // detected again with the flag off); the tool goes with it rather than lingering unseen.
  useEffect(() => {
    if (tool === "house" && !houseAvailable) setTool("select");
  }, [tool, houseAvailable]);

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
      </div>
    );
  }

  return (
    <div className="app">
      {error && (
        <div className="error" onClick={() => setError(null)}>
          {error}
        </div>
      )}
      {notice && !error && (
        <div className="notice" onClick={() => setNotice(null)}>
          {notice}
        </div>
      )}

      {(!(photo && photo.mode) || showProducts) && (
        <>
          <div className="account-bar">
            <span className="brand">Light Design</span>
            <div className="spacer" />
            <AccountMenu
              user={user}
              theme={theme}
              onThemeChange={setTheme}
              onOpenProducts={() => setShowProducts(true)}
              onSignOut={handleSignOut}
            />
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
            ) : (
              <Upload busy={busy} onSubmit={handleUpload} />
            )}
          </main>
        </>
      )}

      {photo && photo.mode && !showProducts && (
        <div className="workspace">
          <header className="header">
            <div className="header-title">
              <strong>Light Design</strong>
              {photo.approved ? (
                <span className="badge ok">Locked</span>
              ) : (
                <span className="badge">Draft</span>
              )}
              {photo.interpreted && (
                <button
                  className={
                    requests.length
                      ? design
                        ? "badge ok status"
                        : "badge warn status"
                      : photo.interpreted.matched
                        ? "badge status"
                        : "badge warn status"
                  }
                  title="Open the lighting for this photo"
                  onClick={() => setSidebarTab("lighting")}
                >
                  {requests.length
                    ? design
                      ? `Lighting confirmed for ${design.regions.length} ${design.regions.length === 1 ? "line or area" : "lines and areas"}`
                      : attention
                        ? `Lighting: ${attention} to check`
                        : "Lighting not confirmed yet"
                    : photo.interpreted.matched
                      ? "Masking only what the instruction named"
                      : "No lighting recognised in the instruction"}
                </button>
              )}
            </div>
            <div className="header-actions">
              <AccountMenu
                user={user}
                theme={theme}
                onThemeChange={setTheme}
                onOpenProducts={() => setShowProducts(true)}
                onSignOut={handleSignOut}
                onReopen={photo.approved ? handleReopen : undefined}
                onNewPhoto={() => {
                  setPhoto(null);
                  setActiveId(null);
                }}
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
                onSelect={setActiveId}
                onBox={handleBox}
                onPoint={handlePoint}
                onHouseTap={handleHouseTap}
                onRemoveAt={handleDelete}
                onHistoryChange={setHistory}
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
              />
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
              {!photo.approved && (
                <>
                  <ToolDock tool={tool} setTool={chooseTool} brushSize={brushSize} setBrushSize={setBrushSize}
                            active={active} houseAvailable={houseAvailable}
                            onHelp={hasGuide(tool) ? () => setGuide(tool) : undefined} />
                  <ViewportControls
                    zoom={zoom}
                    setZoom={setZoom}
                    history={history}
                    onUndo={() => canvasRef.current?.undo()}
                    onRedo={() => canvasRef.current?.redo()}
                    lengths={photo.scale?.measurement ? { on: showLengths, set: setShowLengths } : null}
                  />
                </>
              )}
            </div>
            <aside className="sidebar">
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
                {tool === "measure" && !photo.approved && (
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
                  />
                )}
                {sidebarTab === "regions" ? (
                  <RegionPanel
                    regions={photo.regions}
                    activeId={activeId}
                    drawingKind={drawingKind}
                    showObstacles={showObstacles}
                    onShowObstacles={setShowObstacles}
                    onSelect={setActiveId}
                    onPatch={handlePatch}
                    onDelete={handleDelete}
                    onAdd={(kind) => {
                      handleAdd(kind);
                    }}
                    geometry={geometry?.lines}
                    placement={placement?.regions}
                    measurement={photo.scale?.measurement ?? null}
                    referenceEdited={Boolean(geometry?.reference_edited)}
                    lens={geometry?.camera.focal_source ?? photo.interpreted?.camera?.focal_source ?? null}
                    onMeasure={photo.approved ? undefined : () => chooseTool("measure")}
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
                    onClick={() => handleRedetect(false)}
                  >
                    Re-detect
                  </button>
                  <button
                    title="Detect every kind of area, not only the ones the instruction names"
                    disabled={busy}
                    onClick={() => handleRedetect(true)}
                  >
                    Detect everything
                  </button>
                  <button className="primary" title="Lock these lines and areas" disabled={busy} onClick={handleApprove}>
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
