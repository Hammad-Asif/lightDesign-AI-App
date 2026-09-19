import { useCallback, useEffect, useRef, useState } from "react";

import * as api from "./api";
import { MaskCanvas, type MaskCanvasHandle } from "./canvas/MaskCanvas";
import { Auth } from "./components/Auth";
import { ModeChooser } from "./components/ModeChooser";
import { ThemeToggle, type Theme } from "./components/ThemeToggle";
import { RegionPanel } from "./components/RegionPanel";
import { Toolbar } from "./components/Toolbar";
import { Upload } from "./components/Upload";
import { Specification } from "./components/Specification";
import { BLOCKER_KINDS, geometryOf, KIND_COLORS } from "./types";
import type {
  Design,
  DesignRequest,
  MaskingMode,
  PaletteColour,
  PhotoDetail,
  Product,
  Region,
  RegionKind,
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
  const [drawingKind, setDrawingKind] = useState<RegionKind | null>(null);
  const [products, setProducts] = useState<Product[]>([]);
  const [palette, setPalette] = useState<PaletteColour[]>([]);
  // The photo's confirmed lighting, if any; null until the user confirms it.
  const [design, setDesign] = useState<Design | null>(null);
  const [showLighting, setShowLighting] = useState(false);

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
    setShowLighting(false);
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
      setActiveId(detail.regions[0]?.id ?? null);
      setTool(detail.regions[0]?.geometry === "line" ? "vertex" : "brush");
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
      // Repainting something in front of the house changes what the lines behind it hide.
      if (BLOCKER_KINDS.includes(region.kind)) setPhoto(await api.getPhoto(photo!.id));
      else mergeRegion(region);
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

  const handleBox = (box: number[]) =>
    guard(async () => {
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
    if (active.geometry === "line" && tool !== "vertex" && tool !== "move" && tool !== "box")
      setTool("vertex");
    if (active.geometry === "area" && (tool === "vertex" || tool === "move")) setTool("brush");
  }, [activeId]);

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

      {!(photo && photo.mode) && (
        <>
          <div className="account-bar">
            <span className="brand">Light Design</span>
            <div className="spacer" />
            <span className="muted small">{user.email}</span>
            <ThemeToggle theme={theme} onChange={setTheme} />
            <button onClick={handleSignOut}>Sign out</button>
          </div>
          <main className="entry">
            {photo ? (
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

      {photo && photo.mode && (
        <div className="workspace">
          <header className="header">
            <div className="header-title">
              <strong>Light Design</strong>
              {photo.approved ? (
                <span className="badge ok">Locked</span>
              ) : (
                <span className="badge">Draft</span>
              )}
              <span className="muted small">
                {photo.regions.length} {photo.regions.length === 1 ? "region" : "regions"}
              </span>
            </div>
            <div className="header-actions">
              <span className="muted small">{user.email}</span>
              <ThemeToggle theme={theme} onChange={setTheme} />
              {photo.approved && <button onClick={handleReopen}>Reopen for correction</button>}
              <button
                onClick={() => {
                  setPhoto(null);
                  setActiveId(null);
                }}
              >
                New photo
              </button>
              <button onClick={handleSignOut}>Sign out</button>
            </div>
          </header>

          {!photo.approved && (
            <Toolbar
              tool={tool}
              setTool={chooseTool}
              brushSize={brushSize}
              setBrushSize={setBrushSize}
              zoom={zoom}
              setZoom={setZoom}
              active={active}
              history={history}
              busy={busy}
              onUndo={() => canvasRef.current?.undo()}
              onRedo={() => canvasRef.current?.redo()}
              onRedetect={() => handleRedetect(false)}
              onApprove={handleApprove}
            />
          )}

          {photo.interpreted && (
            <div className="interpreted">
              <span>
                {requests.length
                  ? `Instruction read as ${requests.length} ${requests.length === 1 ? "request" : "requests"}` +
                    (attention ? `, ${attention} to check` : "") +
                    (design ? ` · lighting confirmed for ${design.regions.length} ${design.regions.length === 1 ? "region" : "regions"}` : " · lighting not confirmed")
                  : photo.interpreted.matched
                    ? "Masking only what the instruction named."
                    : "No lighting areas recognised in the instruction — masking everything."}
              </span>
              <span className="interpreted-actions">
                {requests.length > 0 && (
                  <button onClick={() => setShowLighting((open) => !open)}>
                    {showLighting ? "Hide lighting" : design ? "Lighting" : "Review lighting"}
                  </button>
                )}
                {!photo.approved && (
                  <button onClick={() => handleRedetect(true)} disabled={busy}>
                    Detect all region types
                  </button>
                )}
              </span>
            </div>
          )}

          {showLighting && requests.length > 0 && (
            <div className="spec-bar">
              <Specification
                requests={requests}
                products={products}
                palette={palette}
                regions={photo.regions}
                design={design}
                busy={busy}
                onConfirm={handleConfirmLighting}
              />
            </div>
          )}

          <div className="stage">
            <MaskCanvas
              ref={canvasRef}
              photo={photo}
              regions={photo.regions}
              activeId={photo.approved ? null : activeId}
              tool={tool}
              brushSize={brushSize}
              zoom={zoom}
              onMaskCommit={handleMaskCommit}
              onPolylineCommit={handlePolylineCommit}
              onSelect={setActiveId}
              onBox={handleBox}
              onPoint={handlePoint}
              onRemoveAt={handleDelete}
              onHistoryChange={setHistory}
              drawingColor={drawingKind ? KIND_COLORS[drawingKind] : null}
              onDrawFinish={handleDrawFinish}
              onDrawCancel={() => setDrawingKind(null)}
            />
            <RegionPanel
              regions={photo.regions}
              activeId={activeId}
              drawingKind={drawingKind}
              onSelect={setActiveId}
              onPatch={handlePatch}
              onDelete={handleDelete}
              onAdd={handleAdd}
            />
          </div>
        </div>
      )}
    </div>
  );
}
