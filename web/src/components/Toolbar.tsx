import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent, type ReactNode } from "react";

import type { Region, Tool } from "../types";

type ToolSpec = { id: Tool; label: string; key?: string; hint: string; needs?: "line" | "area"; icon: ReactNode };

function svg(children: ReactNode) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {children}
    </svg>
  );
}

// Grouped by what they act on: picking what is there, adding something new from the photo,
// painting an area, shaping a line, and measuring or placing.
const GROUPS: ToolSpec[][] = [
  [
    {
      id: "select",
      label: "Select",
      key: "S",
      hint: "Pick something already on the photo: a line, an area, or the house (tap its roof). Tap empty space to clear. Delete removes the picked line or area.",
      icon: svg(<path d="M5 3l15 8.5-6.4 1.4L11 20z" />),
    },
    {
      id: "house",
      label: "House",
      key: "H",
      hint: "Tap a roof to add it to the house being quoted, or tap it again to take it out. The roof lines are redrawn.",
      icon: svg(
        <>
          <path d="M4 10.5 12 4l8 6.5" />
          <path d="M6 9.5V20h12V9.5" />
        </>,
      ),
    },
  ],
  [
    {
      id: "click",
      label: "Add object",
      key: "A",
      hint: "Tap something new on the photo (a hedge, a tree, a window) to outline it as a new area. Alt-click an outline to remove it.",
      icon: svg(
        <>
          <circle cx="12" cy="12" r="7.5" strokeDasharray="3 2.5" />
          <path d="M12 8.5v7M8.5 12h7" />
        </>,
      ),
    },
    {
      id: "find",
      label: "Find lines",
      key: "F",
      hint: "Drag a box where roof lines were missed; the lines inside it are detected and added. Nothing already there changes.",
      icon: svg(
        <>
          <rect x="3.5" y="3.5" width="12" height="12" rx="1.5" strokeDasharray="3 2.5" />
          <circle cx="16" cy="16" r="3.2" />
          <path d="m18.4 18.4 2.6 2.6" />
        </>,
      ),
    },
  ],
  [
    {
      id: "brush",
      label: "Brush",
      key: "B",
      hint: "Paint onto the selected area",
      needs: "area",
      icon: svg(
        <>
          <path d="M9.5 14.5 17 7a2 2 0 0 1 3 3l-7.5 7.5" />
          <path d="M9.5 14.5c.6 1.8-.2 3.3-1.7 4-1.3.6-3 .6-4.3 0 .9-.4 1-1.4 1-2.3 0-1.6.9-2.8 2.3-3.3.9-.3 2 0 2.7.6Z" />
        </>,
      ),
    },
    {
      id: "erase",
      label: "Erase",
      key: "E",
      hint: "Remove from the selected area",
      needs: "area",
      icon: svg(
        <>
          <path d="M18.5 12.5 11 20H5l-1.5-1.5a2 2 0 0 1 0-2.8l9-9a2 2 0 0 1 2.8 0l3.2 3.2a2 2 0 0 1 0 2.6Z" />
          <path d="M8 20h11" />
        </>,
      ),
    },
  ],
  [
    {
      id: "vertex",
      label: "Points",
      key: "V",
      hint: "Drag a point; click the line to add one; Alt-click a point to delete it",
      needs: "line",
      icon: svg(
        <>
          <path d="M4 18 10 8l5 5 5-9" />
          <circle cx="10" cy="8" r="1.6" fill="currentColor" stroke="none" />
          <circle cx="15" cy="13" r="1.6" fill="currentColor" stroke="none" />
        </>,
      ),
    },
    {
      id: "move",
      label: "Move / extend",
      key: "M",
      hint: "Drag the line to move it, or drag an end to lengthen or shorten it",
      needs: "line",
      icon: svg(
        <>
          <path d="M5 12h14" />
          <path d="M5 12 8 9M5 12l3 3M19 12l-3-3M19 12l-3 3" />
        </>,
      ),
    },
  ],
  [
    {
      id: "measure",
      label: "Measure",
      key: "R",
      hint: "Give the photo real scale: tap both ends of a feature you know the size of, or pick an eave, and type its length.",
      icon: svg(
        <>
          <path d="m3.5 16.5 13-13 4 4-13 13z" />
          <path d="m7 13 1.5 1.5M9.5 10.5 11 12M12 8l1.5 1.5M14.5 5.5 16 7" />
        </>,
      ),
    },
    {
      id: "wreath",
      label: "Wreath",
      key: "W",
      hint: "Tap where a wreath goes. It is drawn at its real size (36 in; change it in the list). Needs a measurement first.",
      icon: svg(
        <>
          <circle cx="12" cy="12" r="7" />
          <circle cx="12" cy="12" r="3.5" />
        </>,
      ),
    },
  ],
];

interface DockProps {
  tool: Tool;
  setTool: (tool: Tool) => void;
  brushSize: number;
  setBrushSize: (size: number) => void;
  active: Region | undefined;
  // The photo has stored roof planes to tap in or out. Every detection stores them now; a photo
  // detected before that has none, and the tool stays disabled until it is detected again.
  houseAvailable: boolean;
  // Opens the current tool's how-to card.
  onHelp?: () => void;
  // Opens the keyboard shortcuts list.
  onShortcuts?: () => void;
}

const DOCK_KEY = "ld-dock-position";

function savedPosition(): { x: number; y: number } | null {
  try {
    const saved = JSON.parse(localStorage.getItem(DOCK_KEY) ?? "null");
    return saved && Number.isFinite(saved.x) && Number.isFinite(saved.y) ? saved : null;
  } catch {
    return null;
  }
}

// A floating vertical dock over the canvas, so the tools that act on it read as part of it rather
// than a disconnected bar of text buttons above. Dragged by its grip to wherever it hides least of
// the house; the place is remembered in this browser, and a double-click on the grip puts it back.
export function ToolDock({ tool, setTool, brushSize, setBrushSize, active, houseAvailable, onHelp, onShortcuts }: DockProps) {
  // Why a tool cannot be used yet, said in the dock on a tap (a tooltip never shows on a tablet),
  // and what the current tool is waiting for.
  const [note, setNote] = useState<string | null>(null);
  useEffect(() => {
    if (!note) return;
    const timer = window.setTimeout(() => setNote(null), 4500);
    return () => window.clearTimeout(timer);
  }, [note]);
  const current = GROUPS.flat().find((item) => item.id === tool);
  const waiting = current?.needs && active?.geometry !== current.needs
    ? `Select ${current.needs === "line" ? "a line" : "an area"} on the photo to use ${current.label}.`
    : null;
  const dockRef = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState<{ x: number; y: number } | null>(savedPosition);
  const dragRef = useRef<{ dx: number; dy: number } | null>(null);

  // Kept inside the canvas area: on a drag, and when the window (so the area) gets smaller.
  const clamp = (x: number, y: number) => {
    const dock = dockRef.current;
    const area = dock?.parentElement;
    if (!dock || !area) return { x, y };
    return {
      x: Math.round(Math.min(Math.max(0, x), Math.max(0, area.clientWidth - dock.offsetWidth))),
      y: Math.round(Math.min(Math.max(0, y), Math.max(0, area.clientHeight - dock.offsetHeight))),
    };
  };
  useEffect(() => {
    if (!position) return;
    const refit = () => setPosition((current) => (current ? clamp(current.x, current.y) : current));
    refit();
    window.addEventListener("resize", refit);
    return () => window.removeEventListener("resize", refit);
  }, [position === null]);
  useEffect(() => {
    try {
      if (position) localStorage.setItem(DOCK_KEY, JSON.stringify(position));
      else localStorage.removeItem(DOCK_KEY);
    } catch {
      // Storage blocked: the dock still moves, it just starts at its corner next time.
    }
  }, [position]);

  const onGripDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    const dock = dockRef.current;
    if (!dock) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = { dx: event.clientX - dock.offsetLeft, dy: event.clientY - dock.offsetTop };
  };
  const onGripMove = (event: ReactPointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (drag) setPosition(clamp(event.clientX - drag.dx, event.clientY - drag.dy));
  };
  const onGripUp = (event: ReactPointerEvent<HTMLDivElement>) => {
    dragRef.current = null;
    event.currentTarget.releasePointerCapture(event.pointerId);
  };

  return (
    <div
      ref={dockRef}
      className={position ? "tool-dock moved" : "tool-dock"}
      style={position ? { left: position.x, top: position.y } : undefined}
      role="toolbar"
      aria-label="Editing tools"
      aria-orientation="vertical"
    >
      <div
        className="dock-grip"
        title="Drag to move the tools; double-click to put them back"
        onPointerDown={onGripDown}
        onPointerMove={onGripMove}
        onPointerUp={onGripUp}
        onDoubleClick={() => setPosition(null)}
      >
        <svg viewBox="0 0 24 8" aria-hidden="true">
          {[4, 9, 14, 19].flatMap((x) => [2, 6].map((y) => <circle key={`${x}-${y}`} cx={x + 0.5} cy={y} r="1.3" />))}
        </svg>
      </div>
      {GROUPS.map((group, index) => (
        <div key={index} className="dock-group">
          {group.map((item) => {
            const missing = item.needs && active?.geometry !== item.needs;
            const noPlanes = item.id === "house" && !houseAvailable;
            const hint = missing
              ? `Select ${item.needs === "line" ? "a line" : "an area"} first`
              : noPlanes
                ? "This photo has no roofs to pick yet: press Re-detect, then tap roofs in or out of the house"
                : item.key
                  ? `${item.hint} (${item.key})`
                  : item.hint;
            const unavailable = Boolean(missing || noPlanes);
            return (
              <button
                key={item.id}
                title={hint}
                aria-pressed={tool === item.id}
                aria-disabled={unavailable || undefined}
                aria-keyshortcuts={item.key}
                className={`dock-tool${tool === item.id ? " active" : ""}${unavailable ? " unavailable" : ""}`}
                onClick={() => (unavailable ? setNote(hint) : setTool(item.id))}
              >
                {item.icon}
                <span>{item.label}</span>
                {item.key && <kbd className="dock-key" aria-hidden="true">{item.key}</kbd>}
              </button>
            );
          })}
        </div>
      ))}

      {(note || waiting) && (
        <p className="dock-note" role="status">
          {note ?? waiting}
        </p>
      )}

      <div className="dock-help-row">
        {onHelp && (
          <button className="dock-help" title="How this tool works" onClick={onHelp}>
            How to
          </button>
        )}
        {onShortcuts && (
          <button className="dock-help" title="Keyboard shortcuts (?)" aria-label="Keyboard shortcuts" onClick={onShortcuts}>
            ⌨
          </button>
        )}
      </div>

      {(tool === "brush" || tool === "erase") && (
        <label className="slider dock-slider" title="Brush size ([ and ])">
          <span>Size</span>
          <input
            type="range"
            min={4}
            max={200}
            value={brushSize}
            onChange={(event) => setBrushSize(Number(event.target.value))}
          />
          <span>{brushSize}</span>
        </label>
      )}
    </div>
  );
}

const ZOOM_STEPS = [1, 1.5, 2, 3, 4, 6];

interface ViewportProps {
  zoom: number;
  setZoom: (zoom: number) => void;
  // Night preview: the photo darkened and only the lights, glowing, as the customer will see them.
  night: { on: boolean; set: (on: boolean) => void };
  // Real lengths on the photo's lines; offered once the photo has a measurement.
  lengths?: { on: boolean; set: (on: boolean) => void } | null;
}

// Zoom sits in its own floating strip, bottom-left over the canvas, so the tool dock stays about
// the drawing tools only (the mouse wheel over the photo zooms too). Undo and redo are on the bar.
export function ViewportControls({ zoom, setZoom, night, lengths }: ViewportProps) {
  const zoomIn = ZOOM_STEPS.find((step) => step > zoom + 1e-6);
  const zoomOut = [...ZOOM_STEPS].reverse().find((step) => step < zoom - 1e-6);

  return (
    <div className="viewport-controls">
      <div className="tool-group" aria-label="Zoom">
        <button title="Zoom out" aria-label="Zoom out" disabled={!zoomOut} onClick={() => zoomOut && setZoom(zoomOut)}>
          −
        </button>
        <button title="Fit the whole photo (the mouse wheel over the photo zooms; drag with Space held to move round)" className="zoom-level" onClick={() => setZoom(1)}>
          {zoom === 1 ? "Fit" : `${Math.round(zoom * 100)}%`}
        </button>
        <button title="Zoom in" aria-label="Zoom in" disabled={!zoomIn} onClick={() => zoomIn && setZoom(zoomIn)}>
          +
        </button>
      </div>
      <div className="tool-group">
        <button
          title={night.on ? "Back to editing (N)" : "See the lights at night, as the customer will (N)"}
          aria-pressed={night.on}
          className={night.on ? "active night-toggle" : "night-toggle"}
          onClick={() => night.set(!night.on)}
        >
          {night.on ? "☾ Night" : "☾ Night preview"}
        </button>
      </div>
      {lengths && !night.on && (
        <div className="tool-group">
          <button
            title={lengths.on ? "Hide the real lengths on the lines" : "Show each line's real length on the photo"}
            aria-pressed={lengths.on}
            className={lengths.on ? "active" : ""}
            onClick={() => lengths.set(!lengths.on)}
          >
            Lengths
          </button>
        </div>
      )}
    </div>
  );
}

interface HistoryProps {
  canUndo: boolean;
  canRedo: boolean;
  onUndo: () => void;
  onRedo: () => void;
}

// Undo and redo on the bar: every change to the lines and areas (painting, moving a line, a type
// or side picked in the list, a line or area added or deleted), newest first.
export function HistoryButtons({ canUndo, canRedo, onUndo, onRedo }: HistoryProps) {
  return (
    <div className="history-buttons" role="group" aria-label="Undo and redo">
      <button className="icon-button" title="Undo (Ctrl/Cmd+Z)" aria-label="Undo" disabled={!canUndo} onClick={onUndo}>
        {svg(
          <>
            <path d="M9 14 4 9l5-5" />
            <path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11" />
          </>,
        )}
      </button>
      <button className="icon-button" title="Redo (Ctrl/Cmd+Y or Shift+Ctrl/Cmd+Z)" aria-label="Redo" disabled={!canRedo} onClick={onRedo}>
        {svg(
          <>
            <path d="m15 14 5-5-5-5" />
            <path d="M20 9H9.5a5.5 5.5 0 0 0 0 11H13" />
          </>,
        )}
      </button>
    </div>
  );
}
