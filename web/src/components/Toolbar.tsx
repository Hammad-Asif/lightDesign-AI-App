import type { ReactNode } from "react";

import type { Region, Tool } from "../types";

type ToolSpec = { id: Tool; label: string; key?: string; hint: string; needs?: "line" | "area"; icon: ReactNode };

function svg(children: ReactNode) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {children}
    </svg>
  );
}

// Grouped by what they act on: picking, painting an area, shaping a line, asking the model again.
const GROUPS: ToolSpec[][] = [
  [
    {
      id: "select",
      label: "Select",
      key: "S",
      hint: "Click a line or area on the photo to select it",
      icon: svg(<path d="M5 3l15 8.5-6.4 1.4L11 20z" />),
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
      id: "box",
      label: "Box",
      hint: "Drag a box around something to mask it again",
      icon: svg(<rect x="4.5" y="4.5" width="15" height="15" rx="1.5" strokeDasharray="4 3" />),
    },
    {
      id: "click",
      label: "Click object",
      hint: "Click an object to mask it; Alt-click a mask to remove it",
      icon: svg(
        <>
          <circle cx="12" cy="12" r="3" />
          <path d="M12 2v4M12 18v4M2 12h4M18 12h4" />
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
}

// A floating vertical dock docked over the canvas, so the tools that act on it read as part of
// it rather than a disconnected bar of text buttons above.
export function ToolDock({ tool, setTool, brushSize, setBrushSize, active }: DockProps) {
  return (
    <div className="tool-dock" role="toolbar" aria-label="Editing tools" aria-orientation="vertical">
      {GROUPS.map((group, index) => (
        <div key={index} className="dock-group">
          {group.map((item) => {
            const missing = item.needs && active?.geometry !== item.needs;
            const hint = missing
              ? `Select ${item.needs === "line" ? "a line" : "an area"} first`
              : item.key
                ? `${item.hint} (${item.key})`
                : item.hint;
            return (
              <button
                key={item.id}
                title={hint}
                aria-pressed={tool === item.id}
                className={tool === item.id ? "dock-tool active" : "dock-tool"}
                disabled={missing}
                onClick={() => setTool(item.id)}
              >
                {item.icon}
                <span>{item.label}</span>
              </button>
            );
          })}
        </div>
      ))}

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

const ZOOM_STEPS = [1, 1.5, 2, 3, 4];

interface ViewportProps {
  zoom: number;
  setZoom: (zoom: number) => void;
  history: { canUndo: boolean; canRedo: boolean };
  onUndo: () => void;
  onRedo: () => void;
}

// Zoom and history sit in their own floating strip, bottom-left over the canvas, so the tool
// dock above stays about the drawing tools only.
export function ViewportControls({ zoom, setZoom, history, onUndo, onRedo }: ViewportProps) {
  const zoomIn = ZOOM_STEPS.find((step) => step > zoom + 1e-6);
  const zoomOut = [...ZOOM_STEPS].reverse().find((step) => step < zoom - 1e-6);

  return (
    <div className="viewport-controls">
      <div className="tool-group" aria-label="Zoom">
        <button title="Zoom out" aria-label="Zoom out" disabled={!zoomOut} onClick={() => zoomOut && setZoom(zoomOut)}>
          −
        </button>
        <button title="Fit the whole photo" className="zoom-level" onClick={() => setZoom(1)}>
          {zoom === 1 ? "Fit" : `${Math.round(zoom * 100)}%`}
        </button>
        <button title="Zoom in" aria-label="Zoom in" disabled={!zoomIn} onClick={() => zoomIn && setZoom(zoomIn)}>
          +
        </button>
      </div>
      <div className="tool-group">
        <button title="Undo (Ctrl/Cmd+Z)" disabled={!history.canUndo} onClick={onUndo}>
          Undo
        </button>
        <button title="Redo (Shift+Ctrl/Cmd+Z)" disabled={!history.canRedo} onClick={onRedo}>
          Redo
        </button>
      </div>
    </div>
  );
}
