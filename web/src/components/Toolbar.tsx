import type { Region, Tool } from "../types";

interface Props {
  tool: Tool;
  setTool: (tool: Tool) => void;
  brushSize: number;
  setBrushSize: (size: number) => void;
  zoom: number;
  setZoom: (zoom: number) => void;
  active: Region | undefined;
  history: { canUndo: boolean; canRedo: boolean };
  busy: boolean;
  onUndo: () => void;
  onRedo: () => void;
  onRedetect: () => void;
  onDetectAll: () => void;
  onApprove: () => void;
}

type ToolSpec = { id: Tool; label: string; key?: string; hint: string; needs?: "line" | "area" };

// Grouped by what they act on: picking, painting an area, shaping a line, asking the model again.
const GROUPS: ToolSpec[][] = [
  [{ id: "select", label: "Select", key: "S", hint: "Click a line or area on the photo to select it" }],
  [
    { id: "brush", label: "Brush", key: "B", hint: "Paint onto the selected area", needs: "area" },
    { id: "erase", label: "Erase", key: "E", hint: "Remove from the selected area", needs: "area" },
  ],
  [
    { id: "vertex", label: "Points", key: "V", hint: "Drag a point; click the line to add one; Alt-click a point to delete it", needs: "line" },
    { id: "move", label: "Move / extend", key: "M", hint: "Drag the line to move it, or drag an end to lengthen or shorten it", needs: "line" },
  ],
  [
    { id: "box", label: "Box", hint: "Drag a box around something to mask it again" },
    { id: "click", label: "Click object", hint: "Click an object to mask it; Alt-click a mask to remove it" },
  ],
];

const ZOOM_STEPS = [1, 1.5, 2, 3, 4];

export function Toolbar({
  tool,
  setTool,
  brushSize,
  setBrushSize,
  zoom,
  setZoom,
  active,
  history,
  busy,
  onUndo,
  onRedo,
  onRedetect,
  onDetectAll,
  onApprove,
}: Props) {
  const zoomIn = ZOOM_STEPS.find((step) => step > zoom + 1e-6);
  const zoomOut = [...ZOOM_STEPS].reverse().find((step) => step < zoom - 1e-6);

  return (
    <div className="toolbar" role="toolbar" aria-label="Editing tools">
      {GROUPS.map((group, index) => (
        <div key={index} className="tool-group">
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
                className={tool === item.id ? "tool active" : "tool"}
                disabled={missing}
                onClick={() => setTool(item.id)}
              >
                {item.label}
              </button>
            );
          })}
        </div>
      ))}

      {(tool === "brush" || tool === "erase") && (
        <label className="slider" title="Brush size ([ and ])">
          Size
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

      <div className="spacer" />

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

      <div className="tool-group">
        <button
          title="Detect again from the instruction. Replaces every detected line and area; ones you drew stay."
          disabled={busy}
          onClick={onRedetect}
        >
          Re-detect
        </button>
        <button
          title="Detect every kind of area, not only the ones the instruction names"
          disabled={busy}
          onClick={onDetectAll}
        >
          Detect everything
        </button>
      </div>

      <button className="primary" title="Lock these lines and areas" disabled={busy} onClick={onApprove}>
        Approve
      </button>
    </div>
  );
}
