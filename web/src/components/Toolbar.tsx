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
  onApprove: () => void;
}

const TOOLS: { id: Tool; label: string; hint: string }[] = [
  { id: "brush", label: "Brush", hint: "Add to the selected area mask" },
  { id: "erase", label: "Erase", hint: "Remove from the selected area mask" },
  { id: "vertex", label: "Vertices", hint: "Drag roofline points, click a segment to add, alt-click to delete" },
  { id: "box", label: "Re-segment", hint: "Drag a box to re-run segmentation on that area" },
  { id: "click", label: "Click object", hint: "Click an object to mask it, alt-click a mask to remove it" },
];

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
  onApprove,
}: Props) {
  const isLine = active?.geometry === "line";

  return (
    <div className="toolbar">
      <div className="tools">
        {TOOLS.map((item) => {
          const disabled =
            (item.id === "vertex" && !isLine) ||
            ((item.id === "brush" || item.id === "erase") && active?.geometry !== "area");
          return (
            <button
              key={item.id}
              title={item.hint}
              className={tool === item.id ? "tool active" : "tool"}
              disabled={disabled}
              onClick={() => setTool(item.id)}
            >
              {item.label}
            </button>
          );
        })}
      </div>

      <label className="slider">
        Brush
        <input
          type="range"
          min={4}
          max={200}
          value={brushSize}
          onChange={(event) => setBrushSize(Number(event.target.value))}
        />
        <span>{brushSize}</span>
      </label>

      <label className="slider">
        Zoom
        <input
          type="range"
          min={1}
          max={4}
          step={0.25}
          value={zoom}
          onChange={(event) => setZoom(Number(event.target.value))}
        />
        <span>{zoom.toFixed(2)}×</span>
      </label>

      <div className="spacer" />

      <button disabled={!history.canUndo} onClick={onUndo}>Undo</button>
      <button disabled={!history.canRedo} onClick={onRedo}>Redo</button>
      <button disabled={busy} onClick={onRedetect}>Re-detect</button>
      <button className="primary" disabled={busy} onClick={onApprove}>
        Approve mask set
      </button>
    </div>
  );
}
