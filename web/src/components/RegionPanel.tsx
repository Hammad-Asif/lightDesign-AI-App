import { polylineLength } from "../canvas/layers";
import { AREA_KINDS, KIND_COLORS, KIND_GROUPS, KIND_LABELS, LINE_KINDS } from "../types";
import type { Edge, Region, RegionKind, Side } from "../types";

interface Props {
  regions: Region[];
  activeId: string | null;
  drawingKind: RegionKind | null;
  onSelect: (id: string) => void;
  onPatch: (
    id: string,
    patch: Partial<Pick<Region, "label" | "kind" | "side" | "edge" | "color" | "visible">>,
  ) => void;
  onDelete: (id: string) => void;
  onAdd: (kind: RegionKind) => void;
}

const SIDES: Side[] = ["front", "left", "right"];
const EDGES: Edge[] = ["eave", "ridge"];

// Share of a line that passes behind objects, as a whole percentage.
function hiddenShare(region: Region): number {
  const total = region.polyline ? polylineLength(region.polyline) : 0;
  if (!total || !region.hidden?.length) return 0;
  const hidden = region.hidden.reduce((sum, [start, end]) => sum + (end - start), 0);
  return Math.round((100 * hidden) / total);
}

export function RegionPanel({
  regions,
  activeId,
  drawingKind,
  onSelect,
  onPatch,
  onDelete,
  onAdd,
}: Props) {
  const lines = regions.filter((region) => region.geometry === "line");
  const areas = regions.filter((region) => region.geometry === "area");

  const row = (region: Region) => (
    <li
      key={region.id}
      className={region.id === activeId ? "region active" : "region"}
      data-geometry={region.geometry}
      data-edge={region.edge ?? ""}
      onClick={() => onSelect(region.id)}
    >
      <div className="region-head">
        <span className="swatch" style={{ background: region.color }} />
        <input
          value={region.label}
          onClick={(event) => event.stopPropagation()}
          onChange={(event) => onPatch(region.id, { label: event.target.value })}
        />
        <button
          className="icon"
          title={region.visible ? "Hide" : "Show"}
          onClick={(event) => {
            event.stopPropagation();
            onPatch(region.id, { visible: !region.visible });
          }}
        >
          {region.visible ? "◉" : "○"}
        </button>
        <button
          className="icon"
          title="Delete region"
          onClick={(event) => {
            event.stopPropagation();
            onDelete(region.id);
          }}
        >
          ✕
        </button>
      </div>

      <div className="region-meta">
        <select
          value={region.kind}
          onClick={(event) => event.stopPropagation()}
          onChange={(event) => onPatch(region.id, { kind: event.target.value as RegionKind })}
        >
          {(region.geometry === "line" ? LINE_KINDS : AREA_KINDS).map((kind) => (
            <option key={kind} value={kind}>
              {KIND_LABELS[kind]}
            </option>
          ))}
        </select>

        {region.geometry === "line" && (
          <>
            <select
              value={region.side ?? "front"}
              onClick={(event) => event.stopPropagation()}
              onChange={(event) => onPatch(region.id, { side: event.target.value as Side })}
            >
              {SIDES.map((side) => (
                <option key={side} value={side}>
                  {side}
                </option>
              ))}
            </select>
            {region.kind === "roof" && (
              <select
                value={region.edge ?? "eave"}
                onClick={(event) => event.stopPropagation()}
                onChange={(event) => onPatch(region.id, { edge: event.target.value as Edge })}
              >
                {EDGES.map((edge) => (
                  <option key={edge} value={edge}>
                    {edge}
                  </option>
                ))}
              </select>
            )}
          </>
        )}

        {region.kind === "custom" && (
          <input
            type="color"
            className="colour"
            title="Pick a colour for this region"
            value={region.color}
            onClick={(event) => event.stopPropagation()}
            onChange={(event) => onPatch(region.id, { color: event.target.value })}
          />
        )}

        <span className="muted small">
          {region.source === "auto" ? `auto · ${region.score.toFixed(2)}` : "manual"}
        </span>
      </div>

      {region.geometry === "line" && hiddenShare(region) > 0 && (
        <p className="muted small hidden-note">
          {hiddenShare(region)}% passes behind objects — no lights there
        </p>
      )}
    </li>
  );

  return (
    <aside className="sidebar">
      <h2>Lines · {lines.length}</h2>
      {lines.length ? (
        <ul>{lines.map(row)}</ul>
      ) : (
        <p className="muted small">None yet. Add one below and click along the path.</p>
      )}

      <h2>Areas · {areas.length}</h2>
      {areas.length ? (
        <ul>{areas.map(row)}</ul>
      ) : (
        <p className="muted small">None yet. Add one below and paint with the brush.</p>
      )}

      <h2>Add region</h2>
      {drawingKind && (
        <p className="drawing-hint">
          Click along the {KIND_LABELS[drawingKind].toLowerCase()} on the photo. Enter to finish,
          Esc to cancel.
        </p>
      )}
      {KIND_GROUPS.map((group) => (
        <div key={group.title} className="add-group">
          <h3>{group.title}</h3>
          <div className="add-grid">
            {group.kinds.map((kind) => (
              <button
                key={kind}
                className={drawingKind === kind ? "add-kind drawing" : "add-kind"}
                onClick={() => onAdd(kind)}
              >
                <span className="swatch" style={{ background: KIND_COLORS[kind] }} />
                {KIND_LABELS[kind]}
              </button>
            ))}
          </div>
        </div>
      ))}
    </aside>
  );
}
