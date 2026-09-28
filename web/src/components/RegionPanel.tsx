import { useEffect, useRef, useState, type ReactNode } from "react";

import { polylineLength } from "../canvas/layers";
import { AREA_KINDS, KIND_COLORS, KIND_GROUPS, KIND_LABELS, LINE_KINDS } from "../types";
import type { Edge, Region, RegionKind, Side } from "../types";

interface Props {
  regions: Region[];
  activeId: string | null;
  drawingKind: RegionKind | null;
  showObstacles: boolean;
  onShowObstacles: (show: boolean) => void;
  onSelect: (id: string) => void;
  onPatch: (
    id: string,
    patch: Partial<Pick<Region, "label" | "kind" | "side" | "edge" | "color" | "visible">>,
  ) => void;
  onDelete: (id: string) => void;
  onAdd: (kind: RegionKind) => void;
}

const SIDES: Side[] = ["front", "left", "right"];
const EDGES: Edge[] = ["eave", "rake", "hip", "ridge", "valley", "junction", "unknown"];
// The reference app's palette, one colour per roof edge role (kept in step with the API's
// detect.EDGE_COLORS). Edges without a named role keep the plain roof colour.
export const EDGE_COLORS: Partial<Record<Edge, string>> = {
  ridge: "#ff3b30", hip: "#ff9f0a", valley: "#0a84ff", eave: "#39ff14", rake: "#bf5af2",
};
// The key covers the edges a "roofline" instruction lights (intent.OUTLINE_EDGES); the others
// are drawn for reference and their rows say what they are.
const LIT_EDGES: { edge: Edge; name: string }[] = [
  { edge: "eave", name: "Eave" }, { edge: "rake", name: "Rake" }, { edge: "hip", name: "Hip" },
];

// Share of a line that passes behind objects, as a whole percentage.
function hiddenShare(region: Region): number {
  const total = region.polyline ? polylineLength(region.polyline) : 0;
  if (!total || !region.hidden?.length) return 0;
  const hidden = region.hidden.reduce((sum, [start, end]) => sum + (end - start), 0);
  return Math.round((100 * hidden) / total);
}

// Detected rows carry a trailing " (not lit)" that is already said by the section they sit in.
const shortLabel = (label: string) => label.replace(/\s*\(not lit\)$/, "");

const SOURCE_NAMES: Record<string, string> = { v3: "V3", partition: "partition" };

// Which detectors found a roof line: " · V3 + partition", or nothing for rows without sources.
const sourceNote = (region: Region) =>
  region.sources?.length ? ` · ${region.sources.map((s) => SOURCE_NAMES[s] ?? s).join(" + ")}` : "";

function Section({
  title,
  count,
  open,
  onToggle,
  note,
  actions,
  children,
}: {
  title: string;
  count?: number;
  open: boolean;
  onToggle: (open: boolean) => void;
  note?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <details className="panel-section" open={open} onToggle={(event) => onToggle(event.currentTarget.open)}>
      <summary>
        <span>{title}</span>
        {count !== undefined && <span className="count">{count}</span>}
        {actions && (
          // A click here must not also toggle the <details> it sits inside.
          <span className="summary-actions" onClick={(event) => event.preventDefault()}>
            {actions}
          </span>
        )}
      </summary>
      {note}
      {children}
    </details>
  );
}

// A named roof edge groups under its role; every other line kind (driveway edge, fence, ...)
// groups under itself. Matches how the reference tool talks about a house: "5 eaves, 3 rakes".
function groupLines(lines: Region[]): { key: string; label: string; regions: Region[] }[] {
  const order = ["eave", "rake", "hip"];
  const groups = new Map<string, { key: string; label: string; regions: Region[] }>();
  for (const region of lines) {
    const key = region.kind === "roof" ? region.edge ?? "unknown" : region.kind;
    const label =
      region.kind === "roof"
        ? (LIT_EDGES.find((item) => item.edge === key)?.name ?? key[0].toUpperCase() + key.slice(1))
        : KIND_LABELS[region.kind];
    if (!groups.has(key)) groups.set(key, { key, label, regions: [] });
    groups.get(key)!.regions.push(region);
  }
  return [...groups.values()].sort((a, b) => {
    const pa = order.indexOf(a.key);
    const pb = order.indexOf(b.key);
    if (pa === -1 && pb === -1) return 0;
    return (pa === -1 ? order.length : pa) - (pb === -1 ? order.length : pb);
  });
}

export function RegionPanel({
  regions,
  activeId,
  drawingKind,
  showObstacles,
  onShowObstacles,
  onSelect,
  onPatch,
  onDelete,
  onAdd,
}: Props) {
  const targets = regions.filter((region) => region.role !== "blocker");
  // A line is lit when it is visible: detected-but-unrequested edges (ridges, roof meeting wall)
  // arrive hidden, and so does any line the user switches off.
  const lit = targets.filter((region) => region.geometry === "line" && region.visible);
  const unlit = targets.filter((region) => region.geometry === "line" && !region.visible);
  const areas = targets.filter((region) => region.geometry === "area");
  const blockers = regions.filter((region) => region.role === "blocker");

  const holds = (group: Region[]) => group.some((region) => region.id === activeId);
  const [open, setOpen] = useState({ unlit: false, blockers: false, add: false });
  // Lit-line groups (Eave, Rake, ...) start open; collapsed ones are remembered by key.
  const [litGroupsOpen, setLitGroupsOpen] = useState<Record<string, boolean>>({});
  // A selection made on the photo opens the section it sits in, so it is never selected unseen.
  useEffect(() => {
    setOpen((current) => ({
      unlit: current.unlit || holds(unlit),
      blockers: current.blockers || holds(blockers),
      add: current.add || Boolean(drawingKind),
    }));
    if (activeId) {
      const activeGroupKey = lit.find((region) => region.id === activeId);
      if (activeGroupKey) {
        const key = activeGroupKey.kind === "roof" ? activeGroupKey.edge ?? "unknown" : activeGroupKey.kind;
        setLitGroupsOpen((current) => (current[key] === false ? { ...current, [key]: true } : current));
      }
    }
  }, [activeId, drawingKind]);

  // Selecting on the photo has to show up here, even when the row is scrolled out of sight.
  const activeRow = useRef<HTMLLIElement | null>(null);
  useEffect(() => {
    activeRow.current?.scrollIntoView({ block: "nearest" });
  }, [activeId]);

  const row = (region: Region) => {
    const isActive = region.id === activeId;
    const hidden = region.geometry === "line" ? hiddenShare(region) : 0;
    return (
      <li
        key={region.id}
        ref={isActive ? activeRow : undefined}
        className={isActive ? "region active" : "region"}
        data-geometry={region.geometry}
        data-role={region.role}
        data-kind={region.kind}
        data-edge={region.edge ?? ""}
        onClick={() => onSelect(region.id)}
      >
        <div className="region-head">
          <span className="swatch" style={{ background: region.color }} />
          {isActive ? (
            <input
              aria-label="Name"
              value={region.label}
              onClick={(event) => event.stopPropagation()}
              onChange={(event) => onPatch(region.id, { label: event.target.value })}
            />
          ) : (
            <span className="region-name">{shortLabel(region.label)}</span>
          )}
          {hidden > 0 && !isActive && (
            <span className="region-flag" title={`${hidden}% passes behind objects — no lights there`}>
              {hidden}% hidden
            </span>
          )}
          {region.kind === "roof" && region.edge === "unknown" && !isActive && (
            <span className="region-flag warn" title="Edge type not identified">
              check
            </span>
          )}
          <button
            className="icon"
            title={region.visible ? (region.geometry === "line" ? "Don't light this line" : "Hide") : region.geometry === "line" ? "Light this line" : "Show"}
            aria-label={region.visible ? "Hide" : "Show"}
            onClick={(event) => {
              event.stopPropagation();
              onPatch(region.id, { visible: !region.visible });
            }}
          >
            {region.visible ? "◉" : "○"}
          </button>
          <button
            className="icon"
            title="Delete"
            aria-label="Delete"
            onClick={(event) => {
              event.stopPropagation();
              onDelete(region.id);
            }}
          >
            ✕
          </button>
        </div>

        {isActive && (
          <>
            <div className="region-meta">
              <select
                aria-label="Kind"
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
                    aria-label="Side"
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
                      aria-label="Edge"
                      value={region.edge ?? "eave"}
                      onClick={(event) => event.stopPropagation()}
                      onChange={(event) => {
                        const edge = event.target.value as Edge;
                        // a new role brings its colour with it, so the drawing stays readable
                        onPatch(region.id, { edge, ...(EDGE_COLORS[edge] ? { color: EDGE_COLORS[edge] } : {}) });
                      }}
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

              <span className="source">
                {region.source === "auto"
                  ? `found · ${Math.round((region.confidence ?? region.score) * 100)}%${sourceNote(region)}`
                  : region.source === "inferred"
                    ? "continued"
                    : "drawn by you"}
              </span>
            </div>

            {hidden > 0 && (
              <p className="muted small hidden-note">{hidden}% passes behind objects — no lights there</p>
            )}
            {region.kind === "roof" && region.edge === "unknown" && (
              <p className="small unknown-note">Edge type not identified. Choose eave, rake, hip or ridge.</p>
            )}
          </>
        )}
      </li>
    );
  };

  const shownEdges = LIT_EDGES.filter(({ edge }) => lit.some((line) => line.edge === edge));

  return (
    <div className="region-panel">
      <h2>
        Lines to light <span className="count">{lit.length}</span>
      </h2>
      {shownEdges.length > 0 && (
        <ul className="legend" aria-label="What the line colours mean">
          {shownEdges.map(({ edge, name }) => (
            <li key={edge}>
              <span className="swatch" style={{ background: EDGE_COLORS[edge] }} />
              {name}
            </li>
          ))}
        </ul>
      )}
      {lit.length ? (
        groupLines(lit).map((group) => {
          const allVisible = group.regions.every((region) => region.visible);
          return (
            <Section
              key={group.key}
              title={group.label}
              count={group.regions.length}
              open={litGroupsOpen[group.key] ?? true}
              onToggle={(value) => setLitGroupsOpen((current) => ({ ...current, [group.key]: value }))}
              actions={
                group.regions.length > 1 && (
                  <button
                    className="icon"
                    title={allVisible ? "Don't light any of these" : "Light all of these"}
                    aria-label={allVisible ? "Hide group" : "Light group"}
                    onClick={() => group.regions.forEach((region) => onPatch(region.id, { visible: !allVisible }))}
                  >
                    {allVisible ? "◉" : "○"}
                  </button>
                )
              }
            >
              <ul>{group.regions.map(row)}</ul>
            </Section>
          );
        })
      ) : (
        <p className="muted small">None yet. Add a line below and click along the roofline to draw it.</p>
      )}

      {unlit.length > 0 && (
        <Section
          title="Other roof edges, not lit"
          count={unlit.length}
          open={open.unlit}
          onToggle={(value) => setOpen((current) => ({ ...current, unlit: value }))}
          note={<p className="muted small">Found on the roof but not part of the instruction. Use ○ to light one.</p>}
        >
          <ul>{unlit.map(row)}</ul>
        </Section>
      )}

      <h2>
        Areas <span className="count">{areas.length}</span>
      </h2>
      {areas.length ? (
        <ul>{areas.map(row)}</ul>
      ) : (
        <p className="muted small">None yet. Add one below, then paint it with the brush.</p>
      )}

      {blockers.length > 0 && (
        <Section
          title="Obstacles in front of the house"
          count={blockers.length}
          open={open.blockers}
          onToggle={(value) => setOpen((current) => ({ ...current, blockers: value }))}
          note={
            <>
              <p className="muted small">
                Found so the roofline can continue behind trees, palms and cars. Never lit. If a
                line joins wrongly, correct the obstacle here.
              </p>
              <label className="check">
                <input
                  type="checkbox"
                  checked={showObstacles}
                  onChange={(event) => onShowObstacles(event.target.checked)}
                />
                Show on the photo
              </label>
            </>
          }
        >
          <ul>{blockers.map(row)}</ul>
        </Section>
      )}

      <Section
        title="Add a line or area"
        open={open.add}
        onToggle={(value) => setOpen((current) => ({ ...current, add: value }))}
      >
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
      </Section>
    </div>
  );
}
