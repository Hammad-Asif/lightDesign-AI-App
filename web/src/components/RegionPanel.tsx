import { useEffect, useRef, useState, type ReactNode } from "react";

import { polylineLength } from "../canvas/layers";
import { AREA_KINDS, KIND_COLORS, KIND_GROUPS, KIND_LABELS, LINE_KINDS } from "../types";
import type { Edge, LineGeometry, Measurement, Region, RegionKind, RegionPlacement, ScaleCheck, ScaleOverride, Side } from "../types";
import { formatLength, spreadText } from "./MeasurePanel";

interface Props {
  regions: Region[];
  activeId: string | null;
  drawingKind: RegionKind | null;
  showObstacles: boolean;
  onShowObstacles: (show: boolean) => void;
  // A row or a photo pick selects; null goes back from one item to the whole list.
  onSelect: (id: string | null) => void;
  onPatch: (
    id: string,
    patch: Partial<Pick<Region, "label" | "kind" | "side" | "edge" | "color" | "visible">>,
  ) => void;
  onDelete: (id: string) => void;
  onAdd: (kind: RegionKind) => void;
  // Each roof line's real length and how it was placed (GET /photos/{id}/geometry), by region id.
  geometry?: Record<string, LineGeometry>;
  // A line the geometry cannot place: its hand-typed length (its middle is dragged on the photo).
  onOverride?: (id: string, override: ScaleOverride | null) => void;
  // The confirmed lighting's bulbs and counts, by region id (GET /photos/{id}/placement).
  placement?: Record<string, RegionPlacement>;
  // The photo's scale: the saved measurement, where the lens came from, and opening Measure.
  measurement?: Measurement | null;
  referenceEdited?: boolean;
  // The measurement makes the house implausibly wide (GET /geometry's warning).
  scaleWarning?: string | null;
  // Every independent check that failed (GET /geometry's checks), and the side the measurement went on.
  scaleChecks?: ScaleCheck[];
  referenceSide?: Side | null;
  lens?: "vp" | "exif" | "assumed" | null;
  onMeasure?: () => void;
  // Measure again with the last saved values.
  onRemeasure?: () => void;
  // The whole house, picked by tapping its roof with Select: how many roofs it has, and changing
  // which roofs belong to it (the House tool).
  house?: { roofs: number; onChangeRoofs?: () => void; onClear: () => void } | null;
  // Scroll the selected row into view when the selection changes (off while Measure is open, so
  // the panel stays on the measurement).
  follow?: boolean;
}

const LENS: Record<string, string> = {
  vp: "lens read from the perspective",
  exif: "lens from the photo's data",
  assumed: "lens assumed, so other walls are estimates",
};

// What the photo's scale is, and whether every roof line has a real length yet.
function ScaleSummary({ roofLines, geometry, measurement, lens, onMeasure, onRemeasure, onSelect, activeId, edited,
                        warning, checks, side }: {
  roofLines: Region[];
  activeId: string | null;
  edited?: boolean;
  warning?: string | null;
  checks?: ScaleCheck[];
  side?: Side | null;
  geometry?: Record<string, LineGeometry>;
  measurement?: Measurement | null;
  lens?: string | null;
  onMeasure?: () => void;
  onRemeasure?: () => void;
  onSelect: (id: string) => void;
}) {
  if (!roofLines.length) return null;
  if (!measurement) {
    return (
      <div className="scale-summary prompt">
        <p>Measure one known length to see every line's real length and place the lights.</p>
        {onMeasure && <button onClick={onMeasure}>Measure</button>}
      </div>
    );
  }
  const waiting = roofLines.filter((line) => geometry?.[line.id] && (geometry[line.id].length_in == null || geometry[line.id].warning));
  const measured = roofLines.filter((line) => geometry?.[line.id]?.length_in != null).length;
  const otherChecks = (checks ?? []).filter((check) => check.message !== warning);
  return (
    <div className="scale-summary">
      <p>
        Scale: {formatLength(measurement.length_in)}, {side ?? measurement.facade ?? "front"} side
        {lens ? ` · ${LENS[lens]}` : ""}
        {onMeasure && (
          <>
            {" "}
            <button className="link" onClick={onMeasure}>Change</button>
          </>
        )}
        {onRemeasure && (
          <>
            {" "}
            <button className="link" title="Measure again with the last values" onClick={onRemeasure}>
              Re-measure
            </button>
          </>
        )}
      </p>
      {edited && (
        <p className="warn-text">The line you measured has been edited since, and the scale moved with it. Check its length.</p>
      )}
      {warning && <p className="warn-text">{warning}</p>}
      {otherChecks.length > 0 && (
        // The checks are details: one line says how many, opening it lists them.
        <details className="scale-checks">
          <summary className="warn-text small">
            {otherChecks.length} {otherChecks.length === 1 ? "thing" : "things"} to check about the scale
          </summary>
          {otherChecks.map((check, index) => (
            <p key={index} className="warn-text small">
              {check.message}{" "}
              {check.lines.length > 0 && (
                <button className="link" title="Select the line this is about" onClick={() => onSelect(check.lines[0])}>
                  {check.labels?.[0] ?? "Show"}
                </button>
              )}
            </p>
          ))}
        </details>
      )}
      <p className="muted">
        {measured} {measured === 1 ? "line" : "lines"} measured
        {waiting.length > 0 && (
          <>
            {" · "}
            <span className="warn-text">{waiting.length} {waiting.length === 1 ? "needs" : "need"} attention</span>{" "}
            <button
              className="link"
              title="Select the next line that has no length yet"
              onClick={() => {
                const current = waiting.findIndex((line) => line.id === activeId);
                onSelect(waiting[(current + 1) % waiting.length].id);
              }}
            >
              Show
            </button>
          </>
        )}
      </p>
    </div>
  );
}

// Why a line has no real length yet, and what the salesperson can do about it.
const NEEDS: Record<string, string> = {
  measurement: "Needs a measurement on this side of the house. Use the Measure tool.",
  measurement_wall: "The measurement's wall is not in this photo's perspective. Measure again.",
  side: "No side of the house could be read for this line. Choose one, or set its middle and length by hand.",
  override: "Its depth cannot be read from the photo. Drag its middle on the photo and type its length.",
  length: "Type its real length.",
  anchor: "Needs an eave near it to be measured from.",
  edge_on: "Seen edge-on, so it cannot be measured. Type its real length.",
  unreliable:
    "Its length moves too much with the photo's own small errors to be trusted (a wall or roof seen nearly along the line of sight). Type its real length.",
};

// A line whose middle and length the salesperson sets by hand: one set already, one the geometry
// could not place, and one only estimated from the lines around it.
export function needsOverride(line: LineGeometry | undefined): boolean {
  return Boolean(line && (line.method === "override" || line.method === "estimate"
    || (line.needs && !line.needs.startsWith("measurement"))));
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

const DENSITIES = ["light", "medium", "heavy"] as const;

// What the lighting puts on a region: counts, and the estimates the salesperson can correct.
function Lights({ region, lights, onOverride }: {
  region: Region;
  lights: RegionPlacement;
  onOverride?: (override: ScaleOverride | null) => void;
}) {
  const [count, setCount] = useState(lights.estimated ? "" : String(lights.count ?? ""));
  const [strand, setStrand] = useState(lights.strand_set ? String(lights.strand_ft ?? "") : "");
  if (lights.needs) {
    return <p className="small unknown-note">{lights.needs === "measurement" ? "Needs a measurement before its lights can be placed." : `No lights yet: ${lights.needs}.`}</p>;
  }
  const stop = (event: { stopPropagation: () => void }) => event.stopPropagation();
  const hidden = lights.hidden ? ` · ${lights.hidden} behind ${region.kind === "palm_trunk" || region.kind === "palm_collar" ? "the trunk" : "objects"}, counted` : "";

  if (region.kind === "wreath") {
    return (
      <div className="lights" onClick={stop}>
        <p className="muted small">1 wreath · {lights.count} bulbs</p>
        {onOverride && (
          <label className="small">
            Size{" "}
            <select value={lights.size_in ?? 36} onChange={(event) => onOverride({ size_in: Number(event.target.value) as 24 | 36 | 48 })}>
              {[24, 36, 48].map((size) => (
                <option key={size} value={size}>
                  {size} in
                </option>
              ))}
            </select>
          </label>
        )}
      </div>
    );
  }
  if (lights.strand_ft != null) {
    return (
      <div className="lights" onClick={stop}>
        <p className="muted small">
          {lights.strand_set ? "" : "≈ "}
          {lights.strand_ft} ft of strand, {lights.count} bulbs{hidden}
          {lights.trunk ? ` · trunk about ${lights.trunk.diameter_in} in across, ${lights.trunk.height_ft} ft tall` : ""}
          {lights.fronds ? ` · along ${lights.fronds.count} ${lights.fronds.count === 1 ? "frond" : "fronds"}, crown to tip` : ""}
          {lights.strand_set ? " · length set by you" : ` · estimated from ${lights.scale_from}`}
        </p>
        {onOverride && (
          <div className="override-length">
            <label>
              Strand <input type="number" min="0" step="0.5" value={strand} placeholder={String(lights.strand_estimate_ft ?? "")}
                            onChange={(event) => setStrand(event.target.value)} /> ft
            </label>
            <button disabled={!(Number(strand) > 0)} onClick={() => onOverride({ strand_ft: Number(strand) })}>
              Set
            </button>
          </div>
        )}
      </div>
    );
  }
  if (lights.area_sqft != null) {
    return (
      <div className="lights" onClick={stop}>
        <p className="muted small">
          {lights.estimated ? `≈ ${lights.count} bulbs, estimated` : `${lights.count} bulbs, set by you (estimate ${lights.estimate})`} ·
          about {lights.area_sqft} sq ft as seen, {lights.density} density, scaled from {lights.scale_from}
        </p>
        {onOverride && (
          <div className="override-length">
            <select aria-label="Density" value={lights.density} onChange={(event) => onOverride({ density: event.target.value as (typeof DENSITIES)[number] })}>
              {DENSITIES.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
            <label>
              Bulbs <input type="number" min="0" value={count} placeholder={String(lights.estimate ?? "")}
                           onChange={(event) => setCount(event.target.value)} />
            </label>
            <button disabled={count === "" || Number(count) < 0} onClick={() => onOverride({ count: Math.round(Number(count)) })}>
              Set
            </button>
          </div>
        )}
      </div>
    );
  }
  return <p className="muted small">{lights.count} bulbs{hidden}</p>;
}

function LengthNote({ line }: { line: LineGeometry }) {
  if (line.length_in == null) return <p className="small unknown-note">{NEEDS[line.needs ?? ""] ?? "Not measured yet."}</p>;
  return (
    <>
      <p className="muted small length-note">
        {formatLength(line.length_in)}
        {spreadText(line.spread)}
        {line.plane ? ` · ${line.plane}` : ""}
        {line.assumption ? ` · ${line.assumption}` : ""}
      </p>
      {line.warning && <p className="small unknown-note">Check this: {line.warning}.</p>}
    </>
  );
}

function OverrideLength({ current, onSave, onClear }: { current: number | null; onSave: (inches: number) => void; onClear?: () => void }) {
  const [feet, setFeet] = useState(current ? String(Math.round((current / 12) * 10) / 10) : "");
  return (
    <div className="override-length" onClick={(event) => event.stopPropagation()}>
      <label>
        Real length <input type="number" min="0" step="0.5" value={feet} onChange={(event) => setFeet(event.target.value)} /> ft
      </label>
      <button disabled={!(Number(feet) > 0)} onClick={() => onSave(Number(feet) * 12)}>
        Set
      </button>
      {onClear && (
        <button className="link" onClick={onClear}>
          Clear
        </button>
      )}
    </div>
  );
}

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
  geometry,
  onOverride,
  placement,
  measurement,
  lens,
  onMeasure,
  referenceEdited,
  scaleWarning,
  scaleChecks,
  referenceSide,
  onRemeasure,
  house = null,
  follow = true,
}: Props) {
  const targets = regions.filter((region) => region.role !== "blocker");
  // A line is lit when it is visible: detected-but-unrequested edges (ridges, roof meeting wall)
  // arrive hidden, and so does any line the user switches off.
  const lit = targets.filter((region) => region.geometry === "line" && region.visible);
  const unlit = targets.filter((region) => region.geometry === "line" && !region.visible);
  const areas = targets.filter((region) => region.geometry === "area");
  const blockers = regions.filter((region) => region.role === "blocker");
  const roofLines = targets.filter((region) => region.geometry === "line" && region.kind === "roof");

  const holds = (group: Region[]) => group.some((region) => region.id === activeId);
  // "More" holds the roof edges that are not lit and the obstacles: found, rarely worked on.
  const [open, setOpen] = useState({ more: false, add: false });
  // Lit-line groups (Eave, Rake, ...) start open; collapsed ones are remembered by key.
  const [litGroupsOpen, setLitGroupsOpen] = useState<Record<string, boolean>>({});
  // A selection made on the photo opens the section it sits in, so it is never selected unseen.
  useEffect(() => {
    setOpen((current) => ({
      more: current.more || holds(unlit) || holds(blockers),
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

  // Selecting on the photo has to show up here, even when the row is scrolled out of sight. The
  // scroll waits for the render that opened the row's section (a row in a closed section cannot
  // be scrolled to), then one frame for its layout, and centres the row.
  const activeRow = useRef<HTMLLIElement | null>(null);
  const pendingScroll = useRef(false);
  useEffect(() => {
    if (activeId && follow) pendingScroll.current = true;
  }, [activeId, follow]);
  useEffect(() => {
    if (!pendingScroll.current || !activeRow.current) return;
    const row = activeRow.current;
    const frame = requestAnimationFrame(() => {
      if (!row.offsetParent) return; // still inside a closed section: the next render tries again
      row.scrollIntoView({ block: "center", behavior: "smooth" });
      pendingScroll.current = false;
    });
    return () => cancelAnimationFrame(frame);
  });

  const row = (region: Region) => {
    const isActive = region.id === activeId;
    const hidden = region.geometry === "line" ? hiddenShare(region) : 0;
    const measured = region.geometry === "line" ? geometry?.[region.id] : undefined;
    const lights = placement?.[region.id];
    // One flag on a closed row, the one that matters most: a type to check, else the real length,
    // else the bulbs, else how much is hidden. The open row says all of it.
    const flag = isActive
      ? null
      : region.kind === "roof" && region.edge === "unknown"
        ? <span className="region-flag warn" title="Edge type not identified">check</span>
        : measured?.length_in != null
          ? <span className="region-flag" title={measured.assumption ?? "Measured"}>{formatLength(measured.length_in)}</span>
          : lights?.count != null
            ? (
              <span className="region-flag" title={lights.estimated ? "Estimated" : "Bulbs placed"}>
                {lights.estimated ? "≈ " : ""}
                {lights.count} {lights.count === 1 ? "bulb" : "bulbs"}
              </span>
            )
            : hidden > 0
              ? <span className="region-flag" title={`${hidden}% passes behind objects — no lights there`}>{hidden}% hidden</span>
              : null;
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
          {flag}
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
            title="Delete (Delete key)"
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
                    value={region.side ?? ""}
                    onClick={(event) => event.stopPropagation()}
                    onChange={(event) => onPatch(region.id, { side: (event.target.value || null) as Side | null })}
                  >
                    {region.side == null && <option value="">no side</option>}
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

              <span
                className="source"
                title={region.source === "auto" ? `Detection confidence ${Math.round((region.confidence ?? region.score) * 100)}%${sourceNote(region)}` : undefined}
              >
                {region.source === "auto" ? "found automatically" : region.source === "inferred" ? "continued behind an object" : "drawn by you"}
              </span>
            </div>

            {hidden > 0 && (
              <p className="muted small hidden-note">{hidden}% passes behind objects — no lights there</p>
            )}
            {region.kind === "roof" && region.edge === "unknown" && (
              <p className="small unknown-note">Edge type not identified. Choose eave, rake, hip or ridge.</p>
            )}
            {measured && <LengthNote line={measured} />}
            {lights && (
              <Lights
                region={region}
                lights={lights}
                onOverride={onOverride ? (override) => onOverride(region.id, override) : undefined}
              />
            )}
            {measured && needsOverride(measured) && onOverride && (
              <OverrideLength
                key={region.id}
                current={measured.length_in}
                onSave={(length_in) => onOverride(region.id, { length_in })}
                onClear={measured.method === "override" ? () => onOverride(region.id, null) : undefined}
              />
            )}
          </>
        )}
      </li>
    );
  };

  const shownEdges = LIT_EDGES.filter(({ edge }) => lit.some((line) => line.edge === edge));
  const litRoof = roofLines.filter((region) => region.visible).length;

  // Select-first: with a line or area selected, the panel is about that one thing (its type, side,
  // length and lights), as an inspector; the whole list is one step back.
  const selected = activeId ? regions.find((region) => region.id === activeId) : undefined;
  if (selected) {
    const where =
      selected.role === "blocker"
        ? "Obstacle in front of the house"
        : selected.geometry === "line"
          ? selected.visible ? "Line to light" : "Roof edge, not lit"
          : "Area";
    return (
      <div className="region-panel inspector">
        <div className="inspector-head">
          <button className="link" onClick={() => onSelect(null)} title="Back to every line and area (Esc)">
            ← All lines & areas
          </button>
          <span className="muted small">{where}</span>
        </div>
        <ul>{row(selected)}</ul>
        <p className="muted small inspector-keys">
          Delete key removes it · Esc goes back to the list · Ctrl+Z undoes a change
        </p>
      </div>
    );
  }

  return (
    <div className="region-panel">
      {house && (
        <div className="house-card">
          <div className="house-card-head">
            <strong>The house</strong>
            <button className="link" onClick={house.onClear}>
              Clear
            </button>
          </div>
          <p className="muted small">
            {house.roofs} {house.roofs === 1 ? "roof" : "roofs"} · {roofLines.length} roof {roofLines.length === 1 ? "line" : "lines"}, {litRoof} lit
          </p>
          <div className="house-card-actions">
            <button
              disabled={litRoof === roofLines.length}
              onClick={() => roofLines.forEach((region) => !region.visible && onPatch(region.id, { visible: true }))}
            >
              Light all its lines
            </button>
            <button disabled={!litRoof} onClick={() => roofLines.forEach((region) => region.visible && onPatch(region.id, { visible: false }))}>
              Light none
            </button>
            {house.onChangeRoofs && (
              <button title="Tap roofs in or out of the house (House tool)" onClick={house.onChangeRoofs}>
                Change roofs
              </button>
            )}
          </div>
        </div>
      )}

      <ScaleSummary
        roofLines={roofLines}
        geometry={geometry}
        measurement={measurement}
        lens={lens}
        onMeasure={onMeasure}
        onRemeasure={onRemeasure}
        onSelect={onSelect}
        activeId={activeId}
        edited={referenceEdited}
        warning={scaleWarning}
        checks={scaleChecks}
        side={referenceSide}
      />

      <div className="panel-bar">
        <span className="muted small">
          {lit.length} {lit.length === 1 ? "line" : "lines"} to light · {areas.length} {areas.length === 1 ? "area" : "areas"}
        </span>
        <button
          className={open.add ? "add-toggle active" : "add-toggle"}
          aria-expanded={open.add}
          onClick={() => setOpen((current) => ({ ...current, add: !current.add }))}
        >
          + Add
        </button>
      </div>
      {open.add && (
        <div className="add-panel">
          {drawingKind ? (
            <p className="drawing-hint">
              Click along the {KIND_LABELS[drawingKind].toLowerCase()} on the photo. Enter to finish, Esc to cancel.
            </p>
          ) : (
            <p className="muted small">Pick what to add: a line is drawn by clicking along it, an area is painted with the brush.</p>
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
        </div>
      )}

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
        <p className="muted small">None yet. Use + Add above, then click along the roofline to draw one.</p>
      )}

      <h2>
        Areas <span className="count">{areas.length}</span>
      </h2>
      {areas.length ? (
        <ul>{areas.map(row)}</ul>
      ) : (
        <p className="muted small">None yet. Use + Add above, or Add object on the photo.</p>
      )}

      {(unlit.length > 0 || blockers.length > 0) && (
        <Section
          title="More"
          count={unlit.length + blockers.length}
          open={open.more}
          onToggle={(value) => setOpen((current) => ({ ...current, more: value }))}
        >
          {unlit.length > 0 && (
            <div className="more-group">
              <h3>
                Other roof edges, not lit <span className="count">{unlit.length}</span>
              </h3>
              <p className="muted small">Found on the roof but not asked for. Use ○ to light one.</p>
              <ul>{unlit.map(row)}</ul>
            </div>
          )}
          {blockers.length > 0 && (
            <div className="more-group">
              <h3>
                Obstacles in front of the house <span className="count">{blockers.length}</span>
              </h3>
              <p className="muted small">
                Found so the roofline can continue behind trees, palms and cars. Never lit. If a line joins wrongly,
                correct the obstacle here.
              </p>
              <label className="check">
                <input type="checkbox" checked={showObstacles} onChange={(event) => onShowObstacles(event.target.checked)} />
                Show on the photo
              </label>
              <ul>{blockers.map(row)}</ul>
            </div>
          )}
        </Section>
      )}
    </div>
  );
}
