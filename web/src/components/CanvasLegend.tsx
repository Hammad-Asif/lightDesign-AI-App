import { useState } from "react";

import type { Edge, Region } from "../types";
import { EDGE_COLORS } from "./RegionPanel";

const NAMES: Partial<Record<Edge, string>> = { eave: "Eave", rake: "Rake", hip: "Hip", ridge: "Ridge", valley: "Valley" };

// What the line colours and dashes on the photo mean, so a line type never rests on colour alone:
// the selected or hovered line is also named on the photo. Folds to one button.
export function CanvasLegend({ regions }: { regions: Region[] }) {
  const [open, setOpen] = useState(true);
  const roof = regions.filter((region) => region.kind === "roof" && region.geometry === "line");
  if (!roof.length) return null;
  const edges = (Object.keys(NAMES) as Edge[]).filter((edge) => roof.some((region) => region.edge === edge));
  const unknown = roof.some((region) => region.edge === "unknown");
  const unlit = roof.some((region) => !region.visible);

  return (
    <div className={open ? "canvas-legend" : "canvas-legend folded"}>
      <button className="legend-toggle" aria-expanded={open} onClick={() => setOpen(!open)} title="What the lines mean">
        Key {open ? "▾" : "▸"}
      </button>
      {open && (
        <ul>
          {edges.map((edge) => (
            <li key={edge}>
              <span className="legend-line" style={{ background: EDGE_COLORS[edge] }} />
              {NAMES[edge]}
            </li>
          ))}
          {unlit && (
            <li>
              <span className="legend-line dashed" />
              Not lit
            </li>
          )}
          {unknown && (
            <li>
              <span className="legend-line unknown" />
              Type to check
            </li>
          )}
          <li>
            <span className="legend-line short-dash" />
            Low confidence
          </li>
        </ul>
      )}
    </div>
  );
}
