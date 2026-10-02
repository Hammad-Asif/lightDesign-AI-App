import type { Edge, Region } from "../types";
import { EDGE_COLORS } from "./RegionPanel";

// The types a salesperson picks for a hand-drawn roof line. Junction and unknown stay in the
// sidebar's full list; a drawn line is one of these.
const PICKABLE: Edge[] = ["eave", "rake", "hip", "ridge", "valley"];

// Shown right after a roof line is drawn: the server's suggestion is already selected (from the
// house's vanishing points and the lines its ends joined), and one tap confirms or changes it.
// The real length appears as soon as the line has a type.
export function EdgePicker({ region, suggested, onPick, onDone }: {
  region: Region;
  // Still the server's suggestion: nobody has picked yet.
  suggested: boolean;
  onPick: (edge: Edge) => void;
  onDone: () => void;
}) {
  return (
    <div className="edge-picker" role="group" aria-label="New roof line type">
      <span>New roof line. What is it?</span>
      {PICKABLE.map((edge) => (
        <button
          key={edge}
          className={region.edge === edge ? "chip on" : "chip"}
          aria-pressed={region.edge === edge}
          onClick={() => (region.edge === edge ? onDone() : onPick(edge))}
        >
          <span className="swatch" style={{ background: EDGE_COLORS[edge] }} />
          {edge}
        </button>
      ))}
      {suggested && region.edge && PICKABLE.includes(region.edge) && <span className="muted">suggested from the house</span>}
      <div className="spacer" />
      <button className="link" onClick={onDone}>
        Done
      </button>
    </div>
  );
}
