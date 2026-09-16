import type { MaskingMode } from "../types";

interface Props {
  busy: boolean;
  onChoose: (mode: MaskingMode) => void;
}

export function ModeChooser({ busy, onChoose }: Props) {
  return (
    <div className="panel">
      <h1>Choose masking mode</h1>
      <p className="lede">Either way, every area stays editable until you approve it.</p>

      <div className="mode-grid">
        <button className="mode-card" disabled={busy} onClick={() => onChoose("automatic")}>
          <strong>Automatic</strong>
          <span>
            Finds the roofline, hedges, shrubs, topiaries, palm trunks, palm fronds and tree
            canopies, and marks anything standing in front of them.
          </span>
        </button>

        <button className="mode-card" disabled={busy} onClick={() => onChoose("manual")}>
          <strong>Manual</strong>
          <span>Draw every line and paint every area yourself, with nothing detected first.</span>
        </button>
      </div>

      {busy && <p className="muted small">Finding the areas in your photo…</p>}
    </div>
  );
}
