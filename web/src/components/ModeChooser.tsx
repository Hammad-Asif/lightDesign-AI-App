import type { MaskingMode } from "../types";

interface Props {
  busy: boolean;
  onChoose: (mode: MaskingMode) => void;
}

export function ModeChooser({ busy, onChoose }: Props) {
  return (
    <div className="panel centered">
      <h1>Choose masking mode</h1>
      <p className="muted">Either way, every region stays editable before you approve.</p>

      <div className="mode-grid">
        <button className="mode-card" disabled={busy} onClick={() => onChoose("automatic")}>
          <strong>Automatic</strong>
          <span>
            Detects roofline, hedges, shrubs, topiaries, palm trunks, palm fronds and tree
            canopies as separate maskable regions.
          </span>
        </button>

        <button className="mode-card" disabled={busy} onClick={() => onChoose("manual")}>
          <strong>Manual</strong>
          <span>Start from a blank region and draw the mask yourself with the brush.</span>
        </button>
      </div>

      {busy && <p className="muted">Detecting regions…</p>}
    </div>
  );
}
