import { useState } from "react";

import { Specification } from "./Specification";
import type { MaskingMode, PaletteColour, Photo, Product } from "../types";

interface Props {
  photo: Photo;
  products: Product[];
  palette: PaletteColour[];
  busy: boolean;
  onInterpret: (prompt: string) => void;
  onChoose: (mode: MaskingMode) => void;
}

export function ModeChooser({ photo, products, palette, busy, onInterpret, onChoose }: Props) {
  const [prompt, setPrompt] = useState(photo.prompt);
  const requests = photo.interpreted?.requests ?? [];
  const read = photo.interpreted?.read_with ?? null;
  const changed = prompt.trim() !== photo.prompt;
  const detectable = photo.interpreted?.matched ?? false;

  return (
    <div className="panel wide">
      <h1>Check the instruction</h1>
      <p className="lede">This is how it was read. Correct the wording here, or the details later in the editor.</p>

      <label className="field">
        <span>Where do the lights go?</span>
        <textarea rows={3} value={prompt} onChange={(event) => setPrompt(event.target.value)} />
      </label>
      {changed && (
        <button disabled={busy} onClick={() => onInterpret(prompt)}>
          Read it again
        </button>
      )}

      {read && (
        <p className="muted small spec-note">
          {read.sentence}
          {read.unclear.length > 0 && ` Not read: ${read.unclear.join("; ")}.`}
          {read.dropped.length > 0 && ` Left out: ${read.dropped.join("; ")}.`}
        </p>
      )}

      {requests.length > 0 ? (
        <Specification requests={requests} products={products} palette={palette} />
      ) : (
        <p className="muted small">No lighting was recognised in the instruction.</p>
      )}
      {requests.length > 0 && !detectable && (
        <p className="muted small">
          Nothing here is found automatically — these are drawn by hand. Automatic mode would look for every area type.
        </p>
      )}

      <h2 className="spec-heading">How should the areas be marked?</h2>
      <div className="mode-grid">
        <button className="mode-card" disabled={busy} onClick={() => onChoose("automatic")}>
          <strong>Automatic</strong>
          <span>
            {detectable
              ? "Finds the areas named above, and marks anything standing in front of them."
              : "Finds the roofline, hedges, shrubs, topiaries, palm trunks, palm fronds and tree canopies, and marks anything standing in front of them."}
          </span>
        </button>

        <button className="mode-card" disabled={busy} onClick={() => onChoose("manual")}>
          <strong>Manual</strong>
          <span>Draw every line and paint every area yourself, with nothing detected first.</span>
        </button>
      </div>

      {busy && <p className="muted small">Working…</p>}
    </div>
  );
}
