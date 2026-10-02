import { useEffect, useState } from "react";

import type { Measurement, PhotoDetail, Region, Side } from "../types";

export type MeasureMode = "points" | "line";

const SIDES: { value: Side; label: string }[] = [
  { value: "front", label: "Front" },
  { value: "left", label: "Left side" },
  { value: "right", label: "Right side" },
];

// To the whole inch, as a tape is read: 27 ft 5 in.
export function formatLength(inches: number): string {
  const total = Math.round(inches);
  const feet = Math.floor(total / 12);
  const rest = total - feet * 12;
  return rest ? `${feet} ft ${rest} in` : `${feet} ft`;
}

// Where the lens focal length came from, which decides how far depth-based numbers can be trusted.
function lensNote(photo: PhotoDetail): string {
  const camera = photo.interpreted?.camera;
  if (!camera) return "The lens is read from the photo when you save a measurement.";
  const mm = `${Math.round(camera.focal_35mm)} mm`;
  if (camera.focal_source === "vp") return `Lens read from the photo's perspective (about ${mm}).`;
  if (camera.focal_source === "exif") return `Lens from the photo's own data (${mm}).`;
  return `Lens assumed (${mm}): this photo shows no perspective to read it from. Lengths on the wall you measure are unaffected; other walls are estimates.`;
}

interface Props {
  photo: PhotoDetail;
  mode: MeasureMode;
  setMode: (mode: MeasureMode) => void;
  // Two-points mode: the points tapped so far (0, 1 or 2).
  points: number[][];
  onResetPoints: () => void;
  // Line mode: the selected line, if it is one.
  line: Region | null;
  busy: boolean;
  onSave: (measurement: Measurement) => void;
  onClear: () => void;
}

export function MeasurePanel({ photo, mode, setMode, points, onResetPoints, line, busy, onSave, onClear }: Props) {
  const [feet, setFeet] = useState("");
  const [inches, setInches] = useState("");
  const [facade, setFacade] = useState<Side | "">("");
  const saved = photo.scale?.measurement ?? null;
  const facades = photo.interpreted?.camera?.facades;
  const choices = facades?.length ? SIDES.filter((side) => facades.includes(side.value)) : SIDES;

  useEffect(() => {
    setFacade("");
  }, [mode]);

  const total = (Number(feet) || 0) * 12 + (Number(inches) || 0);
  const reference = mode === "points" ? points.length === 2 : Boolean(line);
  const ready = reference && total > 0 && !busy;

  const save = () => {
    const base = { length_in: total, facade: facade || null };
    onSave(mode === "points" ? { ...base, points } : { ...base, region_id: line!.id });
  };

  return (
    <section className="measure-panel">
      <h3>Measurement</h3>
      <p className="muted small">
        One known length gives the photo real scale. The other sides of the house take theirs through the corners they
        share with this one.
      </p>

      <div className="chips" role="group" aria-label="What to measure">
        <button className={mode === "points" ? "chip on" : "chip"} aria-pressed={mode === "points"} onClick={() => setMode("points")}>
          Two points
        </button>
        <button className={mode === "line" ? "chip on" : "chip"} aria-pressed={mode === "line"} onClick={() => setMode("line")}>
          Along a line
        </button>
      </div>

      {mode === "points" ? (
        <p className="small">
          {points.length === 0 && "Tap one end of a feature you know the size of, such as a garage door."}
          {points.length === 1 && "Now tap its other end."}
          {points.length === 2 && (
            <>
              Two points set.{" "}
              <button className="link" onClick={onResetPoints}>
                Tap again
              </button>
            </>
          )}
        </p>
      ) : (
        <p className="small">{line ? `Measuring: ${line.label}` : "Tap a line on the photo, such as an eave."}</p>
      )}

      <div className="measure-length">
        <label>
          <input type="number" min="0" value={feet} onChange={(event) => setFeet(event.target.value)} /> ft
        </label>
        <label>
          <input type="number" min="0" max="11.9" step="0.5" value={inches} onChange={(event) => setInches(event.target.value)} /> in
        </label>
      </div>

      <label className="measure-side">
        Side of the house
        <select value={facade} onChange={(event) => setFacade(event.target.value as Side | "")}>
          <option value="">From the perspective</option>
          {choices.map((side) => (
            <option key={side.value} value={side.value}>
              {side.label}
            </option>
          ))}
        </select>
      </label>

      <div className="measure-actions">
        <button className="primary" disabled={!ready} onClick={save}>
          Save measurement
        </button>
        {saved && (
          <button disabled={busy} onClick={onClear}>
            Clear
          </button>
        )}
      </div>

      {saved && (
        <p className="small">
          Saved: {formatLength(saved.length_in)} on the {saved.facade ?? "front"} side.
        </p>
      )}
      <p className="muted small">{lensNote(photo)}</p>
    </section>
  );
}
