import { useEffect, useState } from "react";

import type { Measurement, PhotoDetail, PhotoGeometry, Region, ScaleCheckInput, Side } from "../types";

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

// "±6%" when a length's spread (how far it moves under the photo's own errors) is worth saying.
export function spreadText(spread: number | null | undefined): string {
  return spread != null && spread >= 0.03 ? ` ±${Math.round(spread * 100)}%` : "";
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
  // Measure again with the last saved values: the same points or line, length and side.
  onRemeasure?: (saved: Measurement) => void;
  // A second known length on a line, checking the measurement (null clears it), and what the
  // solve made of it.
  onCheck?: (check: ScaleCheckInput | null) => void;
  checkResult?: PhotoGeometry["reference_check"];
  // How far the saved measurement can carry the house, and what the doors' heights say of it.
  referenceQuality?: PhotoGeometry["reference_quality"];
  scaleCheck?: PhotoGeometry["scale_check"];
  // The line whose length would help most, and selecting it to measure.
  nextMeasurement?: PhotoGeometry["next_measurement"];
  onSelectLine?: (id: string) => void;
}

// Below this many pixels a reference's own tapping error is several percent of every length.
const SHORT_REFERENCE_PX = 100;

function referencePixels(mode: MeasureMode, points: number[][], line: Region | null): number {
  const span = (a: number[], b: number[]) => Math.hypot(b[0] - a[0], b[1] - a[1]);
  if (mode === "points") return points.length === 2 ? span(points[0], points[1]) : 0;
  const polyline = line?.polyline;
  if (!polyline || polyline.length < 2) return 0;
  return polyline.slice(1).reduce((sum, point, i) => sum + span(polyline[i], point), 0);
}

// The saved measurement as it is sent again: a side read from the photo is left to be read again.
export function remeasured(saved: Measurement): Measurement {
  const again: Measurement = { length_in: saved.length_in, facade: saved.facade_given ? saved.facade ?? null : null };
  return saved.region_id ? { ...again, region_id: saved.region_id } : { ...again, points: saved.points };
}

export function MeasurePanel({
  photo, mode, setMode, points, onResetPoints, line, busy, onSave, onClear, onRemeasure, onCheck, checkResult,
  referenceQuality, scaleCheck, nextMeasurement, onSelectLine,
}: Props) {
  const [feet, setFeet] = useState("");
  const [inches, setInches] = useState("");
  const [checkFeet, setCheckFeet] = useState("");
  const [checkInches, setCheckInches] = useState("");
  const savedCheck = photo.scale?.check ?? null;
  const checkTotal = (Number(checkFeet) || 0) * 12 + (Number(checkInches) || 0);
  const checkLabel = savedCheck ? photo.regions.find((region) => region.id === savedCheck.region_id)?.label : null;
  const [facade, setFacade] = useState<Side | "">("");
  // The side chosen with Re-measure survives the mode switch it causes.
  const [keepFacade, setKeepFacade] = useState(false);
  const saved = photo.scale?.measurement ?? null;
  const facades = photo.interpreted?.camera?.facades;
  const choices = facades?.length ? SIDES.filter((side) => facades.includes(side.value)) : SIDES;

  useEffect(() => {
    if (keepFacade) {
      setKeepFacade(false);
      return;
    }
    setFacade("");
  }, [mode]);

  const remeasure = () => {
    if (!saved || !onRemeasure) return;
    const total = Math.round(saved.length_in * 2) / 2;
    setFeet(String(Math.floor(total / 12)));
    setInches(String(total - Math.floor(total / 12) * 12 || ""));
    setFacade(saved.facade_given && saved.facade ? saved.facade : "");
    setKeepFacade((saved.region_id ? "line" : "points") !== mode);
    onRemeasure(saved);
  };

  const total = (Number(feet) || 0) * 12 + (Number(inches) || 0);
  const reference = mode === "points" ? points.length === 2 : Boolean(line);
  // Every length scales with the reference: a slip of a pixel at each end is 2 px of its length.
  const pixels = referencePixels(mode, points, line);
  const slip = pixels ? Math.round((200 / pixels) * 10) / 10 : 0;
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
      {pixels > 0 && pixels < SHORT_REFERENCE_PX && (
        <p className="small unknown-note">
          This reference is only {Math.round(pixels)} px long: a one-pixel slip at each end changes every length by about{" "}
          {slip}%. Measure something longer if you can, such as a garage door or a whole eave.
        </p>
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
        {saved && onRemeasure && (
          <button
            disabled={busy}
            title="Measure again with the last values: the same points or line, length and side"
            onClick={remeasure}
          >
            Re-measure
          </button>
        )}
        {saved && (
          <button disabled={busy} onClick={onClear}>
            Clear
          </button>
        )}
      </div>

      {saved && (
        <p className="small">
          Saved: {formatLength(saved.length_in)}{" "}
          {saved.facade_given ? `on the ${saved.facade} side.` : "with its side read from the photo."}
        </p>
      )}
      {saved && referenceQuality && referenceQuality.grade !== "good" && (
        <p className={referenceQuality.grade === "poor" ? "small unknown-note" : "muted small"}>
          {referenceQuality.grade === "poor" ? "Weak measurement" : "Fair measurement"}: {referenceQuality.reasons.join("; ")}.
        </p>
      )}
      {saved && scaleCheck && (
        <p className={scaleCheck.conflict ? "small unknown-note" : "small ok-text"}>
          {scaleCheck.conflict
            ? `The doors disagree: they suggest about ${formatLength(scaleCheck.suggested_in)} instead of ${formatLength(scaleCheck.typed_in)}.`
            : `The ${scaleCheck.sources.length === 1 ? "door agrees" : "doors agree"} with this measurement.`}
        </p>
      )}
      {saved && nextMeasurement && (
        <p className="small">
          {nextMeasurement.kind === "instead" ? "Better to measure" : "Measure next"}: {nextMeasurement.label ?? "this line"}{" "}
          ({nextMeasurement.why}).{" "}
          {onSelectLine && (
            <button className="link" disabled={busy} onClick={() => onSelectLine(nextMeasurement.line_id)}>
              Select it
            </button>
          )}
        </p>
      )}
      <p className="muted small">{lensNote(photo)}</p>

      {saved && onCheck && (
        <div className="measure-check">
          <h4>Check with a second length</h4>
          <p className="muted small">
            In "Along a line", tap another line whose length you know (a second eave, a garage door top) and type it. On a
            wall your measurement already reaches it only shows how far the photo's length is from yours; on a wall it does
            not reach, it measures that wall.
          </p>
          <p className="small">{line ? `Line: ${line.label}` : "No line selected."}</p>
          <div className="measure-length">
            <label>
              <input type="number" min="0" value={checkFeet} onChange={(event) => setCheckFeet(event.target.value)} /> ft
            </label>
            <label>
              <input type="number" min="0" max="11.9" step="0.5" value={checkInches}
                     onChange={(event) => setCheckInches(event.target.value)} /> in
            </label>
          </div>
          <div className="measure-actions">
            <button disabled={!line || checkTotal <= 0 || busy}
                    onClick={() => line && onCheck({ region_id: line.id, length_in: checkTotal })}>
              Check
            </button>
            {savedCheck && (
              <button disabled={busy} onClick={() => onCheck(null)}>
                Clear check
              </button>
            )}
          </div>
          {savedCheck && checkResult?.used && (
            <p className="small ok-text">
              {checkLabel ?? "The line"}: you {formatLength(checkResult.typed_in)}. No corner carries your first measurement to
              its wall, so this length measures that wall and the lines on it.
            </p>
          )}
          {savedCheck && checkResult && !checkResult.used && (
            <p className={checkResult.off != null && Math.abs(checkResult.off) <= 0.05 ? "small ok-text" : "small unknown-note"}>
              {checkLabel ?? "The line"}: you {formatLength(checkResult.typed_in)},{" "}
              {checkResult.solved_in == null
                ? "but the photo cannot measure that line, so it cannot check the scale."
                : `the photo ${formatLength(checkResult.solved_in)} (${checkResult.off! >= 0 ? "+" : ""}${Math.round(
                    checkResult.off! * 100,
                  )}%). ${Math.abs(checkResult.off!) <= 0.05 ? "The scale agrees." : "One of the two lengths, or its side, is off."}`}
            </p>
          )}
        </div>
      )}
    </section>
  );
}
