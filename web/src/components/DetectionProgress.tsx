import { useEffect, useState } from "react";

// What detection is doing, by time since it started: the server answers once, at the end, so the
// stages are its usual order and timing, not a live report.
const STAGES: [number, string][] = [
  [0, "Finding the roofs"],
  [6, "Finding what stands in front of the house"],
  [14, "Drawing the roof lines"],
  [32, "Joining corners and lines behind objects"],
  [55, "Almost there"],
];

interface Props {
  // "Detecting" for a whole photo, "Finding lines" in a box.
  label: string;
  // Find lines is one short step: no stages.
  staged: boolean;
}

// Over the whole app while detection runs: the photo dims, nothing can be edited mid-run, and the
// salesperson sees it is working and roughly how long it takes.
export function DetectionProgress({ label, staged }: Props) {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    const started = Date.now();
    const timer = window.setInterval(() => setSeconds(Math.floor((Date.now() - started) / 1000)), 500);
    return () => window.clearInterval(timer);
  }, []);
  const current = staged ? [...STAGES].reverse().find(([at]) => seconds >= at)! : null;
  const index = current ? STAGES.indexOf(current) : -1;

  return (
    <div className="progress-backdrop" role="status" aria-live="polite">
      <div className="progress-card">
        <span className="spinner" aria-hidden="true" />
        <strong>{label}</strong>
        {current && <p>{current[1]}…</p>}
        {staged && (
          <ol className="progress-steps" aria-hidden="true">
            {STAGES.slice(0, -1).map(([, text], i) => (
              <li key={text} className={i < index ? "done" : i === index ? "now" : ""} />
            ))}
          </ol>
        )}
        <p className="muted small">
          {seconds}s{staged ? " · usually under a minute" : ""}
        </p>
      </div>
    </div>
  );
}
