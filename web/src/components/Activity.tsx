import { useEffect, useState } from "react";
import { onActivity } from "../api";

// A small pill with a spinner and what the app is waiting on ("Detecting", "Measuring"), so a
// long call never looks like a frozen screen. Shown only after SHOW_AFTER_MS, so quick calls
// do not flicker it; "+n" when more calls run at once.
const SHOW_AFTER_MS = 200;

export function Activity() {
  const [labels, setLabels] = useState<string[]>([]);
  const [shown, setShown] = useState(false);

  useEffect(() => onActivity(setLabels), []);

  const active = labels.length > 0;
  useEffect(() => {
    if (!active) {
      setShown(false);
      return;
    }
    const timer = window.setTimeout(() => setShown(true), SHOW_AFTER_MS);
    return () => window.clearTimeout(timer);
  }, [active]);

  if (!active || !shown) return null;
  const label = labels[labels.length - 1];
  const more = labels.length - 1;
  return (
    <div className="activity" role="status" aria-live="polite">
      <span className="activity-spinner" aria-hidden="true" />
      <span className="activity-label">{label}</span>
      {more > 0 && <span className="activity-more">+{more}</span>}
    </div>
  );
}
