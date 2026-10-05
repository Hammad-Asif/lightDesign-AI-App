import type { ScaleDebug } from "../types";

// Development (SCALE_DEBUG): the last scale solve, step by step. Picking a step draws its overlay
// over the photo (services/scale_debug.py writes one per step); selecting a line shows its whole
// record through the solve. Shown only when the server returned debug data.

const STEP_TITLES: Record<string, string> = {
  "1": "Camera",
  "2": "Sides",
  "3": "Measurement",
  "4": "Walls",
  "5": "Pitch",
  "6": "Anchors",
  "7": "Choice",
  "8": "Checks",
  "9": "Lengths",
};

// The overlay file for a step: the step's number picks it (9_result draws scale_9_lengths.png).
export function stepFile(debug: ScaleDebug, step: string): string | null {
  const files = debug.files ?? {};
  const digit = step.split("_")[0];
  const name = Object.keys(files).find((file) => file.startsWith(`scale_${digit}_`) && file.endsWith(".png"));
  return name ? files[name] : null;
}

function Value({ value }: { value: unknown }) {
  if (value === null || value === undefined) return <span className="muted">none</span>;
  if (typeof value === "object") return <pre className="debug-json">{JSON.stringify(value, null, 2)}</pre>;
  return <span>{String(value)}</span>;
}

interface Props {
  debug: ScaleDebug;
  step: string | null;
  onStep: (step: string | null) => void;
  activeId: string | null;
  onSelect: (id: string) => void;
}

export function ScaleDebugPanel({ debug, step, onStep, activeId, onSelect }: Props) {
  // An API still running older code sends a payload without some of these: shown as empty, never a crash.
  const stepRecords = debug.steps ?? {};
  const lines = debug.lines ?? [];
  const numbers = debug.numbers ?? {};
  const steps = Object.keys(stepRecords).sort();
  const record = step ? stepRecords[step] ?? null : null;
  const line = lines.find((entry) => entry.id === activeId);
  // A step's record names lines by id: shown with their numbers so they match the overlay.
  const named = (value: unknown) =>
    JSON.parse(JSON.stringify(value ?? null), (_, v) => (typeof v === "string" && numbers[v] ? `${numbers[v]} (${v.slice(0, 8)})` : v));

  return (
    <section className="scale-debug">
      <h3>
        Scale debug <span className="muted small">{debug.written_at}</span>
      </h3>
      {steps.length === 0 && <p className="muted small">No steps recorded. Restart the API so it runs the current scale code.</p>}
      <div className="chips" role="group" aria-label="Scale step">
        <button className={step ? "chip" : "chip on"} aria-pressed={!step} onClick={() => onStep(null)}>
          Off
        </button>
        {steps.map((name) => (
          <button
            key={name}
            className={name === step ? "chip on" : "chip"}
            aria-pressed={name === step}
            title={name}
            disabled={!stepFile(debug, name) && name !== step}
            onClick={() => onStep(name)}
          >
            {name.split("_")[0]}. {STEP_TITLES[name.split("_")[0]] ?? name}
          </button>
        ))}
      </div>

      {record && (
        <details open className="debug-step">
          <summary>Step {step}</summary>
          <dl>
            {Object.entries(named(record) as Record<string, unknown>).map(([key, value]) => (
              <div key={key} className="debug-row">
                <dt>{key}</dt>
                <dd>
                  <Value value={value} />
                </dd>
              </div>
            ))}
          </dl>
        </details>
      )}

      <details open={Boolean(line)} className="debug-step">
        <summary>{line ? `${line.n} ${line.label ?? ""}` : "Select a line to see its record"}</summary>
        {line && (
          <dl>
            {Object.entries(named(line) as Record<string, unknown>)
              .filter(([key]) => !["id", "n", "label", "points3d"].includes(key))
              .map(([key, value]) => (
                <div key={key} className="debug-row">
                  <dt>{key}</dt>
                  <dd>
                    <Value value={value} />
                  </dd>
                </div>
              ))}
          </dl>
        )}
      </details>

      <details className="debug-step">
        <summary>Lines</summary>
        <ul className="debug-lines">
          {lines.map((entry) => (
            <li key={entry.id}>
              <button className={entry.id === activeId ? "link on" : "link"} onClick={() => onSelect(entry.id)}>
                {entry.n} {entry.label ?? String(entry.edge ?? "")}
              </button>{" "}
              <span className="muted small">
                {String(entry.method ?? "")} {entry.length_in != null ? `${Math.round(Number(entry.length_in))} in` : String(entry.needs ?? "")}
              </span>
            </li>
          ))}
        </ul>
      </details>
    </section>
  );
}
