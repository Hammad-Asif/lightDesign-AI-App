export interface Step {
  id: "detect" | "lines" | "measure" | "lighting" | "approve";
  label: string;
  // done: nothing left; todo: still to do; skip: not needed for this photo.
  state: "done" | "todo" | "skip";
  // What is left, in a few words (the tooltip, and the Next button's text).
  detail: string;
  run?: () => void;
}

interface Props {
  steps: Step[];
  busy: boolean;
}

// Where the photo stands, from detection to approval, and one button for the next thing to do.
// Each step can be clicked to go to it; a done step shows a tick.
export function StepBar({ steps, busy }: Props) {
  const next = steps.find((step) => step.state === "todo");
  return (
    <nav className="step-bar" aria-label="Progress">
      <ol>
        {steps.map((step, index) => (
          <li key={step.id} className={`step ${step.state}${next?.id === step.id ? " current" : ""}`}>
            <button
              disabled={busy || !step.run}
              title={step.detail}
              aria-current={next?.id === step.id ? "step" : undefined}
              onClick={step.run}
            >
              <span className="step-mark" aria-hidden="true">
                {step.state === "done" ? "✓" : step.state === "skip" ? "–" : index + 1}
              </span>
              <span className="step-label">{step.label}</span>
              <span className="visually-hidden">
                {step.state === "done" ? " (done)" : step.state === "skip" ? " (not needed)" : " (to do)"}
              </span>
            </button>
          </li>
        ))}
      </ol>
      {next?.run && (
        <button className="primary step-next" disabled={busy} onClick={next.run} title={next.detail}>
          Next: {next.label}
        </button>
      )}
    </nav>
  );
}
