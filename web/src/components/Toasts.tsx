import { useEffect } from "react";

export interface Notice {
  text: string;
  // One thing to do about it, e.g. Undo after a delete.
  action?: { label: string; run: () => void };
}

const NOTICE_MS = 7000;
const ERROR_MS = 12000;

interface Props {
  notice: Notice | null;
  error: string | null;
  onDismissNotice: () => void;
  onDismissError: () => void;
}

// Messages appear in one place, bottom centre over the work, and go by themselves: a notice after
// seven seconds, an error after twelve (an error is also read out at once to a screen reader).
// Hovering or focusing one does not stop the timer; each has its own close button.
export function Toasts({ notice, error, onDismissNotice, onDismissError }: Props) {
  useEffect(() => {
    if (!notice) return;
    const timer = window.setTimeout(onDismissNotice, NOTICE_MS);
    return () => window.clearTimeout(timer);
  }, [notice]);
  useEffect(() => {
    if (!error) return;
    const timer = window.setTimeout(onDismissError, ERROR_MS);
    return () => window.clearTimeout(timer);
  }, [error]);

  return (
    <div className="toasts">
      <div role="alert" aria-live="assertive">
        {error && (
          <div className="toast error-toast">
            <span>{error}</span>
            <button className="toast-close" aria-label="Dismiss" onClick={onDismissError}>
              ✕
            </button>
          </div>
        )}
      </div>
      <div role="status" aria-live="polite">
        {notice && (
          <div className="toast">
            <span>{notice.text}</span>
            {notice.action && (
              <button
                className="toast-action"
                onClick={() => {
                  notice.action!.run();
                  onDismissNotice();
                }}
              >
                {notice.action.label}
              </button>
            )}
            <button className="toast-close" aria-label="Dismiss" onClick={onDismissNotice}>
              ✕
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
