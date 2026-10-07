import { useEffect, useRef } from "react";

export interface Confirmation {
  title: string;
  body: string;
  confirm: string;
  // A red button: the action throws work away.
  danger?: boolean;
  run: () => void;
}

interface Props {
  confirmation: Confirmation;
  onClose: () => void;
}

// Asked before an action that cannot be stepped back from (detecting again, approving, deleting a
// project). Focus starts on Cancel, so Enter never confirms by accident; Esc and the backdrop cancel.
export function ConfirmDialog({ confirmation, onClose }: Props) {
  const cancelRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    cancelRef.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.stopPropagation();
        onClose();
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => {
      window.removeEventListener("keydown", onKey, true);
      previous?.focus?.();
    };
  }, []);

  return (
    <div className="dialog-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <div className="dialog" role="alertdialog" aria-modal="true" aria-labelledby="confirm-title" aria-describedby="confirm-body">
        <h2 id="confirm-title">{confirmation.title}</h2>
        <p id="confirm-body">{confirmation.body}</p>
        <div className="dialog-actions">
          <button ref={cancelRef} onClick={onClose}>
            Cancel
          </button>
          <button
            className={confirmation.danger ? "danger-button" : "primary"}
            onClick={() => {
              onClose();
              confirmation.run();
            }}
          >
            {confirmation.confirm}
          </button>
        </div>
      </div>
    </div>
  );
}
