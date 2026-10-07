import { useEffect, useRef } from "react";

const GROUPS: { title: string; keys: [string, string][] }[] = [
  {
    title: "Tools",
    keys: [
      ["S", "Select"],
      ["H", "House: tap roofs in or out"],
      ["A", "Add object"],
      ["F", "Find lines in a box"],
      ["B", "Brush"],
      ["E", "Erase"],
      ["V", "Points of a line"],
      ["M", "Move or extend a line"],
      ["R", "Measure"],
      ["W", "Wreath"],
      ["[  ]", "Brush smaller / bigger"],
    ],
  },
  {
    title: "Editing",
    keys: [
      ["Delete", "Delete the selected line or area"],
      ["Esc", "Clear the selection, or stop drawing"],
      ["Enter", "Finish the line being drawn"],
      ["Ctrl+Z", "Undo"],
      ["Ctrl+Y", "Redo (also Shift+Ctrl+Z)"],
    ],
  },
  {
    title: "View",
    keys: [
      ["Wheel", "Zoom about the pointer"],
      ["Pinch", "Zoom on a tablet; two fingers move round"],
      ["Space + drag", "Move round a zoomed photo"],
      ["N", "Night preview"],
      ["?", "This list"],
    ],
  },
];

// Every keyboard shortcut in one place, from "?" or the dock's "?" button.
export function ShortcutSheet({ onClose }: { onClose: () => void }) {
  const closeRef = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    closeRef.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape" || event.key === "?") {
        event.stopPropagation();
        event.preventDefault();
        onClose();
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, []);

  return (
    <div className="dialog-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <div className="dialog shortcuts" role="dialog" aria-modal="true" aria-labelledby="shortcuts-title">
        <div className="dialog-head">
          <h2 id="shortcuts-title">Keyboard shortcuts</h2>
          <button ref={closeRef} className="icon-button" aria-label="Close" onClick={onClose}>
            ✕
          </button>
        </div>
        <div className="shortcut-groups">
          {GROUPS.map((group) => (
            <section key={group.title}>
              <h3>{group.title}</h3>
              <dl>
                {group.keys.map(([key, what]) => (
                  <div key={key}>
                    <dt>
                      <kbd>{key}</kbd>
                    </dt>
                    <dd>{what}</dd>
                  </div>
                ))}
              </dl>
            </section>
          ))}
        </div>
      </div>
    </div>
  );
}
