import { useState } from "react";

import type { Photo } from "../types";

interface Props {
  photo: Photo;
  busy: boolean;
  onRename: (name: string) => void;
}

// The open project's name in the header's breadcrumb; a click turns it into a field to rename it.
// A photo with no name yet shows its instruction, and says it can be named.
export function ProjectName({ photo, busy, onRename }: Props) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(photo.name ?? "");
  const shown = photo.name?.trim() || photo.prompt?.trim() || "Untitled project";

  if (editing) {
    const save = () => {
      setEditing(false);
      if (draft.trim() !== (photo.name ?? "").trim()) onRename(draft.trim());
    };
    return (
      <input
        className="crumb-input"
        aria-label="Project name"
        autoFocus
        value={draft}
        maxLength={200}
        placeholder="The customer or the address"
        onChange={(event) => setDraft(event.target.value)}
        onBlur={save}
        onKeyDown={(event) => {
          if (event.key === "Enter") save();
          if (event.key === "Escape") {
            event.stopPropagation();
            setDraft(photo.name ?? "");
            setEditing(false);
          }
        }}
      />
    );
  }
  return (
    <button
      className={photo.name?.trim() ? "crumb-name" : "crumb-name unnamed"}
      disabled={busy}
      title={photo.name?.trim() ? "Rename this project" : "Name this project (the customer or the address)"}
      onClick={() => {
        setDraft(photo.name ?? "");
        setEditing(true);
      }}
    >
      {shown}
    </button>
  );
}
