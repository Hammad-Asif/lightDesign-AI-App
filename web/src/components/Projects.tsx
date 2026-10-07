import { useEffect, useRef, useState } from "react";

import * as api from "../api";
import { thumbnail, thumbnailNow } from "../thumbnails";
import type { Photo, ProjectCard, ProjectStatus } from "../types";
import type { Confirmation } from "./ConfirmDialog";

const PAGE = 10;
const SEARCH_WAIT_MS = 300;

interface Props {
  userId: string;
  busy: boolean;
  onOpen: (id: string) => void;
  onNew: () => void;
  // Deleting a project is asked first, in the app's own dialog.
  onConfirm: (confirmation: Confirmation) => void;
  onError: (message: string) => void;
}

// The unfiltered list is read once per sign-in and kept here: going back to it from a photo shows
// it at once, with the open photo's card brought up to date by the app (rememberProject). A search
// or a status filter is read from the server each time.
type Loaded = { userId: string; items: ProjectCard[]; next: number | null; total: number };
let loaded: Loaded | null = null;

export function forgetProjects() {
  loaded = null;
}

// A photo just uploaded (its card goes on top, as the server lists it) or worked on (its card is
// brought up to date where it is).
export function rememberProject(photo: Photo) {
  if (!loaded) return;
  const known = loaded.items.find((item) => item.id === photo.id);
  const card: ProjectCard = {
    id: photo.id,
    prompt: photo.prompt,
    name: photo.name ?? "",
    mode: photo.mode,
    approved: photo.approved,
    updated_at: new Date().toISOString(),
    thumbnail_url: known?.thumbnail_url ?? `/photos/${photo.id}/thumbnail`,
  };
  loaded = {
    ...loaded,
    items: known ? loaded.items.map((item) => (item.id === photo.id ? { ...item, ...card } : item)) : [card, ...loaded.items],
    total: known ? loaded.total : loaded.total + 1,
    // The server's pages shift by one with a new photo on top.
    next: known || loaded.next === null ? loaded.next : loaded.next + 1,
  };
}

function statusOf(card: ProjectCard): ProjectStatus {
  return card.approved ? "locked" : card.mode ? "in_progress" : "not_started";
}

const STATUS: Record<ProjectStatus, { text: string; className: string }> = {
  locked: { text: "Locked", className: "badge ok" },
  in_progress: { text: "In progress", className: "badge" },
  not_started: { text: "Not started", className: "badge warn" },
};

const FILTERS: { value: ProjectStatus | null; label: string }[] = [
  { value: null, label: "All" },
  { value: "not_started", label: "Not started" },
  { value: "in_progress", label: "In progress" },
  { value: "locked", label: "Locked" },
];

// "Edited 3 days ago": how long since the project was last worked on.
function edited(iso?: string | null): string | null {
  if (!iso) return null;
  const seconds = (Date.now() - new Date(iso).getTime()) / 1000;
  if (!Number.isFinite(seconds)) return null;
  const units: [number, string][] = [[31536000, "year"], [2592000, "month"], [604800, "week"], [86400, "day"], [3600, "hour"], [60, "minute"]];
  for (const [size, unit] of units) {
    const count = Math.floor(seconds / size);
    if (count >= 1) return `Edited ${count} ${unit}${count === 1 ? "" : "s"} ago`;
  }
  return "Edited just now";
}

const titleOf = (card: ProjectCard) => card.name?.trim() || card.prompt?.trim() || "Untitled project";

function Thumbnail({ path, onMissing }: { path: string; onMissing: () => void }) {
  const [src, setSrc] = useState<string | null>(() => thumbnailNow(path));
  useEffect(() => {
    if (src) return;
    let current = true;
    thumbnail(path).then(
      (url) => current && setSrc(url),
      () => current && onMissing(),
    );
    return () => {
      current = false;
    };
  }, [path]);
  return src ? <img src={src} alt="" onError={onMissing} /> : <span className="project-thumb-wait" aria-hidden="true" />;
}

function Card({ card, busy, onOpen, onMissing, onRename, onDelete }: {
  card: ProjectCard;
  busy: boolean;
  onOpen: () => void;
  onMissing: () => void;
  onRename: (name: string) => void;
  onDelete: () => void;
}) {
  const [renaming, setRenaming] = useState(false);
  const [draft, setDraft] = useState(card.name ?? "");
  const status = STATUS[statusOf(card)];
  const when = edited(card.updated_at ?? card.created_at);

  const save = () => {
    setRenaming(false);
    if (draft.trim() !== (card.name ?? "").trim()) onRename(draft.trim());
  };

  return (
    <li className="project-item">
      <button className="project-card" disabled={busy} onClick={onOpen} title={`Open ${titleOf(card)}`}>
        <Thumbnail path={card.thumbnail_url} onMissing={onMissing} />
      </button>
      <div className="project-meta">
        {renaming ? (
          <input
            className="project-rename"
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
                setDraft(card.name ?? "");
                setRenaming(false);
              }
            }}
          />
        ) : (
          <button className="project-title" onClick={onOpen} disabled={busy} title={titleOf(card)}>
            {titleOf(card)}
          </button>
        )}
        <div className="project-sub">
          <span className={status.className}>{status.text}</span>
          {when && <span className="muted small">{when}</span>}
          <span className="project-actions">
            <button className="link" disabled={busy} onClick={() => setRenaming(true)}>
              Rename
            </button>
            <button className="link danger-link" disabled={busy} onClick={onDelete}>
              Delete
            </button>
          </span>
        </div>
        {card.name?.trim() && card.prompt?.trim() && <p className="project-prompt muted small">{card.prompt}</p>}
      </div>
    </li>
  );
}

// The home screen: this account's photos, newest first, ten at a time, with a search, a status
// filter, and a plain way to start a new one. A photo whose image is gone has no card.
export function Projects({ userId, busy, onOpen, onNew, onConfirm, onError }: Props) {
  const [status, setStatus] = useState<ProjectStatus | null>(null);
  const [search, setSearch] = useState("");
  const [query, setQuery] = useState("");
  const filtered = Boolean(status || query.trim());
  const [state, setState] = useState<Loaded | null>(() => (loaded?.userId === userId ? loaded : null));
  const [error, setError] = useState<string | null>(null);
  const [fetching, setFetching] = useState(false);
  const [missing, setMissing] = useState<Set<string>>(() => new Set());
  const request = useRef(0);

  const load = (offset: number, filter = { status, q: query }) => {
    const id = ++request.current;
    const isFiltered = Boolean(filter.status || filter.q.trim());
    setFetching(true);
    setError(null);
    api
      .listProjects(offset, PAGE, filter)
      .then((page) => {
        if (id !== request.current) return; // a newer search answered first
        setState((current) => {
          const base = offset > 0 && current ? current.items : [];
          const seen = new Set(base.map((item) => item.id));
          const next = { userId, items: [...base, ...page.items.filter((item) => !seen.has(item.id))], next: page.next_offset, total: page.total };
          if (!isFiltered) loaded = next;
          return next;
        });
      })
      .catch((cause) => id === request.current && setError(cause instanceof Error ? cause.message : String(cause)))
      .finally(() => id === request.current && setFetching(false));
  };

  // The search waits for a pause in typing.
  useEffect(() => {
    const timer = window.setTimeout(() => setQuery(search), SEARCH_WAIT_MS);
    return () => window.clearTimeout(timer);
  }, [search]);

  useEffect(() => {
    if (!filtered && loaded?.userId === userId) {
      request.current += 1;
      setState(loaded);
      return;
    }
    setState(null);
    load(0, { status, q: query });
  }, [userId, status, query]);

  const drop = (id: string) => {
    setMissing((current) => new Set(current).add(id));
    if (loaded) loaded = { ...loaded, items: loaded.items.filter((item) => item.id !== id), total: Math.max(0, loaded.total - 1) };
    setState((current) => (current ? { ...current, total: Math.max(0, current.total - 1) } : current));
  };

  const rename = (card: ProjectCard, name: string) => {
    api.renamePhoto(card.id, name).then(
      () => {
        const update = (items: ProjectCard[]) => items.map((item) => (item.id === card.id ? { ...item, name } : item));
        if (loaded) loaded = { ...loaded, items: update(loaded.items) };
        setState((current) => (current ? { ...current, items: update(current.items) } : current));
      },
      (cause) => onError(cause instanceof Error ? cause.message : String(cause)),
    );
  };

  const remove = (card: ProjectCard) =>
    onConfirm({
      title: `Delete ${titleOf(card)}?`,
      body: "The photo, its lines and areas, and its lighting are deleted for good. This cannot be undone.",
      confirm: "Delete project",
      danger: true,
      run: () => {
        api.deletePhoto(card.id).then(
          () => drop(card.id),
          (cause) => onError(cause instanceof Error ? cause.message : String(cause)),
        );
      },
    });

  const cards = (state?.items ?? []).filter((item) => !missing.has(item.id));

  return (
    <div className="projects">
      <div className="projects-head">
        <div>
          <h1>Your projects</h1>
          <p className="muted small">
            Pick up a mock-up where you left it, or start a new one.
            {state && state.total > 0 && !filtered ? ` ${state.total} ${state.total === 1 ? "project" : "projects"}.` : ""}
          </p>
        </div>
        <button className="primary" disabled={busy} onClick={onNew}>
          + New photo
        </button>
      </div>

      <div className="projects-tools">
        <input
          type="search"
          className="projects-search"
          aria-label="Search projects"
          placeholder="Search by name, address or instruction"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
        <div className="chips" role="group" aria-label="Show">
          {FILTERS.map((filter) => (
            <button
              key={filter.label}
              className={status === filter.value ? "chip active" : "chip"}
              aria-pressed={status === filter.value}
              onClick={() => setStatus(filter.value)}
            >
              {filter.label}
            </button>
          ))}
        </div>
      </div>

      {error && <p className="form-error">{error}</p>}
      {!state && !error && (
        <ul className="project-grid" aria-hidden="true">
          {Array.from({ length: 4 }, (_, i) => (
            <li key={i} className="project-item skeleton">
              <span className="project-thumb-wait" />
              <span className="skeleton-line" />
            </li>
          ))}
        </ul>
      )}
      {state && cards.length === 0 && state.next === null && (
        filtered ? (
          <div className="projects-empty">
            <p>No projects match.</p>
            <button
              onClick={() => {
                setSearch("");
                setQuery("");
                setStatus(null);
              }}
            >
              Show all projects
            </button>
          </div>
        ) : (
          <div className="projects-empty">
            <p>No projects yet.</p>
            <button className="primary" onClick={onNew}>
              Start with a photo of the house
            </button>
          </div>
        )
      )}

      {cards.length > 0 && (
        <ul className="project-grid">
          {cards.map((card) => (
            <Card
              key={card.id}
              card={card}
              busy={busy}
              onOpen={() => onOpen(card.id)}
              onMissing={() => drop(card.id)}
              onRename={(name) => rename(card, name)}
              onDelete={() => remove(card)}
            />
          ))}
        </ul>
      )}

      {state && state.next !== null && (
        <div className="projects-more">
          <button disabled={fetching} onClick={() => load(state.next!)}>
            {fetching ? "Loading…" : "Load more"}
          </button>
        </div>
      )}
    </div>
  );
}
