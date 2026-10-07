import { useEffect, useRef, useState } from "react";

import type { User } from "../types";

interface Props {
  user: User;
  onOpenProducts: () => void;
  onSignOut: () => void;
  // Only shown inside the workspace, not on the entry screens.
  onReopen?: () => void;
}

// Account-level controls (who's signed in, the product library, sign out) live behind one menu so
// they never compete with the per-photo actions in the header. The theme and a new photo have
// their own buttons on the bar.
export function AccountMenu({ user, onOpenProducts, onSignOut, onReopen }: Props) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const onDocClick = (event: MouseEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDocClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDocClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const initial = (user.name || user.email).trim().charAt(0).toUpperCase();

  const item = (onClick: () => void, label: string, testId?: string) => (
    <button
      role="menuitem"
      className="menu-item"
      data-testid={testId}
      onClick={() => {
        setOpen(false);
        onClick();
      }}
    >
      {label}
    </button>
  );

  return (
    <div className="account-menu" ref={rootRef}>
      <button
        className="avatar-trigger"
        aria-haspopup="menu"
        aria-expanded={open}
        title={user.email}
        onClick={() => setOpen((value) => !value)}
      >
        <span className="avatar">{initial}</span>
      </button>
      {open && (
        <div className="menu-panel" role="menu">
          <div className="menu-header">
            <span className="avatar">{initial}</span>
            <div className="menu-header-text">
              <strong>{user.name || "Account"}</strong>
              <span className="muted small">{user.email}</span>
            </div>
          </div>
          <div className="menu-divider" />
          {item(onOpenProducts, "Products", "open-products")}
          {onReopen && item(onReopen, "Reopen for correction")}
          <div className="menu-divider" />
          {item(onSignOut, "Sign out")}
        </div>
      )}
    </div>
  );
}
