import { useEffect, useRef, useState } from "react";

import type { User } from "../types";
import type { Theme } from "./ThemeToggle";

interface Props {
  user: User;
  theme: Theme;
  onThemeChange: (theme: Theme) => void;
  onOpenProducts: () => void;
  onSignOut: () => void;
  // Only shown inside the workspace, not on the entry screens.
  onNewPhoto?: () => void;
  onReopen?: () => void;
}

// Global, account-level controls (who's signed in, theme, sign out) live behind one menu so
// they never compete with the per-photo actions in the header or toolbar.
export function AccountMenu({ user, theme, onThemeChange, onOpenProducts, onSignOut, onNewPhoto, onReopen }: Props) {
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
          {onNewPhoto && item(onNewPhoto, "New photo")}
          {item(onOpenProducts, "Products", "open-products")}
          {onReopen && item(onReopen, "Reopen for correction")}
          <button role="menuitem" className="menu-item" onClick={() => onThemeChange(theme === "dark" ? "light" : "dark")}>
            {theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
          </button>
          <div className="menu-divider" />
          {item(onSignOut, "Sign out")}
        </div>
      )}
    </div>
  );
}
