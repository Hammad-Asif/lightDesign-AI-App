"use client";

import Link from "next/link";
import { useEffect, useState, type ReactNode } from "react";
import { api, post } from "@/lib/api";
import type { User } from "@/lib/types";

export function useUser() {
  const [user, setUser] = useState<User | null>(null);
  useEffect(() => {
    api<{ user: User | null }>("/api/auth/me").then((r) => {
      if (r.user) setUser(r.user);
      else window.location.href = `/login?next=${encodeURIComponent(location.pathname + location.search)}`;
    }).catch(() => {});
  }, []);
  return user;
}

export function ThemeToggle() {
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  useEffect(() => { setTheme((document.documentElement.dataset.theme as "dark" | "light") || "dark"); }, []);
  function toggle() {
    const next = theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("theme", next); } catch { /* private mode */ }
    setTheme(next);
  }
  return (
    <button type="button" onClick={toggle} className="tool-btn flex items-center gap-1.5" title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}>
      <span aria-hidden>{theme === "dark" ? "☀" : "☾"}</span>{theme === "dark" ? "Light" : "Dark"}
    </button>
  );
}

export default function AppHeader({ user, children, status, actions }: { user: User | null; children?: ReactNode; status?: ReactNode; actions?: ReactNode }) {
  const [open, setOpen] = useState(false);
  async function signOut() {
    await post("/api/auth/logout").catch(() => {});
    window.location.href = "/login";
  }
  return (
    <header className="h-14 px-5 flex items-center justify-between gap-4 border-b border-rule bg-panel">
      <div className="flex items-center gap-4 min-w-0">
        <Link href="/" className="text-lg font-semibold tracking-tight shrink-0">Roofline</Link>
        {children}
      </div>
      <div className="flex items-center gap-3 shrink-0">
        {status}
        <ThemeToggle />
        {actions}
        {user && (
          <div className="relative">
            <button type="button" onClick={() => setOpen((o) => !o)} aria-expanded={open}
              className="flex items-center gap-2 rounded-md px-2 py-1 text-sm hover:bg-paper focus-visible:outline-2 focus-visible:outline-draft">
              <span className="grid place-items-center h-7 w-7 rounded-full bg-ink text-panel text-xs font-semibold">{(user.name || user.email).slice(0, 1).toUpperCase()}</span>
              <span className="hidden sm:inline max-w-40 truncate">{user.name}</span>
            </button>
            {open && (
              <div className="absolute right-0 mt-1 w-56 rounded-md border border-rule bg-panel shadow-lg z-20 py-1 text-sm" onMouseLeave={() => setOpen(false)}>
                <p className="px-3 py-2 text-muted truncate border-b border-rule">{user.email}</p>
                <Link href="/" className="block px-3 py-2 hover:bg-paper">Projects</Link>
                <button type="button" onClick={signOut} className="block w-full text-left px-3 py-2 hover:bg-paper">Sign out</button>
              </div>
            )}
          </div>
        )}
      </div>
    </header>
  );
}
