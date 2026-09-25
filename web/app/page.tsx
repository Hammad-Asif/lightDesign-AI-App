"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import AppHeader, { useUser } from "@/components/AppHeader";
import { api } from "@/lib/api";
import type { ProjectSummary } from "@/lib/types";

export default function ProjectsPage() {
  const user = useUser();
  const [projects, setProjects] = useState<ProjectSummary[] | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!user) return;
    api<{ projects: ProjectSummary[] }>("/api/projects").then((r) => setProjects(r.projects)).catch((e) => setError(e.message));
  }, [user]);

  async function rename(p: ProjectSummary) {
    const name = window.prompt("Project name", p.name)?.trim();
    if (!name || name === p.name) return;
    try {
      await api(`/api/projects/${p.id}`, { method: "PUT", json: { name } });
      setProjects((list) => list?.map((q) => (q.id === p.id ? { ...q, name } : q)) ?? null);
    } catch (e) { setError((e as Error).message); }
  }
  async function remove(p: ProjectSummary) {
    if (!window.confirm(`Delete "${p.name}"? Its photo and lines are removed for good.`)) return;
    try {
      await api(`/api/projects/${p.id}`, { method: "DELETE" });
      setProjects((list) => list?.filter((q) => q.id !== p.id) ?? null);
    } catch (e) { setError((e as Error).message); }
  }

  return (
    <>
      <AppHeader user={user} />
      <main className="max-w-6xl mx-auto px-5 py-8 space-y-6">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="text-xl font-semibold tracking-tight">Projects</h1>
            <p className="text-sm text-muted">Each project keeps its photo, the detected lines and your edits.</p>
          </div>
          <div className="flex gap-2">
            {/* Detection studio hidden from the UI (owner's request, 2026-09-26); the code is kept in components/Studio.tsx.
            <Link href="/studio" className="tool-btn !py-2" title="The roof detector's own view: planes, inspection layers, geometry sliders">Detection studio</Link> */}
            <Link href="/new" className="rounded-md bg-draft text-white px-4 py-2 text-sm font-medium hover:opacity-90">New mock-up</Link>
          </div>
        </div>

        {error && <p role="alert" className="text-sm rounded-md border border-red-300 bg-red-50 text-red-800 px-3 py-2">{error}</p>}

        {projects === null ? (
          <p className="text-sm text-muted">Loading…</p>
        ) : projects.length === 0 ? (
          <div className="rounded-lg border border-dashed border-rule bg-panel px-6 py-14 text-center space-y-2">
            <p className="font-medium">No projects yet</p>
            <p className="text-sm text-muted">Start one from a daytime photo of the house and say where the lights go.</p>
            <Link href="/new" className="inline-block mt-2 rounded-md bg-draft text-white px-4 py-2 text-sm font-medium hover:opacity-90">New mock-up</Link>
          </div>
        ) : (
          <ul className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {projects.map((p) => (
              <li key={p.id} className="rounded-lg border border-rule bg-panel overflow-hidden flex flex-col">
                <Link href={`/design?project=${p.id}`} className="block bg-lightbox aspect-[3/2]">
                  {/* eslint-disable-next-line @next/next/no-img-element -- served by the backend behind the session cookie */}
                  <img src={p.image_url} alt="" className="h-full w-full object-contain" loading="lazy" />
                </Link>
                <div className="p-3 flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <Link href={`/design?project=${p.id}`} className="font-medium hover:underline block truncate">{p.name}</Link>
                    <p className="text-xs text-muted">
                      {p.design
                        ? `${p.lines} line${p.lines === 1 ? "" : "s"} · ${p.areas} area${p.areas === 1 ? "" : "s"}${p.approved ? " · locked" : ""}`
                        : `${p.segments} line${p.segments === 1 ? "" : "s"}`} · {new Date(p.updated_at * 1000).toLocaleString()}
                    </p>
                  </div>
                  <div className="flex gap-1 shrink-0 text-xs">
                    <button type="button" onClick={() => rename(p)} className="rounded px-2 py-1 text-draft hover:bg-paper">Rename</button>
                    <button type="button" onClick={() => remove(p)} className="rounded px-2 py-1 text-red-700 hover:bg-red-50">Delete</button>
                  </div>
                </div>
              </li>
            ))}
          </ul>
        )}
      </main>
    </>
  );
}
