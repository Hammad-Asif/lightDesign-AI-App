"use client";

import { useState } from "react";
import {
  ADD_GROUPS, EDGE_LABEL, EXTRA_KINDS, kindGeometry, kindLabel, placeOf, regionColour, regionsFor,
  type Catalogue, type Design, type PatternStep, type Region, type Target,
} from "@/lib/design";
import type { Role } from "@/lib/types";

// ───────────────────────────── Lines & areas ─────────────────────────────

export function RegionPanel(p: {
  design: Design; cat: Catalogue | null; selectedId: string | null; locked: boolean;
  hiddenShare: Record<string, number>; showObstacles: boolean; setShowObstacles(v: boolean): void;
  onSelect(id: string | null): void; onUpdate(id: string, patch: Partial<Region>): void; onDelete(id: string): void;
  onAdd(kind: string, edge?: Role): void; onDismissNotes(): void; drawing: boolean;
}) {
  const { design, cat } = p;
  const lines = design.regions.filter((r) => r.geometry === "line");
  const lit = lines.filter((r) => r.visible);
  const unlit = lines.filter((r) => !r.visible);
  const areas = design.regions.filter((r) => r.geometry === "area" && r.role !== "blocker");
  const blockers = design.regions.filter((r) => r.geometry === "area" && r.role === "blocker");
  const [addOpen, setAddOpen] = useState(false);

  const row = (r: Region) => (
    <RegionRow key={r.id} r={r} cat={cat} selected={r.id === p.selectedId} locked={p.locked} hidden={p.hiddenShare[r.id]}
      onSelect={() => p.onSelect(r.id === p.selectedId ? null : r.id)} onUpdate={(patch) => p.onUpdate(r.id, patch)} onDelete={() => p.onDelete(r.id)} />
  );

  return (
    <div className="space-y-5">
      {design.notes.length > 0 && (
        <div className="rounded-md border border-amber-200 bg-amber-50 text-amber-900 px-3 py-2 text-sm space-y-1">
          {design.notes.map((n, i) => <p key={i}>{n}</p>)}
          <button type="button" className="text-xs underline" onClick={p.onDismissNotes}>Dismiss</button>
        </div>
      )}
      <section className="space-y-2">
        <h3 className="flex justify-between font-semibold"><span>Lines to light</span><span className="text-muted font-normal">{lit.length}</span></h3>
        {lit.length ? <div className="space-y-1">{lit.map(row)}</div>
          : <p className="text-muted">None yet. Add a line below and click along the roofline to draw it.</p>}
        {lit.some((r) => r.kind === "roof") && (
          <p className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted">
            {(["eave", "rake", "hip", "ridge", "valley"] as Role[]).map((e) => (
              <span key={e} className="flex items-center gap-1"><span className="inline-block h-0.5 w-4" style={{ background: cat?.edge_colours[e] }} />{EDGE_LABEL[e].toLowerCase()}</span>
            ))}
          </p>
        )}
        {unlit.length > 0 && (
          <details className="rounded-md border border-rule">
            <summary className="cursor-pointer px-3 py-2 text-sm text-muted">Other roof edges, not lit ({unlit.length})</summary>
            <div className="px-1 pb-1 space-y-1">{unlit.map(row)}</div>
          </details>
        )}
      </section>

      <section className="space-y-2">
        <h3 className="flex justify-between font-semibold"><span>Areas</span><span className="text-muted font-normal">{areas.length}</span></h3>
        {areas.length ? <div className="space-y-1">{areas.map(row)}</div>
          : <p className="text-muted">None yet. Add one below, then paint it with the brush.</p>}
      </section>

      {blockers.length > 0 && (
        <section className="space-y-2">
          <h3 className="flex justify-between items-center font-semibold">
            <span>Obstacles</span>
            <label className="flex items-center gap-1.5 text-sm font-normal text-muted cursor-pointer">
              <input type="checkbox" className="accent-draft" checked={p.showObstacles} onChange={(e) => p.setShowObstacles(e.target.checked)} />Show on the photo
            </label>
          </h3>
          <p className="text-xs text-muted">Lines get no lights where they pass behind these.</p>
          <div className="space-y-1">{blockers.map(row)}</div>
        </section>
      )}

      {!p.locked && (
        <section className="border-t border-rule pt-3">
          <button type="button" className="flex items-center gap-2 font-semibold" onClick={() => setAddOpen((o) => !o)} aria-expanded={addOpen}>
            <span className={`text-xs transition-transform ${addOpen ? "rotate-90" : ""}`}>▸</span>Add a line or area
          </button>
          {addOpen && (
            <div className="mt-3 space-y-3">
              {p.drawing && <p className="text-sm text-draft">Click along the photo to place points. Double-click or press Enter to finish, Esc to cancel.</p>}
              {ADD_GROUPS.map((g) => (
                <div key={g.title} className="space-y-1.5">
                  <p className="text-xs uppercase tracking-wide text-muted">{g.title}</p>
                  <div className="flex flex-wrap gap-1.5">
                    {g.items.map((it) => (
                      <button key={`${it.kind}-${it.edge ?? ""}`} type="button" className="tool-btn !py-1 !px-2.5 text-sm flex items-center gap-1.5"
                        onClick={() => p.onAdd(it.kind, it.edge)}>
                        <span className={kindGeometry(cat, it.kind) === "line" ? "inline-block h-0.5 w-3" : "inline-block h-2.5 w-2.5 rounded-sm"} style={{ background: regionColour(cat, it.kind, it.edge) }} />
                        {it.label}
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>
      )}
    </div>
  );
}

function RegionRow({ r, cat, selected, locked, hidden, onSelect, onUpdate, onDelete }: {
  r: Region; cat: Catalogue | null; selected: boolean; locked: boolean; hidden?: number;
  onSelect(): void; onUpdate(patch: Partial<Region>): void; onDelete(): void;
}) {
  const kinds = { ...(cat?.kinds ?? {}), ...EXTRA_KINDS };
  const sameGeometry = Object.entries(kinds).filter(([, v]) => v.geometry === r.geometry);
  const source = r.source === "auto" ? `found${r.score != null ? ` · ${Math.round(r.score * 100)}%` : ""}` : r.source === "refined" ? "found by SAM from your box or click" : "drawn by you";
  return (
    <div className={`rounded-md ${selected ? "bg-paper ring-1 ring-draft" : "hover:bg-paper"}`}>
      <div className="flex items-center gap-2 px-2 py-1.5 cursor-pointer" onClick={onSelect}>
        <span className={r.geometry === "line" ? "inline-block h-1 w-4 rounded" : "inline-block h-3 w-3 rounded-sm"} style={{ background: r.color, opacity: r.visible ? 1 : 0.45 }} />
        <span className={`flex-1 truncate ${r.visible ? "" : "text-muted"}`}>{r.label}</span>
        {hidden != null && hidden > 0.005 && <span className="text-xs text-muted">{Math.round(hidden * 100)}% hidden</span>}
        {r.kind === "roof" && !r.edge && <span className="text-xs text-amber-500">check</span>}
        {r.geometry === "area" && !r.mask && <span className="text-xs text-amber-500">empty</span>}
        {!locked && (
          <>
            <button type="button" title={r.visible ? (r.geometry === "line" ? "Lit: click to leave dark" : "Shown: click to hide") : "Click to light"}
              className="px-1 text-muted hover:text-ink" onClick={(e) => { e.stopPropagation(); onUpdate({ visible: !r.visible }); }}>{r.visible ? "◉" : "○"}</button>
            <button type="button" title="Delete" className="px-1 text-muted hover:text-red-500" onClick={(e) => { e.stopPropagation(); onDelete(); }}>✕</button>
          </>
        )}
      </div>
      {selected && (
        <div className="px-3 pb-3 pt-1 space-y-2 text-sm">
          <label className="flex items-center gap-2"><span className="w-12 text-muted">Name</span>
            <input className="field flex-1 !py-1" value={r.label} disabled={locked} onChange={(e) => onUpdate({ label: e.target.value })} /></label>
          <label className="flex items-center gap-2"><span className="w-12 text-muted">Kind</span>
            <select className="field flex-1 !py-1" value={r.kind} disabled={locked}
              onChange={(e) => { const kind = e.target.value; onUpdate({ kind, color: regionColour(cat, kind, kind === "roof" ? (r.edge ?? "eave") : null), edge: kind === "roof" ? (r.edge ?? "eave") : null, role: kind === "obstruction" ? "blocker" : r.role === "blocker" && kind !== "obstruction" ? "target" : r.role }); }}>
              {sameGeometry.map(([k, v]) => <option key={k} value={k}>{k === "roof" ? "Roof edge" : v.label}</option>)}
            </select></label>
          {r.kind === "roof" && (
            <label className="flex items-center gap-2"><span className="w-12 text-muted">Edge</span>
              <select className="field flex-1 !py-1" value={r.edge ?? ""} disabled={locked} onChange={(e) => { const edge = (e.target.value || null) as Role | null; onUpdate({ edge, color: regionColour(cat, "roof", edge) }); }}>
                <option value="">unknown</option>
                {(["eave", "rake", "hip", "ridge", "valley"] as Role[]).map((e) => <option key={e} value={e}>{EDGE_LABEL[e]}</option>)}
              </select></label>
          )}
          <label className="flex items-center gap-2"><span className="w-12 text-muted">Side</span>
            <select className="field flex-1 !py-1" value={r.side ?? ""} disabled={locked} onChange={(e) => onUpdate({ side: e.target.value || null })}>
              <option value="">any</option><option value="front">front</option><option value="left">left</option><option value="right">right</option><option value="back">back</option>
            </select></label>
          {r.geometry === "area" && (
            <label className="flex items-center gap-2"><span className="w-12 text-muted">Role</span>
              <select className="field flex-1 !py-1" value={r.role} disabled={locked} onChange={(e) => onUpdate({ role: e.target.value as Region["role"] })}>
                <option value="target">light it</option><option value="blocker">obstacle in front of lines</option>
              </select></label>
          )}
          {(r.kind.startsWith("custom") || r.kind === "obstruction") && (
            <label className="flex items-center gap-2"><span className="w-12 text-muted">Colour</span>
              <input type="color" value={r.color} disabled={locked} onChange={(e) => onUpdate({ color: e.target.value })} className="h-7 w-12 bg-transparent" /></label>
          )}
          {hidden != null && hidden > 0.005 && <p className="text-xs text-amber-500">{Math.round(hidden * 100)}% passes behind objects: no lights there.</p>}
          <p className="text-xs text-muted">
            {source}{r.length_px ? ` · ${Math.round(r.length_px)} px` : ""}{r.pitch_apparent ? ` · pitch as seen ${r.pitch_apparent}` : ""}{r.inferred ? " · partly inferred behind an obstacle" : ""}
          </p>

        </div>
      )}
    </div>
  );
}

// ───────────────────────────── Lighting ─────────────────────────────

export function LightingPanel(p: {
  design: Design; cat: Catalogue | null;
  onTargets(targets: Target[]): void; onConfirm(): void;
}) {
  const { design, cat } = p;
  const update = (id: string, patch: Partial<Target>) => p.onTargets(design.targets.map((t) => (t.id === id ? { ...t, ...patch } : t)));
  const colours = Object.entries(cat?.colours ?? {});
  const products = cat?.products ?? [];

  if (!design.targets.length) {
    return <p className="text-muted">The instruction named no places, so there is nothing to light yet. Every line and area you add is kept; lighting is set per request from the instruction.</p>;
  }
  return (
    <div className="space-y-4">
      <p className="text-sm text-muted">One row per request in the instruction. Nothing is rendered until the lighting is confirmed.</p>
      {design.targets.map((t) => {
        const product = products.find((x) => x.id === t.product_id);
        const matched = regionsFor(design, t);
        return (
          <div key={t.id} className="rounded-md border border-rule p-3 space-y-2.5 text-sm">
            <div>
              <p className="font-medium">{placeOf(cat, t)}</p>
              <p className="text-muted">“{t.text}” · {matched.length} line{matched.length === 1 ? "" : "s"} or area{matched.length === 1 ? "" : "s"}</p>
            </div>
            <label className="flex items-center gap-2"><span className="w-16 text-muted">Product</span>
              <select className="field flex-1 !py-1" value={t.product_id}
                onChange={(e) => { const np = products.find((x) => x.id === e.target.value)!; update(t.id, { product_id: np.id, product_name: np.name, spacing_in: np.spacing_in, pattern: np.category === "accent" ? [] : t.pattern.length ? t.pattern : [{ colour: "warm_white", count: 1 }] }); }}>
                {(["socket", "strand", "accent"] as const).map((c) => (
                  <optgroup key={c} label={c === "socket" ? "Bulb lines" : c === "strand" ? "Strands" : "Accents"}>
                    {products.filter((x) => x.category === c).map((x) => <option key={x.id} value={x.id}>{x.name}</option>)}
                  </optgroup>
                ))}
              </select></label>
            {product && product.category !== "accent" && (
              <div className="space-y-1.5">
                <span className="text-muted">Colours{product.category === "socket" ? " (pattern, repeats along the line)" : ""}</span>
                {t.pattern.map((s, i) => (
                  <PatternRow key={i} step={s} colours={colours} canRemove={t.pattern.length > 1} showCount={product.category === "socket"}
                    onChange={(ns) => update(t.id, { pattern: t.pattern.map((x, j) => (j === i ? ns : x)) })}
                    onRemove={() => update(t.id, { pattern: t.pattern.filter((_, j) => j !== i) })} />
                ))}
                {product.category === "socket" && t.pattern.length < 8 && (
                  <button type="button" className="text-draft text-sm" onClick={() => update(t.id, { pattern: [...t.pattern, { colour: "red", count: 1 }] })}>+ Add a colour</button>
                )}
              </div>
            )}
            {product && product.spacing_options.length > 0 && (
              <label className="flex items-center gap-2"><span className="w-16 text-muted">Spacing</span>
                <select className="field flex-1 !py-1" value={t.spacing_in ?? ""} onChange={(e) => update(t.id, { spacing_in: e.target.value ? Number(e.target.value) : product.spacing_in, problems: [] })}>
                  {t.spacing_in != null && !product.spacing_options.includes(t.spacing_in) && <option value={t.spacing_in}>{t.spacing_in} in (not sold)</option>}
                  {product.spacing_options.map((s) => <option key={s} value={s}>{s} in{s === product.spacing_in ? " (default)" : ""}</option>)}
                </select></label>
            )}
            {product && product.category !== "accent" && (
              <label className="flex items-center gap-2 cursor-pointer"><input type="checkbox" className="accent-draft" checked={t.twinkle} onChange={(e) => update(t.id, { twinkle: e.target.checked })} />Twinkle</label>
            )}
            {t.assumed.length > 0 && <p className="text-xs text-muted">Assumed: {t.assumed.join(", ")}.</p>}
            {t.problems.map((x, i) => <p key={i} className="text-xs text-amber-500">{x}</p>)}
            {matched.length === 0 && <p className="text-xs text-amber-500">No line or area matches this yet{t.manual.length ? `: draw the ${t.manual.map((k) => kindLabel(cat, k).toLowerCase()).join(", ")} by hand` : ""}.</p>}
          </div>
        );
      })}
      <button type="button" className="w-full rounded-md bg-draft text-white py-2 font-medium hover:opacity-90" onClick={p.onConfirm}>
        {design.lighting_confirmed ? "Lighting confirmed · confirm again" : "Confirm lighting"}
      </button>
    </div>
  );
}

function PatternRow({ step, colours, canRemove, showCount, onChange, onRemove }: {
  step: PatternStep; colours: [string, { label: string; hex: string }][]; canRemove: boolean; showCount: boolean;
  onChange(s: PatternStep): void; onRemove(): void;
}) {
  const hex = colours.find(([k]) => k === step.colour)?.[1].hex ?? "#ccc";
  return (
    <div className="flex items-center gap-2">
      <span className="inline-block h-4 w-4 rounded-sm shrink-0" style={{ background: hex }} />
      <select className="field flex-1 !py-1" value={step.colour} onChange={(e) => onChange({ ...step, colour: e.target.value })}>
        {colours.map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}
      </select>
      {showCount && (
        <label className="flex items-center gap-1 text-muted">×<input type="number" min={1} max={20} className="field w-14 !py-1" value={step.count}
          onChange={(e) => onChange({ ...step, count: Math.max(1, Math.min(20, Number(e.target.value) || 1)) })} /></label>
      )}
      {canRemove && <button type="button" className="text-muted hover:text-red-500 px-1" onClick={onRemove} title="Remove">✕</button>}
    </div>
  );
}
