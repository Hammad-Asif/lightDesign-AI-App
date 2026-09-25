"use client";

import { useEffect, useRef } from "react";
import { ROLES, type LineDoc, type Role } from "@/lib/types";

const fmt = (v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(1));

export function SegmentTable({ doc, roleColors, editing, hover, selected, onSelect, onRole }: {
  doc: LineDoc; roleColors: Record<string, string>; editing: boolean; hover: number; selected: number;
  onSelect(i: number): void; onRole(i: number, r: Role): void;
}) {
  const body = useRef<HTMLTableSectionElement>(null);
  useEffect(() => {
    if (selected >= 0) body.current?.rows[selected]?.scrollIntoView({ block: "nearest" });
  }, [selected]);
  return (
    <div className="rounded-md border border-rule bg-panel overflow-hidden">
      <div className="max-h-72 overflow-auto">
        <table className="w-full text-sm">
          <thead className="text-left text-muted sticky top-0 bg-panel z-[1]">
            <tr className="border-b border-rule">
              {["#", "Role", "Planes", "From (x, y)", "To (x, y)"].map((h) => <th key={h} className="font-medium px-3 py-2">{h}</th>)}
              {["Slope", "Pitch as seen", "Angle", "Length"].map((h) => <th key={h} className="font-medium px-3 py-2 text-right">{h}</th>)}
            </tr>
          </thead>
          <tbody ref={body}>
            {doc.segments.map((s, i) => (
              <tr key={i} onClick={() => onSelect(i)}
                className={`border-b border-rule/60 last:border-0 ${i === hover || i === selected ? "hl" : ""} ${editing ? "cursor-pointer" : ""}`}>
                <td className="px-3 py-1.5 text-muted">{s.id + 1}</td>
                <td className="px-3 py-1.5 whitespace-nowrap">
                  <span className="chip" style={{ background: s.inferred ? roleColors.inferred : roleColors[s.role] }} />
                  {editing ? (
                    <select className="bg-transparent border-0 p-0 text-sm focus:outline-draft" value={s.role} onClick={(e) => e.stopPropagation()}
                      onChange={(e) => onRole(i, e.target.value as Role)} aria-label={`Role of line ${s.id + 1}`}>
                      {ROLES.map((r) => <option key={r} value={r}>{r}</option>)}
                    </select>
                  ) : s.role}
                  {s.crease ? " (hidden crease)" : ""}
                  {s.inferred && <span className="text-muted"> inferred</span>}
                  {s.edited && <span className="ml-1 text-xs rounded bg-amber-100 text-amber-900 px-1">edited</span>}
                </td>
                <td className="px-3 py-1.5">{(s.planes || []).join(" + ")}</td>
                <td className="px-3 py-1.5">{fmt(s.points[0][0])}, {fmt(s.points[0][1])}</td>
                <td className="px-3 py-1.5">{fmt(s.points[1][0])}, {fmt(s.points[1][1])}</td>
                <td className="px-3 py-1.5 text-right">{s.slope === null ? "∞" : s.slope.toFixed(3)}</td>
                <td className="px-3 py-1.5 text-right">{s.pitch_apparent || "steep"}</td>
                <td className="px-3 py-1.5 text-right">{s.angle_deg.toFixed(1)}°</td>
                <td className="px-3 py-1.5 text-right">{s.length_px.toFixed(0)} px</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function PlaneTable({ doc, focalKnown, edited }: { doc: LineDoc; focalKnown: boolean; edited: boolean }) {
  const rows = doc.planes.filter((p) => p.pitch_apparent || p.pitch_calibrated);
  if (!rows.length) return null;
  return (
    <div className="rounded-md border border-rule bg-panel overflow-hidden">
      <table className="w-full text-sm">
        <thead className="text-left text-muted">
          <tr className="border-b border-rule">
            <th className="font-medium px-3 py-2">Plane</th><th className="font-medium px-3 py-2">Corners</th>
            <th className="font-medium px-3 py-2 text-right">Pitch as seen</th><th className="font-medium px-3 py-2 text-right">True pitch</th>
            <th className="font-medium px-3 py-2 text-right">Nearest standard</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((p, i) => {
            const c = p.pitch_calibrated, a = p.pitch_apparent_standard;
            return (
              <tr key={i} className="border-b border-rule/60 last:border-0">
                <td className="px-3 py-1.5">{p.plane}</td>
                <td className="px-3 py-1.5">{p.polygon.length}</td>
                <td className="px-3 py-1.5 text-right">{p.pitch_apparent ? `${p.pitch_apparent} (${p.pitch_apparent_deg}°)` : ""}</td>
                <td className="px-3 py-1.5 text-right">{c ? `${c.pitch} (${c.pitch_deg}°)` : <span className="text-muted">{focalKnown ? "not constrained in this view" : "needs a focal length"}</span>}</td>
                <td className="px-3 py-1.5 text-right">
                  {c ? <>{c.standard.pitch} <span className="text-muted">{Math.round(c.standard.probability * 100)}%</span></>
                    : a?.pitch ? <>{a.pitch} <span className="text-muted">from the pitch as seen</span></> : ""}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {edited && <p className="px-3 py-1.5 text-xs text-muted border-t border-rule">Plane pitches are re-measured from your edited lines.</p>}
    </div>
  );
}
