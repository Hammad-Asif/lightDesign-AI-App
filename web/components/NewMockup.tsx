"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import AppHeader, { useUser } from "@/components/AppHeader";
import { api, post } from "@/lib/api";
import { placeOf, type Catalogue, type Design, type Target } from "@/lib/design";
import type { Project, SessionInfo } from "@/lib/types";

const PLACEHOLDER = "C9 warm white on the front roofline at 12 inch spacing, mini lights in the hedges";

export default function NewMockup() {
  const user = useUser();
  const router = useRouter();
  const [cat, setCat] = useState<Catalogue | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [text, setText] = useState("");
  const [readText, setReadText] = useState("");
  const [targets, setTargets] = useState<Target[] | null>(null);
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");

  useEffect(() => { if (user) api<Catalogue>("/api/design/catalogue").then(setCat).catch(() => {}); }, [user]);
  useEffect(() => {
    if (!file) { setPreview(null); return; }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  async function read(t: string) {
    const r = await post<{ targets: Target[] }>("/api/instruction/parse", { text: t });
    setTargets(r.targets); setReadText(t);
  }

  async function onContinue() {
    if (!file) return;
    setBusy("Uploading the photo…"); setError("");
    try {
      const form = new FormData(); form.append("file", file);
      const info = await api<SessionInfo>("/api/upload", { method: "POST", body: form });
      setSession(info);
      await read(text.trim());
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(""); }
  }

  async function choose(mode: Design["mode"]) {
    if (!session || !targets) return;
    setBusy(mode === "automatic" ? "Creating the mock-up…" : "Opening the editor…"); setError("");
    try {
      const design: Design = { version: 1, instruction: readText, mode, targets, regions: [], detected: false, notes: [], lighting_confirmed: false, approved: false };
      const name = (session.filename || "House").replace(/\.[^.]+$/, "");
      const created = await post<Project>("/api/projects", {
        session_id: session.session_id, name,
        data: { design, detected: null, edited: null, segmentation: null, geometry_params: { epsilon: 0.01, mergeAngle: 5, snap: 0.008, wdir: 3, stage: 7, focal: null } },
      });
      router.push(`/design?project=${created.id}`);
    } catch (e) { setError((e as Error).message); setBusy(""); }
  }

  const colourHex = (c: string) => cat?.colours[c]?.hex ?? "#ccc";
  const colourLabel = (c: string) => cat?.colours[c]?.label ?? c;

  return (
    <>
      <AppHeader user={user} />
      <main className="min-h-[calc(100vh-3.5rem)] grid place-items-center px-4 py-10">
        {!targets ? (
          <section className="w-full max-w-lg rounded-lg border border-rule bg-panel p-8 space-y-5">
            <div className="space-y-2">
              <h1 className="text-2xl font-semibold tracking-tight">New mock-up</h1>
              <p className="text-muted">Start with a daytime photo of the house, taken before any lights go up.</p>
            </div>
            <label className="block cursor-pointer rounded-md border border-dashed border-draft/70 bg-paper px-4 py-6 text-center hover:border-draft">
              {preview ? (
                <span className="flex items-center gap-3 text-left">
                  {/* eslint-disable-next-line @next/next/no-img-element -- local preview */}
                  <img src={preview} alt="" className="h-16 w-24 object-cover rounded" />
                  <span className="min-w-0"><span className="block truncate font-medium">{file?.name}</span><span className="text-sm text-muted">Click to choose another</span></span>
                </span>
              ) : <span>Choose a photo (JPG, PNG or WEBP)</span>}
              <input type="file" accept="image/jpeg,image/png,image/webp" className="sr-only" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
            </label>
            <label className="block space-y-2">
              <span className="text-muted">Where do the lights go?</span>
              <textarea className="field w-full min-h-28 resize-y" placeholder={PLACEHOLDER} value={text} onChange={(e) => setText(e.target.value)} />
              <span className="block text-sm text-muted">Only the areas you mention get masked. You can change any of them afterwards.</span>
            </label>
            {error && <p role="alert" className="text-sm rounded-md border border-red-300 bg-red-50 text-red-800 px-3 py-2">{error}</p>}
            <button type="button" disabled={!file || !!busy} onClick={onContinue}
              className="w-full rounded-md bg-draft text-white py-2.5 font-medium hover:opacity-90 disabled:opacity-50">{busy || "Continue"}</button>
          </section>
        ) : (
          <section className="w-full max-w-2xl rounded-lg border border-rule bg-panel p-8 space-y-5">
            <div className="space-y-2">
              <h1 className="text-2xl font-semibold tracking-tight">Check the instruction</h1>
              <p className="text-muted">This is how it was read. Correct the wording here, or the details later in the editor.</p>
            </div>
            <label className="block space-y-2">
              <span className="text-muted">Where do the lights go?</span>
              <textarea className="field w-full min-h-24 resize-y" value={text} onChange={(e) => setText(e.target.value)} placeholder={PLACEHOLDER} />
            </label>
            {text.trim() !== readText && (
              <button type="button" className="tool-btn" onClick={() => read(text.trim()).catch((e) => setError(e.message))}>Read it again</button>
            )}
            {targets.length === 0 ? (
              <p className="rounded-md border border-rule bg-paper px-4 py-3 text-muted">No places were named, so nothing will be detected. Choose Manual and draw the lines and areas yourself, or name a place above (for example “roofline” or “hedges”).</p>
            ) : targets.map((t) => (
              <div key={t.id} className="rounded-md border border-rule bg-paper px-4 py-3 space-y-1.5">
                <p className="font-medium">{placeOf(cat, t)}</p>
                <p className="text-muted">“{t.text}”</p>
                <p className="flex flex-wrap items-center gap-x-3 gap-y-1">
                  <span>{t.product_name}</span>
                  {t.pattern.length > 0 && (
                    <span className="flex items-center gap-1.5">
                      {t.pattern.map((s, i) => (
                        <span key={i} className="flex items-center gap-1">
                          <span className="inline-block h-3 w-3 rounded-sm" style={{ background: colourHex(s.colour) }} />
                          {s.count > 1 ? `${s.count} × ` : ""}{colourLabel(s.colour)}{i < t.pattern.length - 1 ? "," : ""}
                        </span>
                      ))}
                    </span>
                  )}
                  {t.spacing_in != null && <span className="text-muted">{t.spacing_in} in spacing</span>}
                  {t.twinkle && <span className="text-muted">twinkle</span>}
                </p>
                {t.assumed.length > 0 && <p className="text-muted">Assumed: {t.assumed.join(", ")}.</p>}
                {t.manual.length > 0 && <p className="text-muted">Drawn by hand: {t.manual.map((k) => cat?.kinds[k]?.label ?? k).join(", ")}.</p>}
                {t.problems.map((pr, i) => <p key={i} className="text-amber-500">{pr}</p>)}
              </div>
            ))}
            <h2 className="font-semibold pt-2">How should the areas be marked?</h2>
            {error && <p role="alert" className="text-sm rounded-md border border-red-300 bg-red-50 text-red-800 px-3 py-2">{error}</p>}
            <button type="button" disabled={!!busy || text.trim() !== readText || targets.length === 0} onClick={() => choose("automatic")}
              className="block w-full text-left rounded-md border border-rule bg-paper px-5 py-4 hover:border-draft disabled:opacity-50 disabled:hover:border-rule">
              <span className="block font-semibold">Automatic</span>
              <span className="block text-muted">Finds the roofline and the areas named above.{cat && !cat.sam_api ? " (SAM 3 is off: only the roof is found, with GrabCut.)" : ""}</span>
            </button>
            <button type="button" disabled={!!busy || text.trim() !== readText} onClick={() => choose("manual")}
              className="block w-full text-left rounded-md border border-rule bg-paper px-5 py-4 hover:border-draft disabled:opacity-50 disabled:hover:border-rule">
              <span className="block font-semibold">Manual</span>
              <span className="block text-muted">Draw every line and paint every area yourself, with nothing detected first.</span>
            </button>
            {busy && <p className="text-muted">{busy}</p>}
          </section>
        )}
      </main>
    </>
  );
}
