import { useEffect, useMemo, useState } from "react";

import { CATEGORY_LABELS, KIND_LABELS, requestLights } from "../types";
import type {
  Design,
  DesignRequest,
  Edge,
  InterpretedRequest,
  LightingSettings,
  PaletteColour,
  Product,
  Region,
} from "../types";

interface Props {
  requests: InterpretedRequest[];
  products: Product[];
  palette: PaletteColour[];
  // Present in the workspace, where each request can be counted against real regions and
  // confirmed; absent on the mode screen, where the reading is only shown.
  regions?: Region[];
  design?: Design | null;
  busy?: boolean;
  onConfirm?: (requests: DesignRequest[]) => void;
}

const EDGE_WORDS: Record<Edge, string> = {
  eave: "eaves", rake: "rakes", hip: "hips", ridge: "ridge", valley: "valleys", junction: "junctions", unknown: "unknown",
};
const WRAP_KINDS = ["palm_trunk", "palm_collar"];

// Where a request puts its lights, in the words the region list uses.
export function placeOf(request: Pick<InterpretedRequest, "kinds" | "sides" | "edges" | "manual">): string {
  const kinds = request.kinds.map((kind) => KIND_LABELS[kind]).join(", ");
  const parts = [kinds];
  if (request.kinds.includes("roof") && request.edges.length) parts.push(request.edges.map((e) => EDGE_WORDS[e]).join(", "));
  if (request.sides.length) parts.push(`${request.sides.join("/")} only`);
  return parts.join(" · ");
}

function initial(request: InterpretedRequest, design: Design | null | undefined): LightingSettings | null {
  const confirmed = design?.requests.find((r) => r.text === request.text && r.kinds.join() === request.kinds.join());
  return confirmed?.settings ?? request.settings;
}

export function Specification({ requests, products, palette, regions, design, busy, onConfirm }: Props) {
  const editable = Boolean(onConfirm);
  const byId = useMemo(() => new Map(products.map((p) => [p.id, p])), [products]);
  const colourName = (key: string) => palette.find((c) => c.key === key)?.name ?? key;

  const [drafts, setDrafts] = useState<(LightingSettings | null)[]>(() => requests.map((r) => initial(r, design)));
  const [dirty, setDirty] = useState(false);
  // Start over from the reading whenever it, or what was confirmed, actually changes — not on
  // every render, since the parent may hand down a fresh array of the same requests.
  const reading = JSON.stringify(requests.map((r) => [r.text, r.kinds, r.settings]));
  useEffect(() => {
    setDrafts(requests.map((r) => initial(r, design)));
    setDirty(false);
  }, [reading, design?.id, design?.regions.length]);

  const update = (index: number, settings: LightingSettings | null) => {
    setDrafts((current) => current.map((item, i) => (i === index ? settings : item)));
    setDirty(true);
  };

  const pickProduct = (index: number, productId: string) => {
    const product = byId.get(productId);
    if (!product) return update(index, null);
    const previous = drafts[index];
    // A socket line needs a pattern; a strand brings its own colours. Spacing starts at the
    // product's own default, since the old one may not be sold for the new product.
    const pattern = product.colours.length
      ? null
      : previous?.pattern ?? { steps: [{ colour: "warm_white", count: 1 }], unit: "bulb" as const };
    update(index, { product_id: productId, pattern, spacing_in: null, twinkle: null });
  };

  const confirm = () => {
    if (!onConfirm) return;
    const ready: DesignRequest[] = [];
    requests.forEach((request, index) => {
      const settings = drafts[index];
      if (settings) ready.push({ text: request.text, kinds: request.kinds, sides: request.sides, edges: request.edges, settings });
    });
    onConfirm(ready);
  };

  if (!requests.length) return null;
  const missing = drafts.some((d) => !d);
  const litCount = design?.regions.length ?? 0;

  return (
    <div className="spec">
      <ol className="spec-list">
        {requests.map((request, index) => {
          const settings = drafts[index];
          const product = settings ? byId.get(settings.product_id) : undefined;
          const count = regions ? regions.filter((region) => requestLights(request, region)).length : null;
          // Roof lines whose edge nobody has identified are never lit on a guess; say how many wait.
          const unidentified = regions && request.kinds.includes("roof")
            ? regions.filter((r) => r.kind === "roof" && r.role === "target" && r.visible && r.edge === "unknown").length
            : 0;
          const socket = product ? product.colours.length === 0 : false;
          const twinkle = settings?.twinkle ?? product?.twinkle ?? false;
          return (
            <li key={index} className="spec-row">
              <div className="spec-place">
                <strong>{placeOf(request)}</strong>
                {request.manual.length > 0 && <span className="muted small"> · drawn by hand</span>}
                {count !== null && (
                  <span className="muted small"> · {count === 0 ? "no regions yet" : `${count} ${count === 1 ? "region" : "regions"}`}</span>
                )}
                {unidentified > 0 && (
                  <span className="muted small"> · {unidentified} roof {unidentified === 1 ? "line" : "lines"} with the edge not set — set it to light {unidentified === 1 ? "it" : "them"}</span>
                )}
                <div className="muted small spec-quote">“{request.text}”</div>
              </div>

              <div className="spec-lighting">
                {editable ? (
                  <select title="Product" value={settings?.product_id ?? ""} onChange={(event) => pickProduct(index, event.target.value)}>
                    <option value="">Choose a product</option>
                    {(Object.keys(CATEGORY_LABELS) as Product["category"][]).map((category) => {
                      const items = products.filter((p) => p.category === category);
                      return items.length ? (
                        <optgroup key={category} label={CATEGORY_LABELS[category]}>
                          {items.map((p) => (
                            <option key={p.id} value={p.id}>{p.name}</option>
                          ))}
                        </optgroup>
                      ) : null;
                    })}
                  </select>
                ) : (
                  <span>{product?.name ?? request.product_name ?? "No product chosen"}</span>
                )}

                {product && socket && settings && (
                  <PatternEditor
                    pattern={settings.pattern ?? { steps: [{ colour: "warm_white", count: 1 }] }}
                    palette={palette}
                    editable={editable}
                    wraps={request.kinds.some((kind) => WRAP_KINDS.includes(kind))}
                    onChange={(pattern) => update(index, { ...settings, pattern })}
                  />
                )}
                {product && !socket && (
                  <span className="muted small">{product.colours.map(colourName).join(", ")}</span>
                )}

                {product && socket && settings && (
                  editable && product.spacing_options_in.length ? (
                    <select
                      title="Spacing"
                      value={settings.spacing_in ?? ""}
                      onChange={(event) => update(index, { ...settings, spacing_in: event.target.value ? Number(event.target.value) : null })}
                    >
                      <option value="">{product.spacing_in} in (default)</option>
                      {product.spacing_options_in.map((option) => (
                        <option key={option} value={option}>{option} in spacing</option>
                      ))}
                    </select>
                  ) : (
                    <span className="muted small">{settings.spacing_in ?? product.spacing_in} in spacing</span>
                  )
                )}
                {product && !socket && product.spacing_in !== null && (
                  <span className="muted small">{product.spacing_in} in spacing</span>
                )}

                {product && settings && (
                  editable ? (
                    <label className="spec-check">
                      <input
                        type="checkbox"
                        checked={twinkle}
                        onChange={(event) => update(index, { ...settings, twinkle: event.target.checked === product.twinkle ? null : event.target.checked })}
                      />
                      Twinkle
                    </label>
                  ) : (
                    twinkle && <span className="muted small">twinkle</span>
                  )
                )}
              </div>

              {request.assumed.length > 0 && <div className="muted small spec-note">Assumed: {request.assumed.join("; ")}.</div>}
              {request.problems.length > 0 && <div className="spec-problem">{request.problems.join(". ")}.</div>}
            </li>
          );
        })}
      </ol>

      {editable && (
        <div className="spec-actions">
          <span className="muted small">
            {design && !dirty
              ? `Lighting confirmed for ${litCount} ${litCount === 1 ? "region" : "regions"}.`
              : design
                ? "Changed since it was confirmed."
                : "Nothing renders until the lighting is confirmed."}
          </span>
          <button className="primary" disabled={busy || missing || (Boolean(design) && !dirty)} onClick={confirm}>
            {design ? "Confirm changes" : "Confirm lighting"}
          </button>
          {missing && <span className="muted small">Every request needs a product first.</span>}
        </div>
      )}
    </div>
  );
}

interface PatternProps {
  pattern: NonNullable<LightingSettings["pattern"]>;
  palette: PaletteColour[];
  editable: boolean;
  // Trunks and collars can change colour each turn of the wrap instead of each bulb.
  wraps: boolean;
  onChange: (pattern: NonNullable<LightingSettings["pattern"]>) => void;
}

function PatternEditor({ pattern, palette, editable, wraps, onChange }: PatternProps) {
  const steps = pattern.steps;
  const swatch = (key: string) => palette.find((c) => c.key === key)?.hex ?? "#888";
  if (!editable) {
    return (
      <span className="spec-pattern">
        {steps.map((step, i) => (
          <span key={i} className="spec-step">
            <span className="swatch" style={{ background: swatch(step.colour) }} />
            {palette.find((c) => c.key === step.colour)?.name ?? step.colour}
            {step.count > 1 && ` ×${step.count}`}
          </span>
        ))}
        {pattern.unit === "turn" && <span className="muted small">per turn</span>}
      </span>
    );
  }
  const setStep = (i: number, step: Partial<{ colour: string; count: number }>) =>
    onChange({ ...pattern, steps: steps.map((item, j) => (j === i ? { ...item, ...step } : item)) });
  return (
    <span className="spec-pattern">
      {steps.map((step, i) => (
        <span key={i} className="spec-step">
          <span className="swatch" style={{ background: swatch(step.colour) }} />
          <select title="Colour" value={step.colour} onChange={(event) => setStep(i, { colour: event.target.value })}>
            {palette.map((c) => (
              <option key={c.key} value={c.key}>{c.name}</option>
            ))}
          </select>
          <input
            type="number"
            min={1}
            max={50}
            value={step.count}
            title="How many in a row"
            onChange={(event) => setStep(i, { count: Math.max(1, Number(event.target.value) || 1) })}
          />
          {steps.length > 1 && (
            <button className="icon" title="Remove colour" onClick={() => onChange({ ...pattern, steps: steps.filter((_, j) => j !== i) })}>
              ×
            </button>
          )}
        </span>
      ))}
      {steps.length < 8 && (
        <button className="icon" title="Add a colour" onClick={() => onChange({ ...pattern, steps: [...steps, { colour: "red", count: 1 }] })}>
          + colour
        </button>
      )}
      {wraps && (
        <select value={pattern.unit ?? "bulb"} onChange={(event) => onChange({ ...pattern, unit: event.target.value as "bulb" | "turn" })}>
          <option value="bulb">changes each bulb</option>
          <option value="turn">changes each turn</option>
        </select>
      )}
    </span>
  );
}
