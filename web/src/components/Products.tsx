import { useEffect, useMemo, useState } from "react";

import * as api from "../api";
import type { PaletteColour, Product, ProductCategory, ProductDraft } from "../types";

// The three shelves the library is read from, in the order a salesperson thinks about them.
const GROUPS: { title: string; note: string; categories: ProductCategory[] }[] = [
  { title: "Bulb lines", note: "Socket lines: the bulbs are chosen per design.", categories: ["c9", "c7"] },
  { title: "Strands", note: "Made in a fixed colour sequence.", categories: ["mini", "icicle"] },
  { title: "Accents", note: "Placed one at a time.", categories: ["wreath", "garland"] },
];

const CATEGORY_NAMES: Record<ProductCategory, string> = {
  c9: "C9", c7: "C7", mini: "Mini lights", icicle: "Icicle lights",
  wreath: "Wreath", garland: "Garland",
};

// A colour per category, for the icon avatar a product with no picture of its own gets.
const CATEGORY_ACCENT: Record<ProductCategory, string> = {
  c9: "#f5a524", c7: "#f0563d", mini: "#38bdf8", icicle: "#60a5fa", wreath: "#22c55e", garland: "#14b8a6",
};

function BulbIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <path d="M9 18h6M10 22h4M12 2a6 6 0 0 0-3.6 10.8c.6.46 1 1.2 1 2.03V15h5.2v-.17c0-.83.4-1.57 1-2.03A6 6 0 0 0 12 2Z" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function PencilIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <path d="M12 20h9M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2m3 0-1 14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2L4 6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function CopyIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <rect x="9" y="9" width="12" height="12" rx="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" strokeLinecap="round" />
    </svg>
  );
}

function PlusIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <path d="M12 5v14M5 12h14" strokeLinecap="round" />
    </svg>
  );
}

// C9 and C7 are socket lines: bulbs of any colour go in, so the product carries none.
const SOCKET: ProductCategory[] = ["c9", "c7"];

// What each category has to describe, mirroring REQUIRED in api/app/services/products.py.
// A field the category does not use is never shown, so a form only ever asks what it needs.
const FIELDS: Record<ProductCategory, string[]> = {
  c9: ["bulb_diameter_in", "spacing_options_in", "spacing_in", "twinkle"],
  c7: ["bulb_diameter_in", "spacing_options_in", "spacing_in", "twinkle"],
  mini: ["bulb_diameter_in", "spacing_in", "strand_length_ft", "colours", "twinkle"],
  icicle: ["bulb_diameter_in", "spacing_in", "strand_length_ft", "drops_in", "colours", "twinkle"],
  wreath: ["diameter_in", "colours"],
  garland: ["diameter_in", "strand_length_ft", "colours"],
};

const SPACINGS = [6, 8, 12, 15, 18, 24, 36];

function blank(category: ProductCategory): ProductDraft {
  return {
    category, name: "", bulb_diameter_in: null, spacing_in: null, spacing_options_in: [],
    strand_length_ft: null, diameter_in: null, bulb_count: null, drops_in: [], colours: [],
    twinkle: false,
  };
}

function draftOf(product: Product): ProductDraft {
  const { id: _id, builtin: _builtin, archived: _archived, image_url: _image, ...draft } = product;
  return draft;
}

// /files serves a product picture only to whoever may see the product, so the <img> has to
// carry the session. A plain cross-origin <img> sends no cookies; same-origin is unaffected.
function credentials(src: string): "use-credentials" | undefined {
  return src.startsWith("http") ? "use-credentials" : undefined;
}

function numbers(text: string): number[] {
  return text.split(",").map((part) => Number(part.trim())).filter((n) => Number.isFinite(n) && n > 0);
}

/** What a product is, in one line, for the list. */
function describe(product: Product, palette: PaletteColour[]): string {
  const parts: string[] = [];
  if (product.bulb_diameter_in) parts.push(`${product.bulb_diameter_in} in bulb`);
  if (product.diameter_in) parts.push(`${product.diameter_in} in ${product.category === "garland" ? "thick" : "across"}`);
  if (product.spacing_options_in.length > 0) parts.push(`${product.spacing_options_in.join(", ")} in spacing`);
  else if (product.spacing_in) parts.push(`${product.spacing_in} in spacing`);
  if (product.strand_length_ft) parts.push(`${product.strand_length_ft} ft`);
  if (product.drops_in.length > 0) parts.push(`drops ${product.drops_in.join(", ")} in`);
  if (SOCKET.includes(product.category)) parts.push("any bulb colour");
  if (product.twinkle) parts.push("twinkle");
  if (product.colours.length > 0 && palette.length === 0) parts.push(product.colours.join(", "));
  return parts.join(" · ");
}

interface FormProps {
  draft: ProductDraft;
  palette: PaletteColour[];
  busy: boolean;
  problems: string[];
  // The picture as it stands: what is stored, or what is waiting to be uploaded on save.
  picture: string | null;
  // "Add a C9" or "Edit Lukas C9": which of the two is not otherwise obvious from the form.
  title: string;
  onPicture: (file: File | null) => void;
  onChange: (draft: ProductDraft) => void;
  onSave: () => void;
  onCancel: () => void;
}

function ProductForm({ draft, palette, busy, problems, picture, title, onPicture, onChange, onSave, onCancel }: FormProps) {
  const shows = (field: string) => FIELDS[draft.category].includes(field);
  const set = (fields: Partial<ProductDraft>) => onChange({ ...draft, ...fields });
  // Changing the category changes which fields exist, so the rest of the draft is cleared
  // rather than carried over as values the new category has no place for.
  const changeCategory = (category: ProductCategory) => onChange({ ...blank(category), name: draft.name });

  return (
    <div className="product-form" data-testid="product-form"
         onKeyDown={(event) => {
           if (event.key === "Escape") onCancel();
           // Enter saves from any single-line field, but not from the colour and spacing chips,
           // where it would mean "toggle this".
           if (event.key === "Enter" && (event.target as HTMLElement).tagName === "INPUT"
               && (event.target as HTMLInputElement).type !== "file" && draft.name.trim() !== "") {
             event.preventDefault();
             onSave();
           }
         }}>
      <h3 className="product-form-title">{title}</h3>
      <div className="product-picture-field">
        {picture ? (
          <img className="product-thumb large" src={picture} alt="" crossOrigin={credentials(picture)} />
        ) : (
          <span className="product-thumb large empty" aria-hidden="true" />
        )}
        <div>
          <span className="field-label">Picture</span>
          <div className="spec-actions">
            <label className="button-like">
              {picture ? "Replace" : "Add a picture"}
              <input type="file" accept="image/jpeg,image/png,image/webp" data-testid="product-picture"
                     onChange={(e) => onPicture(e.target.files?.[0] ?? null)} />
            </label>
            {picture && (
              <button type="button" className="ghost" disabled={busy}
                      data-testid="product-picture-remove" onClick={() => onPicture(null)}>
                Remove
              </button>
            )}
          </div>
          <span className="muted small">JPG, PNG or WEBP. Shown wherever the product is chosen.</span>
        </div>
      </div>

      <div className="product-form-row">
        <label className="field">
          <span>Name</span>
          <input value={draft.name} maxLength={120} data-testid="product-name"
                 onChange={(e) => set({ name: e.target.value })} />
        </label>
        <label className="field">
          <span>Kind</span>
          <select value={draft.category} data-testid="product-category"
                  onChange={(e) => changeCategory(e.target.value as ProductCategory)}>
            {Object.entries(CATEGORY_NAMES).map(([key, name]) => (
              <option key={key} value={key}>{name}</option>
            ))}
          </select>
        </label>
      </div>

      <div className="product-form-row">
        {shows("bulb_diameter_in") && (
          <label className="field">
            <span>Bulb size (in)</span>
            <input type="number" step="0.001" min="0" data-testid="product-bulb" value={draft.bulb_diameter_in ?? ""}
                   onChange={(e) => set({ bulb_diameter_in: e.target.value === "" ? null : Number(e.target.value) })} />
          </label>
        )}
        {shows("diameter_in") && (
          <label className="field">
            <span>{draft.category === "garland" ? "Thickness (in)" : "Diameter (in)"}</span>
            <input type="number" step="0.5" min="0" data-testid="product-diameter" value={draft.diameter_in ?? ""}
                   onChange={(e) => set({ diameter_in: e.target.value === "" ? null : Number(e.target.value) })} />
          </label>
        )}
        {shows("strand_length_ft") && (
          <label className="field">
            <span>Length (ft)</span>
            <input type="number" step="0.5" min="0" data-testid="product-length" value={draft.strand_length_ft ?? ""}
                   onChange={(e) => set({ strand_length_ft: e.target.value === "" ? null : Number(e.target.value) })} />
          </label>
        )}
      </div>

      {shows("spacing_options_in") && (
        <div className="field">
          <span>Spacings it is sold at</span>
          <div className="chips">
            {SPACINGS.map((inches) => {
              const on = draft.spacing_options_in.includes(inches);
              return (
                <button type="button" key={inches} className={on ? "chip on" : "chip"}
                        aria-pressed={on} data-testid={`spacing-${inches}`}
                        onClick={() => {
                          const options = on
                            ? draft.spacing_options_in.filter((s) => s !== inches)
                            : [...draft.spacing_options_in, inches].sort((a, b) => a - b);
                          // The default has to be one of the offered spacings, or the API refuses it.
                          const spacing = options.length > 0 && !options.includes(draft.spacing_in ?? 0)
                            ? options[0] : draft.spacing_in;
                          set({ spacing_options_in: options, spacing_in: spacing });
                        }}>
                  {inches}&quot;
                </button>
              );
            })}
          </div>
        </div>
      )}

      {shows("spacing_in") && (
        <label className="field">
          <span>{draft.category === "icicle" ? "Drop spacing (in)" : "Default spacing (in)"}</span>
          {draft.spacing_options_in.length > 0 ? (
            <select value={draft.spacing_in ?? ""}
                    onChange={(e) => set({ spacing_in: e.target.value === "" ? null : Number(e.target.value) })}>
              {draft.spacing_options_in.map((inches) => (
                <option key={inches} value={inches}>{inches} in</option>
              ))}
            </select>
          ) : (
            <input type="number" step="1" min="0" data-testid="product-spacing" value={draft.spacing_in ?? ""}
                   onChange={(e) => set({ spacing_in: e.target.value === "" ? null : Number(e.target.value) })} />
          )}
        </label>
      )}

      {shows("drops_in") && (
        <label className="field">
          <span>Drop lengths (in, comma separated)</span>
          <input value={draft.drops_in.join(", ")} data-testid="product-drops"
                 onChange={(e) => set({ drops_in: numbers(e.target.value) })} />
        </label>
      )}

      {shows("colours") ? (
        <div className="field">
          <span>Colours it is made in</span>
          <div className="chips">
            {palette.map((colour) => {
              const on = draft.colours.includes(colour.key);
              return (
                <button type="button" key={colour.key} className={on ? "chip on" : "chip"}
                        aria-pressed={on} title={colour.name} data-testid={`colour-${colour.key}`}
                        onClick={() => set({
                          colours: on ? draft.colours.filter((c) => c !== colour.key)
                                      : [...draft.colours, colour.key],
                        })}>
                  <span className="swatch" style={{ background: colour.hex }} />
                  {colour.name}
                </button>
              );
            })}
          </div>
        </div>
      ) : (
        <p className="muted small">
          {CATEGORY_NAMES[draft.category]} lines take any bulb colour: the colours are chosen per design.
        </p>
      )}

      {shows("twinkle") && (
        <label className="check">
          <input type="checkbox" checked={draft.twinkle} onChange={(e) => set({ twinkle: e.target.checked })} />
          <span>Twinkle</span>
        </label>
      )}

      {problems.length > 0 && (
        <div className="form-error" role="alert" data-testid="product-problems">
          {problems.join(". ")}.
        </div>
      )}

      <div className="spec-actions">
        <button disabled={busy || draft.name.trim() === ""} onClick={onSave} data-testid="product-save">Save</button>
        <button className="ghost" disabled={busy} onClick={onCancel}>Cancel</button>
      </div>
    </div>
  );
}

interface Props {
  palette: PaletteColour[];
  onClose: () => void;
  // The library changed: the confirmation screen and the instruction reader both read from it.
  onChanged: () => void;
}

export function Products({ palette, onClose, onChanged }: Props) {
  const [products, setProducts] = useState<Product[]>([]);
  const [retired, setRetired] = useState<Product[]>([]);
  const [editing, setEditing] = useState<string | "new" | null>(null);
  const [draft, setDraft] = useState<ProductDraft>(blank("c9"));
  const [problems, setProblems] = useState<string[]>([]);
  // A picture chosen but not yet uploaded, and what to show for it meanwhile. A new product
  // has no id to upload against, so the file waits until the product has been created.
  const [pending, setPending] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [stored, setStored] = useState<string | null>(null);
  const [dropPicture, setDropPicture] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [showRetired, setShowRetired] = useState(false);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState("");
  // Removing an unused product deletes it for good, so it is asked about on the row itself.
  const [confirming, setConfirming] = useState<string | null>(null);
  // Which group's pill is picked; null means every group, the default view.
  const [activeGroup, setActiveGroup] = useState<string | null>(null);
  const [showBuiltin, setShowBuiltin] = useState(true);
  const [customOnly, setCustomOnly] = useState(false);

  const load = () =>
    Promise.all([api.listProducts(), api.listProducts(true)])
      .then(([live, gone]) => { setProducts(live); setRetired(gone); })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));

  useEffect(() => { void load(); }, []);

  const matching = useMemo(() => {
    const wanted = filter.trim().toLowerCase();
    return products.filter((p) => {
      // "Only mine" wins outright; otherwise a built-in is shown unless unticked, and a
      // product of the user's own is always shown -- there is no "hide mine" toggle.
      if (customOnly ? p.builtin : (p.builtin && !showBuiltin)) return false;
      if (!wanted) return true;
      return p.name.toLowerCase().includes(wanted) || CATEGORY_NAMES[p.category].toLowerCase().includes(wanted);
    });
  }, [products, filter, showBuiltin, customOnly]);

  const byCategory = useMemo(() => {
    const map = new Map<ProductCategory, Product[]>();
    for (const product of matching) {
      map.set(product.category, [...(map.get(product.category) ?? []), product]);
    }
    return map;
  }, [matching]);

  // The object URL for a chosen file is revoked when it is replaced or the form closes.
  const choose = (file: File | null) => {
    setPreview((old) => {
      if (old) URL.revokeObjectURL(old);
      return file ? URL.createObjectURL(file) : null;
    });
    setPending(file);
    // Clearing the picker on a product that already has one means: take the picture off.
    setDropPicture(file === null);
  };

  const closeForm = () => {
    choose(null);
    setDropPicture(false);
    setStored(null);
    setEditing(null);
  };

  const start = (product: Product | null, category: ProductCategory) => {
    setProblems([]);
    setError(null);
    choose(null);
    setDropPicture(false);
    // A duplicate starts without the original's picture: the file belongs to that product.
    setStored(product && !product.builtin && product.image_url
      ? api.fileUrl(product.image_url) : null);
    if (product) {
      // A built-in is duplicated rather than edited: the copy is the user's own and editable.
      const copy = product.builtin;
      setDraft(copy ? { ...draftOf(product), name: `${product.name} (copy)` } : draftOf(product));
      setEditing(copy ? "new" : product.id);
    } else {
      setDraft(blank(category));
      setEditing("new");
    }
  };

  const save = async () => {
    setBusy(true);
    setProblems([]);
    setError(null);
    try {
      const saved = editing === "new"
        ? await api.createProduct(draft)
        : await api.updateProduct(editing as string, draft);
      // The picture is a separate upload, and a new product only has an id once it exists.
      if (pending) await api.putProductImage(saved.id, pending);
      else if (dropPicture && saved.image_url) await api.deleteProductImage(saved.id);
      closeForm();
      await load();
      onChanged();
    } catch (e) {
      // The API answers 422 with the sentences services/products.py writes. Pydantic prefixes a
      // model validator's message with "Value error, "; that frame is not for the salesperson.
      setProblems(
        String((e as Error).message)
          .replace(/^Value error,\s*/, "")
          .split("; ")
          .map((part) => part.trim())
          .filter(Boolean),
      );
    } finally {
      setBusy(false);
    }
  };

  const remove = async (product: Product) => {
    setBusy(true);
    setError(null);
    try {
      await api.deleteProduct(product.id);
      setConfirming(null);
      await load();
      onChanged();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const restore = async (product: Product) => {
    setBusy(true);
    try {
      await api.restoreProduct(product.id);
      await load();
      onChanged();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const builtinCount = products.filter((p) => p.builtin).length;

  return (
    <div className="product-dashboard" data-testid="products-screen">
      <div className="product-dashboard-head">
        <div>
          <span className="breadcrumb muted small">Light Design / Products</span>
          <h1>Products <span className="badge accent2">{matching.length} products shown</span></h1>
          <p className="lede">
            What you install. Instructions resolve against this list, so a product added here can
            be asked for by name straight away.
          </p>
        </div>
        <div className="spec-actions">
          <button type="button" className="product-cta"
                  disabled={busy || editing !== null} onClick={() => start(null, "c9")}>
            <PlusIcon /> Add Custom Product
          </button>
          <button className="ghost" onClick={onClose}>Done</button>
        </div>
      </div>

      {error && <div className="error" role="alert">{error}</div>}

      <div className="product-stats">
        <div className="stat-tile">
          <span className="stat-number">{products.length}</span>
          <span className="stat-label">Total items</span>
        </div>
        <div className="stat-tile">
          <span className="stat-number">{builtinCount}</span>
          <span className="stat-label">Built-in</span>
        </div>
        <div className="stat-tile">
          <span className="stat-number">{products.length - builtinCount}</span>
          <span className="stat-label">Custom variants</span>
        </div>
      </div>

      <div className="product-toolbar">
        <div className="product-pills">
          <button type="button" className={activeGroup === null ? "pill active" : "pill"}
                  onClick={() => setActiveGroup(null)}>
            All products
          </button>
          {GROUPS.map((group) => (
            <button type="button" key={group.title}
                    className={activeGroup === group.title ? "pill active" : "pill"}
                    onClick={() => setActiveGroup(group.title)}>
              {group.title}
            </button>
          ))}
        </div>
        <div className="search-field">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
            <circle cx="11" cy="11" r="7" />
            <path d="M21 21l-4.3-4.3" strokeLinecap="round" />
          </svg>
          <input className="product-search" type="search" value={filter} data-testid="product-search"
                 placeholder={`Search ${products.length} products`} aria-label="Search products"
                 onChange={(event) => setFilter(event.target.value)} />
        </div>
      </div>

      <div className="product-filters">
        <label className="check">
          <input type="checkbox" checked={showBuiltin} onChange={(e) => setShowBuiltin(e.target.checked)} />
          <span>Show built-in standard</span>
        </label>
        <label className="check">
          <input type="checkbox" checked={customOnly} onChange={(e) => setCustomOnly(e.target.checked)} />
          <span>User customised only</span>
        </label>
        <span className="product-summary" data-testid="product-count">
          Showing {matching.length} of {products.length} products
        </span>
      </div>

      {editing !== null && (
        <div className="drawer-overlay" onClick={closeForm}>
          <div className="drawer-panel" onClick={(event) => event.stopPropagation()}>
            <ProductForm draft={draft} palette={palette} busy={busy} problems={problems}
                         title={editing === "new" ? `New ${CATEGORY_NAMES[draft.category]}`
                                                  : `Edit ${draft.name || "product"}`}
                         picture={preview ?? (dropPicture ? null : stored)} onPicture={choose}
                         onChange={setDraft} onSave={() => void save()} onCancel={closeForm} />
          </div>
        </div>
      )}

      {loading && <p className="muted small">Loading the library…</p>}
      {!loading && filter && matching.length === 0 && (
        <p className="muted small" data-testid="no-matches">
          Nothing matches “{filter}”. <button className="link" onClick={() => setFilter("")}>Clear the search</button>
        </p>
      )}

      {!loading && GROUPS.filter((group) =>
        (activeGroup === null || activeGroup === group.title)
        && (!filter || group.categories.some((c) => (byCategory.get(c) ?? []).length > 0)),
      ).map((group) => (
        <section className="product-group" key={group.title}>
          <div className="product-group-head">
            <h2>{group.title}</h2>
            <span className="muted small">{group.note}</span>
          </div>
          {group.categories.filter((category) =>
            !filter || (byCategory.get(category) ?? []).length > 0,
          ).map((category) => (
            <div key={category}>
              <div className="product-category">
                <span className="muted small">
                  {CATEGORY_NAMES[category]}
                  {(byCategory.get(category) ?? []).length > 0 &&
                    ` · ${(byCategory.get(category) ?? []).length}`}
                </span>
                <button className="ghost small add-pill" disabled={busy || editing !== null}
                        data-testid={`add-${category}`} onClick={() => start(null, category)}>
                  <PlusIcon /> Add
                </button>
              </div>
              <ul className="product-list">
                {(byCategory.get(category) ?? []).map((product) => (
                  <li className="product" key={product.id} data-testid={`product-${product.id}`}>
                    <div className="product-card-top">
                      {product.image_url ? (
                        <img className="product-thumb" src={api.fileUrl(product.image_url)} alt=""
                             loading="lazy" crossOrigin={credentials(api.fileUrl(product.image_url))} />
                      ) : (
                        <span className="product-thumb empty" aria-hidden="true"
                              style={{ background: CATEGORY_ACCENT[product.category] }}>
                          <BulbIcon />
                          {product.colours.length > 0 && (
                            <span className="thumb-dot"
                                  style={{ background: palette.find((c) => c.key === product.colours[0])?.hex ?? "#888" }} />
                          )}
                        </span>
                      )}
                      <div className="product-main">
                        <div className="product-name-row">
                          <span className="product-name">{product.name}</span>
                          {product.builtin && <span className="badge">Built in</span>}
                        </div>
                        <span className="muted small">{describe(product, palette)}</span>
                      </div>
                    </div>
                    {product.colours.length > 0 && (
                      <div className="product-colours">
                        {product.colours.map((key, index) => {
                          const colour = palette.find((c) => c.key === key);
                          return (
                            <span className="swatch" key={`${key}-${index}`}
                                  title={colour?.name ?? key}
                                  style={{ background: colour?.hex ?? "#888" }} />
                          );
                        })}
                      </div>
                    )}
                    <div className="product-actions">
                      {product.builtin ? (
                        <button className="ghost small duplicate-btn" disabled={busy || editing !== null}
                                data-testid={`duplicate-${product.id}`} onClick={() => start(product, category)}>
                          <CopyIcon /> Duplicate to edit
                        </button>
                      ) : (
                        <>
                          <button className="ghost icon-btn" disabled={busy || editing !== null}
                                  title="Edit" aria-label="Edit"
                                  data-testid={`edit-${product.id}`} onClick={() => start(product, category)}>
                            <PencilIcon />
                          </button>
                          {confirming === product.id ? (
                            <>
                              <span className="muted small">Remove it?</span>
                              <button className="ghost small" disabled={busy}
                                      data-testid={`confirm-${product.id}`} onClick={() => void remove(product)}>
                                Yes, remove
                              </button>
                              <button className="ghost small" disabled={busy}
                                      onClick={() => setConfirming(null)}>
                                Keep
                              </button>
                            </>
                          ) : (
                            <button className="ghost icon-btn danger" disabled={busy || editing !== null}
                                    title="Remove" aria-label="Remove"
                                    data-testid={`delete-${product.id}`} onClick={() => setConfirming(product.id)}>
                              <TrashIcon />
                            </button>
                          )}
                        </>
                      )}
                    </div>
                  </li>
                ))}
                {(byCategory.get(category) ?? []).length === 0 && (
                  <li className="muted small product-empty">
                    None yet — Add one, or duplicate a built-in to start from.
                  </li>
                )}
              </ul>
            </div>
          ))}
        </section>
      ))}

      {retired.length > 0 && (
        <section className="product-group">
          <button className="ghost small" onClick={() => setShowRetired(!showRetired)}
                  data-testid="toggle-retired">
            {showRetired ? "Hide" : "Show"} retired ({retired.length})
          </button>
          {showRetired && (
            <>
              <p className="muted small">
                Kept because a saved design still names them. They are out of the library and out
                of the instruction reader.
              </p>
              <ul className="product-list">
                {retired.map((product) => (
                  <li className="product" key={product.id}>
                    <div className="product-main">
                      <span className="product-name">{product.name}</span>
                      <span className="muted small">{CATEGORY_NAMES[product.category]}</span>
                    </div>
                    <div className="product-actions">
                      <button className="ghost small" disabled={busy}
                              data-testid={`restore-${product.id}`} onClick={() => void restore(product)}>
                        Bring back
                      </button>
                    </div>
                  </li>
                ))}
              </ul>
            </>
          )}
        </section>
      )}
    </div>
  );
}
