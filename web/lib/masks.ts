// One offscreen canvas per area, in working-image pixels: white where the area is, transparent elsewhere.
// Regions carry their mask as a PNG data URL; this store keeps the live canvases in step with them.

import type { Region } from "./design";

interface Entry {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  src: string | null;                    // the data URL the canvas currently shows
  data: Uint8ClampedArray | null;        // cached pixels for hit tests
  tinted: HTMLCanvasElement | null;
  tint: string;
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => { const img = new Image(); img.onload = () => resolve(img); img.onerror = reject; img.src = src; });
}

export class MaskStore {
  private entries = new Map<string, Entry>();
  constructor(public width: number, public height: number) {}

  private make(): Entry {
    const canvas = document.createElement("canvas");
    canvas.width = this.width; canvas.height = this.height;
    return { canvas, ctx: canvas.getContext("2d", { willReadFrequently: true })!, src: null, data: null, tinted: null, tint: "" };
  }

  entry(id: string): Entry {
    let e = this.entries.get(id);
    if (!e) { e = this.make(); this.entries.set(id, e); }
    return e;
  }

  /** Bring the canvases in line with the regions (after load, undo, redo, detection). */
  async sync(regions: Region[]): Promise<boolean> {
    let changed = false;
    const areaIds = new Set(regions.filter((r) => r.geometry === "area").map((r) => r.id));
    for (const id of [...this.entries.keys()]) if (!areaIds.has(id) && !id.startsWith("__")) { this.entries.delete(id); changed = true; }
    await Promise.all(regions.filter((r) => r.geometry === "area").map(async (r) => {
      const e = this.entry(r.id);
      const src = r.mask ?? null;
      if (e.src === src) return;
      e.ctx.clearRect(0, 0, this.width, this.height);
      if (src) e.ctx.drawImage(await loadImage(src), 0, 0, this.width, this.height);
      e.src = src; e.data = null; e.tinted = null; changed = true;
    }));
    return changed;
  }

  /** A mask that is not a region (the detected roof, "__support"). */
  async setExtra(id: string, src: string | null | undefined) {
    const e = this.entry(id);
    if (e.src === (src ?? null)) return;
    e.ctx.clearRect(0, 0, this.width, this.height);
    if (src) e.ctx.drawImage(await loadImage(src), 0, 0, this.width, this.height);
    e.src = src ?? null; e.data = null; e.tinted = null;
  }

  has(id: string): boolean { return !!this.entries.get(id)?.src; }

  /** A brush stroke from a to b (image px). Erase removes. */
  dab(id: string, a: [number, number], b: [number, number], radius: number, erase: boolean) {
    const e = this.entry(id);
    const c = e.ctx;
    c.save();
    c.globalCompositeOperation = erase ? "destination-out" : "source-over";
    c.strokeStyle = "#fff"; c.fillStyle = "#fff"; c.lineCap = "round"; c.lineWidth = radius * 2;
    c.beginPath(); c.moveTo(a[0], a[1]); c.lineTo(b[0], b[1]); c.stroke();
    c.beginPath(); c.arc(b[0], b[1], radius, 0, Math.PI * 2); c.fill();
    c.restore();
    e.data = null; e.tinted = null;
  }

  /** OR another mask (a SAM result) into an area. */
  async merge(id: string, src: string) {
    const e = this.entry(id);
    e.ctx.drawImage(await loadImage(src), 0, 0, this.width, this.height);
    e.data = null; e.tinted = null;
  }

  /** Save the canvas as the region's mask; returns null when the area is empty. */
  export(id: string): string | null {
    const e = this.entry(id);
    const url = this.isEmpty(id) ? null : e.canvas.toDataURL("image/png");
    e.src = url;
    return url;
  }

  private pixels(e: Entry): Uint8ClampedArray {
    if (!e.data) e.data = e.ctx.getImageData(0, 0, this.width, this.height).data;
    return e.data;
  }

  isEmpty(id: string): boolean {
    const d = this.pixels(this.entry(id));
    for (let i = 3; i < d.length; i += 4 * 7) if (d[i] > 0) return false;
    return true;
  }

  covers(id: string, x: number, y: number): boolean {
    const e = this.entries.get(id);
    if (!e) return false;
    const xi = Math.round(x), yi = Math.round(y);
    if (xi < 0 || yi < 0 || xi >= this.width || yi >= this.height) return false;
    return this.pixels(e)[(yi * this.width + xi) * 4 + 3] > 0;
  }

  /** The area in its colour, for drawing over the photo. */
  tinted(id: string, colour: string): HTMLCanvasElement | null {
    const e = this.entries.get(id);
    if (!e) return null;
    if (!e.tinted || e.tint !== colour) {
      const t = e.tinted ?? document.createElement("canvas");
      t.width = this.width; t.height = this.height;
      const c = t.getContext("2d")!;
      c.clearRect(0, 0, this.width, this.height);
      c.drawImage(e.canvas, 0, 0);
      c.globalCompositeOperation = "source-in";
      c.fillStyle = colour; c.fillRect(0, 0, this.width, this.height);
      c.globalCompositeOperation = "source-over";
      e.tinted = t; e.tint = colour;
    }
    return e.tinted;
  }
}
