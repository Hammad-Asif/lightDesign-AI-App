import { fileUrl } from "./api";

// Project thumbnails are kept in the browser: in the Cache Storage (they survive a reload and are
// never fetched again; a photo's image never changes after upload) and, for this page's life, as
// object URLs, so going back to the projects list shows them at once.
const CACHE = "ld-thumbnails-v1";
const urls = new Map<string, string>();
const loading = new Map<string, Promise<string>>();

async function fetchBlob(url: string): Promise<Blob> {
  let response: Response | undefined;
  try {
    if (typeof caches !== "undefined") {
      const cache = await caches.open(CACHE);
      response = await cache.match(url);
      if (!response) {
        const fresh = await fetch(url, { credentials: "include" });
        if (!fresh.ok) throw new Error(`Thumbnail ${fresh.status}`);
        await cache.put(url, fresh.clone()).catch(() => undefined);
        response = fresh;
      }
    }
  } catch (cause) {
    // A missing photo stays missing; any other failure (storage blocked) falls back to a plain fetch.
    if (cause instanceof Error && cause.message.startsWith("Thumbnail")) throw cause;
    response = undefined;
  }
  if (!response) {
    response = await fetch(url, { credentials: "include" });
    if (!response.ok) throw new Error(`Thumbnail ${response.status}`);
  }
  return response.blob();
}

// The thumbnail's object URL; rejects when the photo's image is gone.
export function thumbnail(path: string): Promise<string> {
  const ready = urls.get(path);
  if (ready) return Promise.resolve(ready);
  let pending = loading.get(path);
  if (!pending) {
    pending = fetchBlob(fileUrl(path))
      .then((blob) => {
        const url = URL.createObjectURL(blob);
        urls.set(path, url);
        return url;
      })
      .finally(() => loading.delete(path));
    loading.set(path, pending);
  }
  return pending;
}

// Already loaded on this page: shown without a placeholder frame.
export function thumbnailNow(path: string): string | null {
  return urls.get(path) ?? null;
}

// On sign-out: the next account on this browser starts with none of these.
export async function forgetThumbnails(): Promise<void> {
  urls.forEach((url) => URL.revokeObjectURL(url));
  urls.clear();
  try {
    if (typeof caches !== "undefined") await caches.delete(CACHE);
  } catch {
    // Storage blocked: nothing was kept.
  }
}
