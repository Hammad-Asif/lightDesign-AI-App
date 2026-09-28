import type {
  Design,
  DesignRequest,
  Edge,
  MaskingMode,
  PaletteColour,
  Photo,
  PhotoDetail,
  Product,
  ProductDraft,
  Region,
  RegionKind,
  Side,
  User,
} from "./types";

export const API_URL = import.meta.env.VITE_API_URL ?? "";

// Thrown on 401 so the app can drop back to the sign-in screen from anywhere.
export class UnauthorizedError extends Error {}

function detailOf(body: { detail?: unknown }, fallback: string): string {
  if (typeof body.detail === "string") return body.detail;
  if (Array.isArray(body.detail) && body.detail[0]?.msg) return String(body.detail[0].msg);
  // A list of plain sentences, as the design confirmation returns when settings do not fit.
  if (Array.isArray(body.detail) && body.detail.every((item: unknown) => typeof item === "string"))
    return (body.detail as string[]).join(" ");
  return fallback;
}

async function json<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const message = detailOf(body, `${response.status} ${response.statusText}`);
    if (response.status === 401) throw new UnauthorizedError(message);
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

// The session is an HttpOnly cookie; "include" sends it even when the API is on another origin.
function request(path: string, init: RequestInit = {}): Promise<Response> {
  return fetch(`${API_URL}${path}`, { credentials: "include", ...init });
}

function send<T>(path: string, method: string, body?: unknown): Promise<T> {
  return request(path, {
    method,
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  }).then(json<T>);
}

export function fileUrl(path: string): string {
  return `${API_URL}${path}`;
}

export const me = () => send<User | null>("/auth/me", "GET");

export const register = (payload: { email: string; name: string; password: string }) =>
  send<User>("/auth/register", "POST", payload);

export const login = (payload: { email: string; password: string }) =>
  send<User>("/auth/login", "POST", payload);

export async function logout(): Promise<void> {
  await request("/auth/logout", { method: "POST" });
}

export async function uploadPhoto(file: File, prompt: string): Promise<Photo> {
  const form = new FormData();
  form.append("file", file);
  form.append("prompt", prompt);
  return json<Photo>(await request("/photos", { method: "POST", body: form }));
}

export const getPhoto = (id: string) => send<PhotoDetail>(`/photos/${id}`, "GET");

export const setMode = (id: string, mode: MaskingMode) =>
  send<Photo>(`/photos/${id}/mode`, "PUT", { mode });

export const detectRegions = (id: string, opts?: { kinds?: RegionKind[]; allKinds?: boolean }) =>
  send<Region[]>(`/photos/${id}/detect`, "POST", {
    kinds: opts?.kinds ?? null,
    all_kinds: opts?.allKinds ?? false,
  });

export const createRegion = (
  id: string,
  payload: { kind: RegionKind; geometry: "line" | "area"; label?: string; side?: Side; edge?: Edge; polyline?: number[][] },
) => send<Region>(`/photos/${id}/regions`, "POST", payload);

export const patchRegion = (
  id: string,
  regionId: string,
  patch: Partial<Pick<Region, "label" | "kind" | "side" | "edge" | "visible" | "polyline">>,
) => send<Region>(`/photos/${id}/regions/${regionId}`, "PATCH", patch);

export const refineRegion = (
  id: string,
  payload: { box?: number[]; point?: number[]; region_id?: string; kind?: RegionKind },
) => send<Region>(`/photos/${id}/regions/refine`, "POST", payload);

// Find in Box: detect roof lines inside a box the user drew. Adds new lines only; nothing existing
// changes. Returns the lines it added, which may be none.
export const findLinesInBox = (id: string, box: number[]) =>
  send<Region[]>(`/photos/${id}/find-lines`, "POST", { box });

// Moves the roof plane under a tap into or out of the house being quoted; the roof lines are redrawn
// from the stored planes, with no detection call.
export const tapHousePlane = (id: string, point: number[]) =>
  send<PhotoDetail>(`/photos/${id}/target`, "POST", { point });

// Re-reads the instruction (a new one when given) without detecting anything: reading is free.
export const interpret = (id: string, prompt?: string) =>
  send<Photo>(`/photos/${id}/interpret`, "POST", { prompt: prompt ?? null });

export const listProducts = (archived = false) =>
  send<Product[]>(`/products${archived ? "?archived=true" : ""}`, "GET");
export const palette = () => send<PaletteColour[]>("/products/palette", "GET");

export const createProduct = (draft: ProductDraft) => send<Product>("/products", "POST", draft);

export const updateProduct = (id: string, draft: ProductDraft) =>
  send<Product>(`/products/${id}`, "PATCH", draft);

export const restoreProduct = (id: string) =>
  send<Product>(`/products/${id}/restore`, "POST");

export async function putProductImage(id: string, file: File): Promise<Product> {
  const form = new FormData();
  form.append("file", file);
  return json<Product>(await request(`/products/${id}/image`, { method: "PUT", body: form }));
}

export const deleteProductImage = (id: string) =>
  send<Product>(`/products/${id}/image`, "DELETE");

export async function deleteProduct(id: string): Promise<void> {
  const response = await request(`/products/${id}`, { method: "DELETE" });
  if (response.status === 401) throw new UnauthorizedError("Sign in required");
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(detailOf(body, `Could not remove the product (${response.status})`));
  }
}

// null until the lighting has been confirmed for the photo.
export const getDesign = (id: string) => send<Design | null>(`/photos/${id}/design`, "GET");

export const putDesign = (id: string, requests: DesignRequest[]) =>
  send<Design>(`/photos/${id}/design`, "PUT", { requests });

export const approvePhoto = (id: string) => send<PhotoDetail>(`/photos/${id}/approve`, "POST");
export const reopenPhoto = (id: string) => send<PhotoDetail>(`/photos/${id}/reopen`, "POST");

export async function deleteRegion(id: string, regionId: string): Promise<void> {
  const response = await request(`/photos/${id}/regions/${regionId}`, { method: "DELETE" });
  if (response.status === 401) throw new UnauthorizedError("Sign in required");
  if (!response.ok) throw new Error(`Could not delete region (${response.status})`);
}

export async function putMask(id: string, regionId: string, blob: Blob): Promise<Region> {
  const form = new FormData();
  form.append("file", blob, `${regionId}.png`);
  return json<Region>(
    await request(`/photos/${id}/regions/${regionId}/mask`, { method: "PUT", body: form }),
  );
}
