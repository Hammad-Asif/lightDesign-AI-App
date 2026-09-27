export function createLayer(width: number, height: number): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

export function ctx2d(canvas: HTMLCanvasElement): CanvasRenderingContext2D {
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Canvas 2D context unavailable");
  return context;
}

export function hexToRgb(hex: string): [number, number, number] {
  const value = parseInt(hex.replace("#", ""), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

export function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Could not load ${src}`));
    image.src = src;
  });
}

export async function paintMaskInto(
  layer: HTMLCanvasElement,
  url: string,
  color: string,
): Promise<void> {
  const image = await loadImage(url);
  const scratch = createLayer(layer.width, layer.height);
  ctx2d(scratch).drawImage(image, 0, 0, layer.width, layer.height);

  const source = ctx2d(scratch).getImageData(0, 0, layer.width, layer.height);
  const output = new ImageData(layer.width, layer.height);
  const [r, g, b] = hexToRgb(color);
  for (let i = 0; i < source.data.length; i += 4) {
    if (source.data[i] > 127) {
      output.data[i] = r;
      output.data[i + 1] = g;
      output.data[i + 2] = b;
      output.data[i + 3] = 255;
    }
  }
  const target = ctx2d(layer);
  target.clearRect(0, 0, layer.width, layer.height);
  target.putImageData(output, 0, 0);
}

export function layerToBlob(layer: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    layer.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("Mask export failed"))),
      "image/png",
    );
  });
}

export function distanceToSegment(
  p: number[],
  a: number[],
  b: number[],
): { distance: number; point: number[] } {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const lengthSq = dx * dx + dy * dy;
  const t = lengthSq === 0 ? 0 : Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / lengthSq));
  const point = [a[0] + t * dx, a[1] + t * dy];
  return { distance: Math.hypot(p[0] - point[0], p[1] - point[1]), point };
}

export function polylineLength(points: number[][]): number {
  let total = 0;
  for (let i = 1; i < points.length; i++) {
    total += Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]);
  }
  return total;
}

// The part of a polyline between two distances from its start, following its bends.
export function slicePolyline(points: number[][], start: number, end: number): number[][] {
  const part: number[][] = [];
  let walked = 0;
  for (let i = 0; i < points.length - 1; i++) {
    const [x0, y0] = points[i];
    const [x1, y1] = points[i + 1];
    const length = Math.hypot(x1 - x0, y1 - y0);
    if (length === 0) continue;
    const from = Math.max(start, walked);
    const to = Math.min(end, walked + length);
    if (to > from) {
      const at = (distance: number) => [
        x0 + ((x1 - x0) * (distance - walked)) / length,
        y0 + ((y1 - y0) * (distance - walked)) / length,
      ];
      if (!part.length) part.push(at(from));
      part.push(at(to));
    }
    walked += length;
    if (walked >= end) break;
  }
  return part;
}
