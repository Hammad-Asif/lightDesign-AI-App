"""Which roof planes belong to the house being quoted (Settings.target_house_only).

A photo of one house usually shows parts of the houses beside it, and SAM 3's "roof" prompt
returns their planes too. This module decides from the plane masks alone, with no extra model
call, which planes are the target house's roof:

  1. The seed is the largest roof near the middle of the photo. A salesperson frames the house
     being quoted, so it is central and whole, while a neighbour is usually cut off by an edge.
  2. The house grows from the seed across seams: planes that share a real stretch of outline,
     as the two sides of a hip do. A plane the left or right edge of the photo cuts off never
     joins across a seam, only as a near-copy of a house plane. A neighbour behind the target
     shares a long outline with it (its roof disappears behind the target's), so a shared
     outline is no evidence for a plane the frame has cut, which is what neighbours look like.
     Nor does one join that is small, lies mostly beside the house and is a different roofing
     material in similar light: a neighbour further back, seen past the target's porch.
  3. Of the planes left over: one the frame cuts off is a neighbour; one at least a quarter of
     the house's size is a second roof section a wall keeps apart from the first, like a
     garage, and stays, but only when it is near the house and the same roofing material as it
     (see _section); a smaller one mostly beside the house's width is a neighbour further back;
     a smaller one within it (a porch, a lower roof) stays.
  4. A house is one column of the photo: neighbours stand beside it, never above or below it. A
     plane lying mostly within the columns of the planes kept so far (and not cut by the frame)
     is the same house, repeated until nothing changes. This keeps an upper storey or turret
     roof behind a wall, which shares no outline with the roofs below and so never joins across
     a seam, and which rule 3 judged only against the seed's own seams (measured: a two-storey
     house lost its upper gables and turret, 12 correct lines, to rule 3).

Measured on the client's five acceptance photos and five others, before rule 4: every neighbour
plane is dropped, and no target plane is. Rule 4 has not been re-measured on those photos; a
neighbour's roof seen above the target's, within its width, is now kept. What neither geometry nor colour tells apart is a neighbour of
the same roofing material that shares a real outline with the target's porch or wing: it stays,
and one tap takes it out. Rules 2 and 3 also misjudge, by design, a plane of the house's own
roof that the photo's edge cuts off (dropped) and a large neighbour wholly in the photo that
touches nothing (kept); the tap corrects both.

The decisions are about whole planes, so they are made on a small copy of the masks. The roof
geometry still runs on every plane, so the house's own edges come out exactly as with the flag
off; on_house then takes the neighbours' lines away, evidence_region keeps their segments from
voting on the vanishing points, and plane_at finds the plane under a tap so the editor can move
it into or out of the house.
"""

import cv2
import numpy as np

WORK_EDGE = 640                 # px: long edge of the copy the plane decisions are made on
FRAME_FRACTION = 0.015          # of the width: a plane this close to the left or right edge is cut off by it
SEAM_REACH_FRACTION = 0.008     # of the long edge: boundary pixels this close to another plane count as shared
SEAM_MIN_FRACTION = 0.08        # of the smaller outline: a shared stretch this long is a seam, shorter is a corner
DUPLICATE_SHARE = 0.50          # of the smaller plane: overlap this large means SAM returned one plane twice
SECTION_SHARE = 0.25            # of the house's area: a separate plane this large is another section of it
INSIDE_SHARE = 0.50             # of a plane's width: this much within the house's width keeps a small plane
# Two roofs this far apart in colour (median CIELAB a*b*, 8-bit scale) are different materials,
# but only judged when they are lit alike: sun and shade change a roof's colour as well.
# Measured on ten photos: same-roof seams 1.0-6.3 apart, a neighbour set back behind a porch 9.1.
MATERIAL_AB = 8.0
MATERIAL_SAME_LIGHT = 25.0      # median L* difference, 8-bit scale
# A separate roof section (rule 3) joins only when it is this near the house (of the width) and
# no further than this in colour from the house's closest-coloured plane, whatever the light.
# Measured on 17 annotated photos: the house's own wings and garages lay 0.07-0.15 of the width
# away and 1.0-3.6 apart in colour; the neighbours kept as sections lay 0.06-0.29 away and 8.1-15.3
# apart, or 2.2-2.8 apart but 0.23-0.27 away. Neighbours were 11 of the 14 wrongly kept planes.
SECTION_GAP_FRACTION = 0.15
SECTION_AB = 6.5
EVIDENCE_MARGIN_FRACTION = 0.02  # of the width: a dropped plane's column is widened this much each side
EVIDENCE_EDGE = 1000            # px: long edge the evidence region is worked out at; segments are tested by midpoint
LINE_REACH_FRACTION = 0.006     # of the width: a line this close to a plane counts as lying on it
TAP_REACH_FRACTION = 0.01       # of the width: a tap this close to a plane selects it
OUTLINE_EPSILON_FRACTION = 0.002  # of the long edge: outlines sent to the editor are simplified this much


def _shrink(masks: list[np.ndarray]) -> list[np.ndarray]:
    height, width = masks[0].shape
    factor = min(1.0, WORK_EDGE / max(height, width))
    if factor >= 1.0:
        return [m.astype(bool) for m in masks]
    size = (max(1, round(width * factor)), max(1, round(height * factor)))
    return [cv2.resize(m.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool) for m in masks]


def cut_by_frame(mask: np.ndarray) -> bool:
    band = max(1, round(FRAME_FRACTION * mask.shape[1]))
    return bool(mask[:, :band].any() or mask[:, -band:].any())


def _seed(masks: list[np.ndarray], cut: list[bool]) -> int:
    """The largest roof near the middle of the photo; one the frame cuts off counts for less."""
    width = masks[0].shape[1]
    best, best_weight = 0, -1.0
    for index, mask in enumerate(masks):
        xs = np.nonzero(mask.any(axis=0))[0]
        area = float(mask.sum())
        if not area:
            continue
        columns = np.nonzero(mask)[1]
        centrality = 1.0 - min(1.0, abs(float(columns.mean()) - width / 2) / (width / 2))
        weight = area * (0.2 + centrality) ** 2 * (0.25 if cut[index] else 1.0)
        if xs.size and weight > best_weight:
            best, best_weight = index, weight
    return best


def _links(masks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(seams, copies): planes sharing a stretch of outline, and planes SAM returned twice."""
    count = len(masks)
    height, width = masks[0].shape
    reach = max(1, round(SEAM_REACH_FRACTION * max(height, width)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))
    small = np.ones((3, 3), np.uint8)
    grown = [cv2.dilate(m.astype(np.uint8), kernel).astype(bool) for m in masks]
    outline = [m & ~cv2.erode(m.astype(np.uint8), small).astype(bool) for m in masks]
    areas = [float(m.sum()) for m in masks]
    lengths = [float(o.sum()) for o in outline]
    seams = np.zeros((count, count), dtype=bool)
    copies = np.zeros((count, count), dtype=bool)
    for i in range(count):
        for j in range(i + 1, count):
            if not areas[i] or not areas[j]:
                continue
            overlap = float((masks[i] & masks[j]).sum())
            if overlap >= DUPLICATE_SHARE * min(areas[i], areas[j]):
                copies[i, j] = copies[j, i] = True
            shared = max(float((outline[i] & grown[j]).sum()), float((outline[j] & grown[i]).sum()))
            if shared >= SEAM_MIN_FRACTION * max(min(lengths[i], lengths[j]), 1.0):
                seams[i, j] = seams[j, i] = True
    return seams, copies


def _inside_share(mask: np.ndarray, x0: int, x1: int) -> float:
    """Share of a plane's width that lies between columns x0 and x1."""
    columns = np.nonzero(mask.any(axis=0))[0]
    if not columns.size:
        return 1.0
    return float(((columns >= x0) & (columns <= x1)).mean())


def _extent(masks: list[np.ndarray], members) -> tuple[float, int, int]:
    """(area, first column, last column) of a set of planes together."""
    union = np.logical_or.reduce([masks[index] for index in members])
    columns = np.nonzero(union.any(axis=0))[0]
    return float(union.sum()), int(columns[0]), int(columns[-1])


def _roof_colours(masks: list[np.ndarray], image) -> list[np.ndarray] | None:
    """Median CIELAB colour of each plane, away from its outline, or None without the photo."""
    if image is None:
        return None
    height, width = masks[0].shape
    rgb = np.asarray(image.convert("RGB")) if hasattr(image, "convert") else np.asarray(image)
    lab = cv2.cvtColor(cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2LAB)
    colours = []
    for mask in masks:
        core = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        pixels = lab[core if core.sum() > 20 else mask]
        colours.append(np.median(pixels, axis=0).astype(float) if len(pixels) else None)
    return colours


def _set_back(other: int, touching: int, house: set, masks: list[np.ndarray], colours) -> bool:
    """Is `other` a neighbour set back behind the house, though it shares an outline with it?

    A roof further back can disappear behind the target's porch or wing, and then shares a long
    stretch of outline with it, like a hip. What gives it away takes all three together: it is
    small next to the house, it lies mostly beside the house, and it is a different roofing
    material from the plane it touches, seen in similar light. Any one alone is ordinary for a
    wing of the house itself, so none of them vetoes a join on its own.
    """
    if colours is None or colours[other] is None or colours[touching] is None:
        return False
    area, x0, x1 = _extent(masks, house)
    if float(masks[other].sum()) >= SECTION_SHARE * area or _inside_share(masks[other], x0, x1) >= INSIDE_SHARE:
        return False
    (l1, a1, b1), (l2, a2, b2) = colours[touching], colours[other]
    return abs(l1 - l2) <= MATERIAL_SAME_LIGHT and float(np.hypot(a1 - a2, b1 - b2)) >= MATERIAL_AB


def _section(other: int, house: set, masks: list[np.ndarray], colours, near: np.ndarray) -> bool:
    """Is `other`, a large plane sharing no seam with the house, another section of it?

    A garage or wing behind its own wall shares no outline with the main roof, but it stands close
    to it and is roofed alike. A neighbour of similar size is further off, or a different roof:
    both are required to join. `near` is the distance (px) from every pixel to the house's planes.
    Without the photo only the distance is judged.
    """
    gap = float(near[masks[other]].min()) / masks[other].shape[1]
    if gap > SECTION_GAP_FRACTION:
        return False
    if colours is None or colours[other] is None:
        return True
    distances = [float(np.hypot(*(colours[other][1:] - colours[index][1:])))
                 for index in house if colours[index] is not None]
    return not distances or min(distances) <= SECTION_AB


def select(masks: list[np.ndarray], image=None) -> list[bool]:
    """Per plane: does it belong to the house being quoted? `image` (the photo) lets roof colour
    tell a neighbour set back behind the house from a wing of it; without it that test is skipped."""
    count = len(masks)
    if count <= 1:
        return [True] * count
    small = _shrink(masks)
    if not any(m.any() for m in small):
        return [True] * count
    cut = [cut_by_frame(m) for m in small]
    seed = _seed(small, cut)
    seams, copies = _links(small)
    colours = _roof_colours(small, image)

    # Rule 2: grow across seams and copies; a plane the frame cuts joins only as a copy, and a
    # neighbour set back behind the house does not join across the outline it shares with it.
    house, frontier = {seed}, [seed]
    while frontier:
        current = frontier.pop()
        for other in range(count):
            if other in house:
                continue
            if copies[current, other] or (
                seams[current, other] and not cut[other] and not _set_back(other, current, house, small, colours)
            ):
                house.add(other)
                frontier.append(other)

    # Rule 3: what is left over, judged against the house's size and width.
    area, x0, x1 = _extent(small, house)
    union = np.logical_or.reduce([small[index] for index in house])
    near = cv2.distanceTransform((~union).astype(np.uint8), cv2.DIST_L2, 3)
    keep = []
    for index in range(count):
        if index in house:
            keep.append(True)
        elif cut[index]:
            keep.append(False)
        elif float(small[index].sum()) >= SECTION_SHARE * area:
            keep.append(_section(index, house, small, colours, near))
        else:
            keep.append(_inside_share(small[index], x0, x1) >= INSIDE_SHARE)

    # Rule 4: a house is one column of the photo. Neighbours stand beside it; anything above or
    # below the house's own roofs is the same house: an upper storey or a turret behind a wall,
    # which shares no outline with the roofs below it and so never joined across a seam. Judged
    # against every plane kept so far, repeated until nothing changes. A plane the frame cuts off
    # still goes, as in rule 3.
    changed = True
    while changed:
        changed = False
        _, x0, x1 = _extent(small, [i for i in range(count) if keep[i]])
        for index in range(count):
            if not keep[index] and not cut[index] and small[index].any() \
                    and _inside_share(small[index], x0, x1) >= INSIDE_SHARE:
                keep[index] = True
                changed = True
    return keep


def evidence_region(masks: list[np.ndarray], keep: list[bool]) -> np.ndarray | None:
    """Where vanishing-point evidence may come from, or None for the whole photo.

    A dropped plane takes its column out of the evidence: its roof and the wall under it, from
    its top edge down, widened a little each side. The target house's own columns are never
    taken out, so a neighbour seen behind the target costs no facade evidence; its roof pixels
    are removed on their own. With nothing dropped the whole photo counts, exactly as with the
    flag off.
    """
    if all(keep) or not masks:
        return None
    full_height, full_width = masks[0].shape
    factor = min(1.0, EVIDENCE_EDGE / max(full_height, full_width))
    if factor < 1.0:
        size = (max(1, round(full_width * factor)), max(1, round(full_height * factor)))
        masks = [cv2.resize(m.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool) for m in masks]
    height, width = masks[0].shape
    margin = max(1, round(EVIDENCE_MARGIN_FRACTION * width))
    kept = np.zeros((height, width), dtype=bool)
    for mask, keep_it in zip(masks, keep):
        if keep_it:
            kept |= mask.astype(bool)
    target_columns = np.zeros(width, dtype=bool)
    columns = np.nonzero(kept.any(axis=0))[0]
    if columns.size:
        target_columns[columns[0]:columns[-1] + 1] = True

    excluded = np.zeros((height, width), dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
    for mask, keep_it in zip(masks, keep):
        if keep_it or not mask.any():
            continue
        rows = np.nonzero(mask.any(axis=1))[0]
        cols = np.nonzero(mask.any(axis=0))[0]
        x0, x1 = max(0, cols[0] - margin), min(width, cols[-1] + 1 + margin)
        y0 = max(0, rows[0] - margin)
        band = np.zeros(width, dtype=bool)
        band[x0:x1] = True
        band &= ~target_columns
        excluded[y0:, band] = True
        excluded |= cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    excluded &= ~kept
    region = ~excluded
    if factor < 1.0:
        region = cv2.resize(region.astype(np.uint8), (full_width, full_height), interpolation=cv2.INTER_NEAREST).astype(bool)
    return region


def on_house(lines: list[tuple], masks: list[np.ndarray], keep: list[bool]) -> list[tuple]:
    """The lines, as (points, side, edge), that belong to the house.

    The geometry runs on every plane, so the house's edges come out exactly as they do with the
    flag off; this then takes the neighbours' lines away. A line belongs to the house when at
    least as much of it lies on the house's planes as on the dropped ones. A line along an
    outline the house shares with a neighbour lies on both and stays, so no target edge is lost
    for touching a neighbour's roof. A tie that says nothing (a gutter just outside both, on
    neither) goes to the nearer plane, so a dropped roof's gutter goes with it.
    """
    if all(keep) or not lines or not masks:
        return list(lines)
    full_height, full_width = masks[0].shape
    factor = min(1.0, EVIDENCE_EDGE / max(full_height, full_width))
    size = (max(1, round(full_width * factor)), max(1, round(full_height * factor)))
    kept = np.zeros((size[1], size[0]), np.uint8)
    dropped = np.zeros_like(kept)
    for mask, keep_it in zip(masks, keep):
        small = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) if factor < 1 else mask.astype(np.uint8)
        (kept if keep_it else dropped)[small > 0] = 1
    reach = max(1, round(LINE_REACH_FRACTION * size[0]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))
    to_kept = cv2.distanceTransform((1 - kept).astype(np.uint8), cv2.DIST_L2, 3) if kept.any() else None
    to_dropped = cv2.distanceTransform((1 - dropped).astype(np.uint8), cv2.DIST_L2, 3) if dropped.any() else None
    kept, dropped = cv2.dilate(kept, kernel).astype(bool), cv2.dilate(dropped, kernel).astype(bool)

    out = []
    for line in lines:
        points = np.asarray(line[0], dtype=float) * factor
        samples = [points[0]]
        for a, b in zip(points[:-1], points[1:]):
            steps = max(1, int(np.linalg.norm(b - a)))
            samples += [a + (b - a) * (step / steps) for step in range(1, steps + 1)]
        samples = np.asarray(samples)
        xs = np.clip(np.rint(samples[:, 0]).astype(int), 0, size[0] - 1)
        ys = np.clip(np.rint(samples[:, 1]).astype(int), 0, size[1] - 1)
        on_kept, on_dropped = float(kept[ys, xs].mean()), float(dropped[ys, xs].mean())
        if on_kept == on_dropped and to_kept is not None and to_dropped is not None:
            if float(np.median(to_kept[ys, xs])) <= float(np.median(to_dropped[ys, xs])):
                out.append(line)
        elif on_kept >= on_dropped:
            out.append(line)
    return out


def copies_of(masks: list[np.ndarray], index: int) -> list[int]:
    """The plane and every plane SAM returned as a copy of it: one roof, so one tap moves them all."""
    small = _shrink(masks)
    area = float(small[index].sum())
    found = [index]
    for other, mask in enumerate(small):
        if other == index or not area or not mask.any():
            continue
        overlap = float((small[index] & mask).sum())
        if overlap >= DUPLICATE_SHARE * min(area, float(mask.sum())):
            found.append(other)
    return found


def plane_at(masks: list[np.ndarray], point: list[int] | tuple[int, int]) -> int | None:
    """The plane under a tap: the smallest one containing it, else the nearest within reach."""
    if not masks:
        return None
    height, width = masks[0].shape
    x, y = int(round(point[0])), int(round(point[1]))
    if not (0 <= x < width and 0 <= y < height):
        return None
    hits = [(float(m.sum()), i) for i, m in enumerate(masks) if m[y, x]]
    if hits:
        return min(hits)[1]
    reach = max(2, round(TAP_REACH_FRACTION * width))
    y0, y1, x0, x1 = max(0, y - reach), min(height, y + reach + 1), max(0, x - reach), min(width, x + reach + 1)
    best = None
    for index, mask in enumerate(masks):
        ys, xs = np.nonzero(mask[y0:y1, x0:x1])
        if not xs.size:
            continue
        distance = float(np.min(np.hypot(xs + x0 - x, ys + y0 - y)))
        if distance <= reach and (best is None or distance < best[0]):
            best = (distance, index)
    return best[1] if best else None


def outlines(masks: list[np.ndarray]) -> list[list[list[list[int]]]]:
    """Each plane's outline as simplified polygons in photo pixels, for the editor to draw."""
    out = []
    for mask in masks:
        height, width = mask.shape
        epsilon = max(1.0, OUTLINE_EPSILON_FRACTION * max(height, width))
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = float(mask.sum())
        polygons = []
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            if polygons and cv2.contourArea(contour) < 0.02 * area:
                break
            simplified = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
            if len(simplified) >= 3:
                polygons.append(simplified.astype(int).tolist())
        out.append(polygons)
    return out
