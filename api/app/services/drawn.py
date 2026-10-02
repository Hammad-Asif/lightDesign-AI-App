"""A roof line the salesperson drew by hand: where its ends join the lines already there, and what
it most likely is. The editor shows the suggestion with the type picker; one tap confirms or
changes it.

Ends. A hand-drawn end rarely lands on the pixel where a detected line ends, and the geometry
(services/scale.py) only treats ends within reach as one corner. So each end within SNAP_FRACTION
of an existing roof line's end moves onto it: a gap filled by hand then joins the lines on either
side, and a hip or ridge drawn to a corner hangs from it.

Type, by the first rule that applies, from the photo's vanishing points and what the ends touch,
never from image slope alone:
  1. it continues a line (an end meets that line's end, and the two run the same way): that
     line's type and side. This is the filled gap, the commonest hand-drawn line.
  2. it runs toward a facade's vanishing point (level on that wall): a ridge when an end meets a
     rake top, a hip top or a ridge, else an eave.
  3. it slopes: a rake when its top meets another rake's top (a gable peak); a hip when it rises
     from an eave's end to a ridge's end; a valley when it falls from a ridge's end to the inside
     of the roof; else a rake.
"""

import numpy as np

from app.services.camera import Camera

SNAP_FRACTION = 0.015       # of the photo's long edge: a drawn end this close to a line's end joins it
CONTINUE_DEG = 12.0         # a drawn line within this of a line it meets continues it
LEVEL_DEG = 6.0             # hand-drawn: looser than detection's 3 degrees toward a vanishing point
ROOF_EDGES = ("eave", "rake", "hip", "ridge", "valley", "junction")


def _ends(points) -> list[np.ndarray]:
    return [np.asarray(points[0], float), np.asarray(points[-1], float)]


def snap(polyline: list[list[float]], others: list[list[list[float]]], size: tuple[int, int]) -> list[list[float]]:
    """The polyline with each end moved onto the nearest existing line end within reach."""
    reach = SNAP_FRACTION * max(size)
    out = [list(map(float, p)) for p in polyline]
    for index in (0, len(out) - 1):
        end = np.asarray(out[index], float)
        best = None
        for points in others:
            for tip in _ends(points):
                gap = float(np.linalg.norm(tip - end))
                if gap <= reach and (best is None or gap < best[0]):
                    best = (gap, tip)
        if best is not None:
            out[index] = [float(best[1][0]), float(best[1][1])]
    return out


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    """Degrees between two directions, ignoring which way each points."""
    cos = abs(float(a @ b)) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-9)
    return float(np.degrees(np.arccos(min(1.0, cos))))


def suggest(polyline: list[list[float]], others: list[tuple[list[list[float]], str | None, str | None]],
            camera: Camera | None, size: tuple[int, int]) -> tuple[str, str | None]:
    """(edge, side) for a drawn roof line. `others` are the photo's roof lines as (points, edge,
    side); `camera` is the photo's (None: no vanishing points, so no level test)."""
    reach = SNAP_FRACTION * max(size)
    a, b = _ends(polyline)
    direction = b - a
    top, bottom = (a, b) if a[1] <= b[1] else (b, a)

    def touching(end):
        """(points, edge, side, which end of theirs is the upper) of each line ending at `end`."""
        found = []
        for points, edge, side in others:
            if edge not in ROOF_EDGES:
                continue
            first, last = _ends(points)
            for tip, other_tip in ((first, last), (last, first)):
                if float(np.linalg.norm(tip - end)) <= reach:
                    found.append((points, edge, side, tip[1] <= other_tip[1]))
        return found

    # 1. Continues a line it meets.
    for end in (a, b):
        for points, edge, side, _ in touching(end):
            first, last = _ends(points)
            if _angle(direction, last - first) <= CONTINUE_DEG:
                return edge, side

    at_top, at_bottom = touching(top), touching(bottom)
    meets_top = {edge for _, edge, _, upper in at_top + at_bottom if upper}   # their upper ends
    # 2. Level on a wall.
    facade = camera.facade_of(a, b, LEVEL_DEG) if camera is not None and camera.horizontal else None
    if facade is not None:
        high = bool(meets_top & {"rake", "hip"}) or any(edge == "ridge" for _, edge, _, _ in at_top + at_bottom)
        return ("ridge" if high else "eave"), facade
    # 3. Sloped.
    top_edges = {edge for _, edge, _, upper in at_top if upper} | {edge for _, edge, _, _ in at_top if edge == "ridge"}
    bottom_edges = {edge for _, edge, _, _ in at_bottom}
    if "rake" in top_edges:
        return "rake", None
    if "ridge" in top_edges and bottom_edges & {"eave", "junction"}:
        return "hip", None
    if "ridge" in top_edges and not bottom_edges:
        return "valley", None
    return "rake", None
