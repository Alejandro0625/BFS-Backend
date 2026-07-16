"""
auto_trim.py — derive linear-foot (LF) trim SUGGESTIONS from the face geometry the
engine already computes. Pure geometry, zero new dependencies.

Blueprint Phase 1c ("a killer differentiator — nobody automates trim"): the vertical
edges of the welded faces ARE the outside/inside corners; the horizontal top/bottom
edges ARE base & top trim; each detected opening's perimeter IS window/door trim.

Money-safety: these are SUGGESTIONS the estimator verifies, never confirmed LF. The
caller keeps them in a separate `autoTrim` field, apart from the estimator's own
`linearItems`, so an unverified number can never inflate a bid total.

Coincident edges (a wall corner shared by two materials, or collinear stacked pieces)
are merged via interval-union so a single physical trim line is counted once.
"""
import math

# an edge is "vertical" if it rises within ~30deg of plumb, "horizontal" if within
# ~30deg of level. The diagonal band (gable rakes) is reported separately so a steep
# roof line is not mislabeled as a corner.
_VERT_MIN_DEG = 60.0
_HORIZ_MAX_DEG = 30.0
_MIN_EDGE_PT = 3.0          # ignore sub-pixel slivers from polygon simplification
_COINCIDENT_PT = 6.0        # edges within this perpendicular distance are the same line


def _union_len(intervals):
    """Total length covered by 1-D intervals, overlaps counted once."""
    if not intervals:
        return 0.0
    intervals = sorted((min(a, b), max(a, b)) for a, b in intervals)
    total = 0.0
    cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            total += ce - cs
            cs, ce = s, e
    total += ce - cs
    return total


def _dedupe(edges, axis):
    """edges: list of (x0,y0,x1,y1) in points. Group collinear edges (same const
    coord within tolerance), union their spans so a shared corner counts once."""
    buckets = {}
    for x0, y0, x1, y1 in edges:
        if axis == "v":
            const = round((x0 + x1) / 2 / _COINCIDENT_PT)
            span = (y0, y1)
        else:
            const = round((y0 + y1) / 2 / _COINCIDENT_PT)
            span = (x0, x1)
        buckets.setdefault(const, []).append(span)
    return sum(_union_len(v) for v in buckets.values())


def compute(polys, pw, ph, ft_per_pt):
    """polys: engine faces with normalized `points` (0..1 display coords) and `holes`.
    Returns a list of auto-trim suggestion dicts (may be empty). Lengths in feet."""
    if not polys or not ft_per_pt or ft_per_pt <= 0:
        return []
    # welded pattern faces have toothed/stepped outlines: drop edges shorter than a real
    # trim run (~1 ft) so those micro-ledges don't inflate the base/top suggestion.
    min_run_pt = max(_MIN_EDGE_PT, 1.0 / ft_per_pt)
    edges_v, edges_h = [], []
    diag_pt = 0.0
    open_pt = 0.0
    for p in polys:
        pts = [(x * pw, y * ph) for x, y in (p.get("points") or [])]
        n = len(pts)
        for i in range(n):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % n]
            dx, dy = x1 - x0, y1 - y0
            L = math.hypot(dx, dy)
            if L < min_run_pt:
                continue
            ang = math.degrees(math.atan2(abs(dy), abs(dx)))
            if ang >= _VERT_MIN_DEG:
                edges_v.append((x0, y0, x1, y1))
            elif ang <= _HORIZ_MAX_DEG:
                edges_h.append((x0, y0, x1, y1))
            else:
                diag_pt += L                      # gable rakes / sloped transitions
        for h in (p.get("holes") or []):
            hp = [(x * pw, y * ph) for x, y in h]
            m = len(hp)
            for j in range(m):
                ax, ay = hp[j]
                bx, by = hp[(j + 1) % m]
                open_pt += math.hypot(bx - ax, by - ay)

    # EXTERIOR-ONLY CORNERS (LF bench 2026-07-15: autoTrim ran 2.67x over — after
    # first-paint splitting EVERY interior joint's vertical edge counted as a corner;
    # his takeoffs price BUILDING corners). A vertical line is exterior when cladding
    # exists on only ONE side of it; interior joints (pieces on both sides) are panel
    # joints, not corner trim.
    bbs = []
    for p in polys:
        xs = [x * pw for x, _ in (p.get("points") or [])]
        ys = [y * ph for _, y in (p.get("points") or [])]
        if xs and ys:
            bbs.append((min(xs), min(ys), max(xs), max(ys)))
    ext_v = []
    for (x0, y0, x1, y1) in edges_v:
        xc = (x0 + x1) / 2
        ylo, yhi = min(y0, y1), max(y0, y1)
        left = right = False
        for (bx0, by0, bx1, by1) in bbs:
            if min(yhi, by1) - max(ylo, by0) < 0.3 * (yhi - ylo):
                continue
            if bx0 < xc - 4:
                left = True
            if bx1 > xc + 4:
                right = True
            if left and right:
                break
        if not (left and right):
            ext_v.append((x0, y0, x1, y1))
    v_ft = _dedupe(ext_v, "v") * ft_per_pt
    diag_ft = diag_pt * ft_per_pt
    open_ft = open_pt * ft_per_pt

    # Base & top trim is the building's ground line + roofline — the silhouette width
    # run twice. PER VIEW-CLUSTER (LF bench iter-2: multi-view sheets hold several
    # elevations stacked vertically; one page-wide silhouette under-counts them, and
    # unrelated views inflate each other). Cluster faces by VERTICAL bands (views are
    # stacked rows on a sheet), silhouette per cluster, sum.
    boxes = []
    for p in polys:
        xs = [x for x, _ in (p.get("points") or [])]
        ys = [y for _, y in (p.get("points") or [])]
        if xs and ys:
            boxes.append((min(xs) * pw, min(ys) * ph, max(xs) * pw, max(ys) * ph))
    clusters = []
    # iter-3: 8% gap bar — stacked walls of ONE elevation (Fleet tower/base bands sit
    # ~2-6% apart) must stay ONE view; separate view rows on multi-view sheets sit
    # further apart. (0.02 doubled Fleet's base&top — battery spot-check caught it.)
    GAP = 0.08 * ph
    for b in sorted(boxes, key=lambda t: t[1]):
        for cl in clusters:
            if not (b[3] < cl["y0"] - GAP or b[1] > cl["y1"] + GAP):
                cl["y0"] = min(cl["y0"], b[1]); cl["y1"] = max(cl["y1"], b[3])
                cl["spans"].append((b[0], b[2]))
                break
        else:
            clusters.append({"y0": b[1], "y1": b[3], "spans": [(b[0], b[2])]})
    silhouette_pt = sum(_union_len(cl["spans"]) for cl in clusters)
    base_top_ft = 2.0 * silhouette_pt * ft_per_pt
    # (interior horizontal boundaries are intentionally NOT reported: on toothed weld
    #  outlines they are geometry noise, not real band-transition trim — a wrong LF number
    #  costs a bid, so we stay silent rather than guess.)

    out = []
    if v_ft >= 1:
        out.append({"material": "Corner trim (auto est.)", "kind": "corner",
                    "lf": round(v_ft, 1), "auto": True,
                    "note": "Vertical face edges = outside/inside corners. Verify against scope."})
    if base_top_ft >= 1:
        out.append({"material": "Base & top trim (auto est.)", "kind": "base_top",
                    "lf": round(base_top_ft, 1), "auto": True,
                    "note": "Clad width run twice (base / coping). Verify."})
    if open_ft >= 1:
        out.append({"material": "Opening trim (auto est.)", "kind": "opening",
                    "lf": round(open_ft, 1), "auto": True,
                    "note": "Perimeter of detected windows/doors. Verify counts."})
    if diag_ft >= 4:
        out.append({"material": "Rake / sloped trim (auto est.)", "kind": "rake",
                    "lf": round(diag_ft, 1), "auto": True,
                    "note": "Sloped face edges (gable rakes / transitions). Verify."})
    return out
