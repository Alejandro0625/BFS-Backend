"""Read the cladding patterns the architect actually DREW — no ML guessing.

The user's insight (correct): CAD exports carry the wall patterns as real vector geometry.
Two ways a panel wall shows up in these PDFs, and we read both:

  1) FILLED SLABS — walls drawn as light-gray filled closed paths (exact boundaries, the
     literal shape the architect drew). Exact SF via shoelace.
  2) SEAM-LINE TRAINS — walls drawn as fields of regularly-spaced parallel lines (panel
     joints / lap courses). We cluster the lines, reject text/tables/title-block, and
     integrate per-column strip heights so window/door gaps are netted out automatically.

Returns polys in the same contract as model_infer.detect() so app.py can prefer this
engine on clean pages and fall back to the model when a drawing has no usable vectors.
"""
import math
import fitz
import texture  # reuse the robust scale reader

MIN_SF = 100          # REVERTED: 30 let junk specks become stepping stones that chained welds
MAX_REGIONS = 40      # across the sheet into one sprawling blob (user-caught regression)


def _pct(v, p):
    s = sorted(v)
    return s[min(len(s) - 1, max(0, int(p / 100 * len(s))))]


def _shoelace(pts):
    a = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2


# ── engine 1: light-gray filled wall slabs ─────────────────────────────────
def _fill_regions(pg):
    W, H = pg.rect.width, pg.rect.height
    out = []
    for d in pg.get_drawings():
        f = d.get("fill")
        if not f or len(f) < 3:
            continue
        r, g, b = f[:3]
        # light gray, near-neutral = panel/wall poche (not logos, not black linework)
        if not (0.80 <= r <= 0.97 and abs(r - g) < 0.04 and abs(g - b) < 0.04):
            continue
        rect = d["rect"]
        if rect.width * rect.height < 0.0012 * W * H:
            continue
        pts = []
        for it in d["items"]:
            if it[0] == "l":
                pts += [(it[1].x, it[1].y), (it[2].x, it[2].y)]
            elif it[0] == "re":
                rr = it[1]
                pts += [(rr.x0, rr.y0), (rr.x1, rr.y0), (rr.x1, rr.y1), (rr.x0, rr.y1)]
            elif it[0] == "c":
                pts += [(it[1].x, it[1].y), (it[4].x, it[4].y)]
        clean = []
        for p in pts:
            if not clean or abs(clean[-1][0] - p[0]) > 0.5 or abs(clean[-1][1] - p[1]) > 0.5:
                clean.append(p)
        if len(clean) < 3:
            continue
        area = _shoelace(clean)
        # degenerate/self-crossing path — fall back to the fill's own bbox
        if area < 0.3 * rect.width * rect.height:
            clean = [(rect.x0, rect.y0), (rect.x1, rect.y0), (rect.x1, rect.y1), (rect.x0, rect.y1)]
            area = rect.width * rect.height
        out.append({"pts": clean, "area_pt2": area, "kind": "fill",
                    "bbox": (rect.x0, rect.y0, rect.x1, rect.y1)})
    return out


# ── engine 2: trains of regularly-spaced parallel seam lines ───────────────
def _collect_axis(pg, axis, min_len=40, stitch=5):
    """axis 'v': vertical lines as (x, y0, y1); axis 'h': horizontal as (y, x0, x1).
    Collinear fragments with gaps <= stitch merge (dashed seams). stitch is kept SMALL so
    window openings are NOT bridged (bridging inflates SF — the direction we can't afford)."""
    frags = {}
    cross = []
    for d in pg.get_drawings():
        for it in d["items"]:
            segs = []
            if it[0] == "l":
                segs = [(it[1].x, it[1].y, it[2].x, it[2].y)]
            elif it[0] == "re":
                r = it[1]
                segs = [(r.x0, r.y0, r.x1, r.y0), (r.x1, r.y0, r.x1, r.y1),
                        (r.x1, r.y1, r.x0, r.y1), (r.x0, r.y1, r.x0, r.y0)]
            for (x1, y1, x2, y2) in segs:
                if axis == "v":
                    if abs(x2 - x1) < 1.2 and abs(y2 - y1) >= 3:
                        frags.setdefault(round((x1 + x2) / 2, 0), []).append((min(y1, y2), max(y1, y2)))
                    elif abs(y2 - y1) < 1.5 and abs(x2 - x1) >= min_len:
                        cross.append((min(x1, x2), max(x1, x2), round((y1 + y2) / 2, 1)))
                else:
                    if abs(y2 - y1) < 1.2 and abs(x2 - x1) >= 3:
                        frags.setdefault(round((y1 + y2) / 2, 0), []).append((min(x1, x2), max(x1, x2)))
                    elif abs(x2 - x1) < 1.5 and abs(y2 - y1) >= min_len:
                        cross.append((min(y1, y2), max(y1, y2), round((x1 + x2) / 2, 1)))
    lines = []
    for c, iv in frags.items():
        iv.sort()
        cur = [iv[0][0], iv[0][1]]
        merged = []
        for (a, b) in iv[1:]:
            if a - cur[1] <= stitch:
                cur[1] = max(cur[1], b)
            else:
                merged.append(tuple(cur)); cur = [a, b]
        merged.append(tuple(cur))
        for (a, b) in merged:
            if b - a >= min_len:
                lines.append((c, a, b))
    return lines, cross


def _train_regions(pg, axis):
    W, H = pg.rect.width, pg.rect.height
    rot = pg.rotation_matrix
    lines, cross = _collect_axis(pg, axis)
    c1 = fitz.Point(0, 0) * rot
    c2 = fitz.Point(W, H) * rot
    RX = max(abs(c1.x), abs(c2.x))
    RY = max(abs(c1.y), abs(c2.y))
    lines.sort()
    n = len(lines)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        ci, a0i, a1i = lines[i]
        j = i + 1
        while j < n and lines[j][0] - ci < 45:
            cj, a0j, a1j = lines[j]
            ov = min(a1i, a1j) - max(a0i, a0j)
            if ov > 0.5 * min(a1i - a0i, a1j - a0j):
                union(i, j)
            j += 1
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(lines[i])
    regions = []
    for g in groups.values():
        if len(g) < 4:
            continue
        g.sort()
        cs = [v[0] for v in g]
        gaps = [cs[k + 1] - cs[k] for k in range(len(cs) - 1) if cs[k + 1] - cs[k] > 0.5]
        med = sorted(gaps)[len(gaps) // 2] if gaps else 0
        parts = [[g[0]]]
        for v in g[1:]:
            if v[0] - parts[-1][-1][0] > max(10 * med, 100):
                parts.append([v])
            else:
                parts[-1].append(v)
        for part in parts:
            pcs = sorted(set(v[0] for v in part))
            medH = _pct([v[2] - v[1] for v in part], 50)

            def col_ok(c):
                return sum(v[2] - v[1] for v in part if v[0] == c) >= 0.4 * medH

            while pcs and not col_ok(pcs[0]):
                pcs = pcs[1:]
            while pcs and not col_ok(pcs[-1]):
                pcs = pcs[:-1]
            if len(pcs) < 4:
                continue
            part = [v for v in part if pcs[0] <= v[0] <= pcs[-1]]
            pgaps = [pcs[k + 1] - pcs[k] for k in range(len(pcs) - 1) if pcs[k + 1] - pcs[k] > 0.5]
            if not pgaps:
                continue
            pmed = sorted(pgaps)[len(pgaps) // 2]
            reg = sum(1 for q in pgaps if 0.4 * pmed <= q <= 2.5 * pmed) / len(pgaps)
            heights = [v[2] - v[1] for v in part]
            tall = _pct(heights, 50) >= 120
            dense = len(pcs) >= 6 and 1.5 <= pmed <= 45 and reg >= 0.7
            sparse = len(pcs) >= 4 and 12 <= pmed <= 48 and reg >= 0.75 and tall
            if not (dense or sparse):
                continue
            c0, c1_ = pcs[0], pcs[-1]
            a0 = min(v[1] for v in part)
            a1 = max(v[2] for v in part)
            if (a1 - a0) < 50 or (c1_ - c0) < 30:
                continue
            # table/grid rejection: full-span cross lines ~ as numerous as pattern lines
            nx = sum(1 for (b0, b1, bc) in cross
                     if a0 - 2 < bc < a1 + 2 and min(b1, c1_) - max(b0, c0) > 0.7 * max(1, (c1_ - c0)))
            if nx >= 0.5 * len(pcs):
                continue
            # title-block exclusion (rotated display space): right 14% / bottom 8%
            if axis == "v":
                mid = fitz.Point((c0 + c1_) / 2, (a0 + a1) / 2) * rot
            else:
                mid = fitz.Point((a0 + a1) / 2, (c0 + c1_) / 2) * rot
            if abs(mid.x) > 0.86 * RX or abs(mid.y) > 0.92 * RY:
                continue
            # sheet-border strips: a thin full-length band hugging a page edge is the border/
            # dimension track, not a wall
            RW, RH = (H, W) if abs(rot.b) > 0.5 or abs(rot.c) > 0.5 else (W, H)
            bx0, bx1, by0, by1 = (c0, c1_, a0, a1) if axis == "v" else (a0, a1, c0, c1_)
            if (by1 - by0) < 0.06 * RH and (bx1 - bx0) > 0.7 * RW and (by0 < 0.03 * RH or by1 > 0.97 * RH):
                continue
            if (bx1 - bx0) < 0.06 * RW and (by1 - by0) > 0.7 * RH and (bx0 < 0.03 * RW or bx1 > 0.97 * RW):
                continue
            topC = _pct([v[1] for v in part], 10) - 6
            botC = _pct([v[2] for v in part], 90) + 6
            part = [(c, max(a, topC), min(b, botC)) for (c, a, b) in part if min(b, botC) - max(a, topC) > 10]
            if len(part) < 4:
                continue
            regions.append({"lines": part, "axis": axis, "spacing": pmed, "kind": "train",
                            "med_len": _pct([v[2] - v[1] for v in part], 50)})
    return regions


def _train_area_pt2(reg):
    """Strip integral: spacing x local line length — openings that interrupt the seams
    are automatically excluded (never inflate SF)."""
    part = reg["lines"]
    cs = sorted(set(v[0] for v in part))
    Hh = {}
    for (c, a, b) in part:
        Hh[c] = max(Hh.get(c, 0), b - a)
    A = 0.0
    for i, c in enumerate(cs):
        l = (cs[i] - cs[i - 1]) / 2 if i > 0 else 0
        r = (cs[i + 1] - cs[i]) / 2 if i < len(cs) - 1 else 0
        A += (l + r) * Hh[c]
    return A


def _train_polygon(reg, snapx=(), snapy=()):
    """Rectilinear outline from runs of similar extents, with every edge SNAPPED to the
    drawing's real structural lines — the border lands on actual corners, not near them."""
    part = reg["lines"]

    def sn(v, cands, tol=9):
        best = None; bd = tol
        for c in cands:
            d = abs(c - v)
            if d < bd:
                bd = d; best = c
        return best if best is not None else v

    cols = {}
    for (c, a, b) in part:
        e = cols.setdefault(c, [a, b])
        e[0] = min(e[0], a); e[1] = max(e[1], b)
    cs = sorted(cols)
    runs = []
    for c in cs:
        t, b = cols[c]
        # 36pt (~4.5ft at 1/8") run tolerance: real parapet steps survive, stair-noise collapses
        if runs and abs(runs[-1][2] - t) < 36 and abs(runs[-1][3] - b) < 36:
            runs[-1] = (runs[-1][0], c, min(runs[-1][2], t), max(runs[-1][3], b))
        else:
            runs.append((c, c, t, b))
    ext = snapy if reg["axis"] == "v" else snapx     # run tops/bottoms snap along the line axis
    runs = [(ca, cb, sn(t, ext), sn(b, ext)) for (ca, cb, t, b) in runs]
    pts = []
    for (ca, cb, t, b) in runs:
        pts += [(ca, t), (cb, t)]
    for (ca, cb, t, b) in reversed(runs):
        pts += [(cb, b), (ca, b)]
    if reg["axis"] == "h":
        pts = [(y, x) for (x, y) in pts]
    try:  # collapse micro stair-steps (<~5pt) so the border reads like an estimator drew it
        import numpy as np, cv2
        arr = np.array(pts, np.float32).reshape(-1, 1, 2)
        ap = cv2.approxPolyDP(arr, 5.0, True).reshape(-1, 2)
        if len(ap) >= 4:
            pts = [(float(x), float(y)) for x, y in ap]
    except Exception:
        pass
    return pts


def _pip_pt(pt, poly):
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i - 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
    return inside


def _collect_rects(pg):
    """All axis-aligned rectangles on the page (display coords): 're' items plus closed
    4-line paths (CAD exports draw window frames both ways). Returns [(x0,y0,x1,y1)]."""
    rot = pg.rotation_matrix
    rects = []

    def add_rect_pts(x0, y0, x1, y1):
        a = fitz.Point(x0, y0) * rot
        b = fitz.Point(x1, y1) * rot
        rects.append((min(a.x, b.x), min(a.y, b.y), max(a.x, b.x), max(a.y, b.y)))

    try:
        for d in pg.get_drawings():
            items = d.get("items", [])
            res = [it for it in items if it[0] == "re"]
            for it in res:
                r = it[1]
                if r.width > 2 and r.height > 2:
                    add_rect_pts(r.x0, r.y0, r.x1, r.y1)
            # closed line-loop that traces its own bbox = a drawn rectangle
            ls = [it for it in items if it[0] == "l"]
            if 3 <= len(ls) <= 6 and not res:
                r = d["rect"]
                if r.width > 2 and r.height > 2:
                    per = sum(((it[1].x - it[2].x) ** 2 + (it[1].y - it[2].y) ** 2) ** 0.5 for it in ls)
                    if abs(per - 2 * (r.width + r.height)) < 0.25 * (r.width + r.height):
                        add_rect_pts(r.x0, r.y0, r.x1, r.y1)
    except Exception:
        pass
    return rects[:6000]


def page_geometry(pg):
    """One-pass page geometry for opening detection (display coords): rectangles + segments."""
    segs = []
    try:
        rot = pg.rotation_matrix
        for d in pg.get_drawings():
            for it in d.get("items", []):
                if it[0] == "l":
                    a = fitz.Point(it[1].x, it[1].y) * rot
                    b = fitz.Point(it[2].x, it[2].y) * rot
                    segs.append((a.x, a.y, b.x, b.y))
    except Exception:
        pass
    return {"rects": _collect_rects(pg), "segs": segs[:30000]}


def find_openings(geo, pts_disp, ft_pt):
    """READ WINDOWS/DOORS THE WAY AN ESTIMATOR DOES. Inside the wall polygon (display coords),
    an opening is an axis-aligned rectangle that (a) is a believable window/door size at this
    scale, (b) shows FENESTRATION EVIDENCE — a nested frame rectangle or interior mullion
    lines — and (c) doors may touch the wall's bottom edge. Returns (opening_polys_norm_NONE —
    caller normalizes), list of rect tuples, total opening area pt^2. MONEY-SAFE: evidence
    required (never deduct a bare reference box), dedupe nested frames (outermost wins),
    total deduction capped at 40% of the wall."""
    rects = geo.get("rects") or []
    if not rects:
        return [], 0.0
    xs = [p[0] for p in pts_disp]; ys = [p[1] for p in pts_disp]
    bx0, bx1, by0, by1 = min(xs), max(xs), min(ys), max(ys)
    region_w = max(1.0, bx1 - bx0)
    wall_area = _shoelace(pts_disp)
    # candidate rects: inside the wall, plausible physical size
    cands = []
    for (x0, y0, x1, y1) in rects:
        w_ft = (x1 - x0) * ft_pt
        h_ft = (y1 - y0) * ft_pt
        if not (1.2 <= w_ft <= 22 and 1.2 <= h_ft <= 16):
            continue
        if (x1 - x0) > 0.8 * region_w:          # full-width band = wall articulation, not a window
            continue
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        corners_in = sum(1 for (px, py) in ((x0 + 1, y0 + 1), (x1 - 1, y0 + 1), (x1 - 1, y1 - 1), (x0 + 1, y1 - 1))
                         if _pip_pt((px, py), pts_disp))
        if corners_in < 3 or not _pip_pt((cx, cy), pts_disp):
            continue
        cands.append((x0, y0, x1, y1))
    if not cands:
        return [], 0.0
    # dedupe nested frames: outermost rect of each concentric family wins
    cands.sort(key=lambda r: -(r[2] - r[0]) * (r[3] - r[1]))
    keep = []
    nested_count = {}
    for r in cands:
        host = None
        for k in keep:
            if r[0] >= k[0] - 2 and r[1] >= k[1] - 2 and r[2] <= k[2] + 2 and r[3] <= k[3] + 2:
                host = k; break
        if host is not None:
            nested_count[host] = nested_count.get(host, 0) + 1
        else:
            keep.append(r); nested_count[r] = 0
    # evidence gate: nested frame OR interior mullion/glazing linework
    segs = geo.get("segs") or []
    openings = []
    for r in keep:
        x0, y0, x1, y1 = r
        if nested_count.get(r, 0) >= 1:
            openings.append(r); continue
        mull = 0
        for (sx0, sy0, sx1, sy1) in segs:
            mx, my = (sx0 + sx1) / 2, (sy0 + sy1) / 2
            if x0 + 2 < mx < x1 - 2 and y0 + 2 < my < y1 - 2:
                L = ((sx1 - sx0) ** 2 + (sy1 - sy0) ** 2) ** 0.5
                if L >= 0.35 * min(x1 - x0, y1 - y0):
                    mull += 1
                    if mull >= 2:
                        break
        if mull >= 2:
            openings.append(r)
    if not openings:
        return [], 0.0
    # cap the deduction — a wall can never lose more than 40% to detected openings
    openings.sort(key=lambda r: -(r[2] - r[0]) * (r[3] - r[1]))
    total = 0.0
    final = []
    cap = 0.40 * wall_area
    for (x0, y0, x1, y1) in openings[:24]:
        a = (x1 - x0) * (y1 - y0)
        if total + a > cap:
            continue
        total += a
        final.append((x0, y0, x1, y1))
    return final, total


def detect(pdf_bytes, page_index, zoom=None):
    """Same contract as model_infer.detect(): (polys, w, h, scale_info)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    sc, conf = texture._read_scale(doc, page_index)
    pg = doc[page_index]
    W, H = pg.rect.width, pg.rect.height
    rot = pg.rotation_matrix
    ft_pt = sc / 72.0

    fills = _fill_regions(pg)
    trains = _train_regions(pg, "v") + _train_regions(pg, "h")
    # structural snap targets: the page's long vertical Xs and horizontal Ys
    _vl, _hc = _collect_axis(pg, "v")
    snapx = sorted(set(c for (c, _, _) in _vl))[:800]
    snapy = sorted(set(y for (_, _, y) in _hc))[:800]
    geo = page_geometry(pg)   # rectangles + segments for window/door detection (display coords)
    doc.close()

    # fills win where they overlap a train (they are the exact drawn shape)
    def bbox_of(reg):
        if reg["kind"] == "fill":
            return reg["bbox"]
        part = reg["lines"]
        c0 = min(v[0] for v in part); c1_ = max(v[0] for v in part)
        a0 = min(v[1] for v in part); a1 = max(v[2] for v in part)
        return (c0, a0, c1_, a1) if reg["axis"] == "v" else (a0, c0, a1, c1_)

    kept_trains = []
    for t in trains:
        tb = bbox_of(t)
        ta = max(1e-6, (tb[2] - tb[0]) * (tb[3] - tb[1]))
        ov = 0.0
        for f in fills:
            fb = f["bbox"]
            ix = max(0, min(tb[2], fb[2]) - max(tb[0], fb[0]))
            iy = max(0, min(tb[3], fb[3]) - max(tb[1], fb[1]))
            ov += ix * iy
        if ov / ta < 0.5:
            kept_trains.append(t)

    polys = []

    def add(pts_page, area_pt2, material, color, sf_exact, deduct_openings=False):
        disp_pts = [((fitz.Point(x, y) * rot).x, (fitz.Point(x, y) * rot).y) for (x, y) in pts_page]
        # WINDOWS/DOORS: detect evidence-backed openings inside this wall. For plain-fill walls
        # SUBTRACT them (their shoelace SF was gross); pattern walls' strip-integral is already
        # net, so openings attach for DISPLAY only (honest cut-outs, no double deduction).
        holes_norm = []
        try:
            op_rects, op_pt2 = find_openings(geo, disp_pts, ft_pt)
            for (ox0, oy0, ox1, oy1) in op_rects:
                holes_norm.append([[round(ox0 / W, 5), round(oy0 / H, 5)], [round(ox1 / W, 5), round(oy0 / H, 5)],
                                   [round(ox1 / W, 5), round(oy1 / H, 5)], [round(ox0 / W, 5), round(oy1 / H, 5)]])
            if deduct_openings and op_pt2 > 0:
                area_pt2 = max(0.0, area_pt2 - op_pt2)
                sf_exact = True     # net-of-openings: protect from shoelace recompute (would re-add windows)
        except Exception:
            pass
        sf = area_pt2 * ft_pt * ft_pt
        if sf < MIN_SF:
            return
        norm = [[round(x / W, 5), round(y / H, 5)] for (x, y) in disp_pts]
        cx = round(sum(p[0] for p in norm) / len(norm), 5)
        cy = round(sum(p[1] for p in norm) / len(norm), 5)
        polys.append({"points": norm, "area_sf": round(sf, 1), "cx": cx, "cy": cy,
                      "fill_color": color, "source": "vector", "material": material,
                      "category": material, "group": material, "sf_exact": sf_exact,
                      "holes": holes_norm[:24],
                      "label": f"~{round(sf):,} SF"})

    for f in fills:
        add(f["pts"], f["area_pt2"], "Panel wall (drawn fill)", [0.35, 0.55, 0.85], False, deduct_openings=True)
    for t in kept_trains:
        sp_ft = t["spacing"] * ft_pt
        len_ft = t["med_len"] * ft_pt
        # physical plausibility: real panel seams are inches to ~4' apart on walls up to ~45'
        # tall. Site-plan artifacts (parking stripes at 1"=20', column grids) fail these.
        if sp_ft > 4.2 or len_ft > 45:
            continue
        if t["axis"] == "v":
            mat = f"Vertical panel - {sp_ft:.1f}' seams" if sp_ft >= 0.8 else "Vertical rib panel"
            col = [0.0, 0.72, 0.85]
        else:
            mat = f"Lap / horizontal - {sp_ft * 12:.0f}in courses" if sp_ft < 2 else f"Horizontal panel - {sp_ft:.1f}'"
            col = [0.25, 0.75, 0.35]
        # strip-integral SF is net of openings — trust it, don't let calibrate re-add openings
        add(_train_polygon(t, snapx, snapy), _train_area_pt2(t), mat, col, True)

    # overlap dedup — regions must never double-count the same wall into the total.
    # Rasterize at low res; drop a region mostly covered by earlier (bigger/fill) ones,
    # and shave the SF of partial overlaps so the summed total stays honest.
    try:
        import numpy as np, cv2
        MW = 900
        MH = max(1, int(MW * H / max(1, W)))
        order = sorted(range(len(polys)), key=lambda i: (0 if polys[i]["material"].startswith("Panel wall") else 1, -polys[i]["area_sf"]))
        covered = np.zeros((MH, MW), np.uint8)
        keep = []
        for i in order:
            p = polys[i]
            cnt = np.array([[int(x * MW), int(y * MH)] for x, y in p["points"]], np.int32)
            m = np.zeros((MH, MW), np.uint8)
            cv2.fillPoly(m, [cnt], 1)
            a = int(m.sum())
            if a == 0:
                continue
            ov = int((m & covered).sum()) / a
            if ov > 0.55:
                continue
            if ov > 0.05:
                p = dict(p); p["area_sf"] = round(p["area_sf"] * (1 - ov), 1)
            covered |= m
            keep.append(p)
        polys = keep
    except Exception:
        pass

    polys.sort(key=lambda p: -p["area_sf"])
    polys = [p for p in polys if p["area_sf"] >= MIN_SF][:MAX_REGIONS]
    polys = weld_faces(polys)   # FIRST IMPRESSION = clean faces: same-pattern pieces welded into one
    for i, p in enumerate(polys):
        p["id"] = i
    return polys, W, H, {"ft_per_in": round(sc, 3), "scale_confirmed": conf}


def weld_faces(polys, gap=0.015, raster=1600):
    """THE FACE IS ONE PIECE. Cluster same-pattern regions whose bboxes nearly touch (butt-
    joined pieces of one wall) and weld each cluster into a single clean outline. SF = SUM of
    members' own (already net) SF — welding never invents area; holes carried through. Also
    smooths single regions' staircase micro-notches (the 'zigzag' first impression) via the
    same raster-close pass. Different materials that touch never weld (same group only)."""
    try:
        import numpy as np
        import cv2 as _cv
    except Exception:
        return polys

    def bbox(p):
        xs = [q[0] for q in p["points"]]; ys = [q[1] for q in p["points"]]
        return (min(xs), min(ys), max(xs), max(ys))

    out = []
    remaining = list(polys)
    while remaining:
        seed = remaining.pop(0)
        grp = seed.get("group") or seed.get("material")
        members = [seed]
        grew = True
        while grew:
            grew = False
            mb = [bbox(m) for m in members]
            keep = []
            cb = (min(m[0] for m in mb), min(m[1] for m in mb), max(m[2] for m in mb), max(m[3] for m in mb))
            for p in remaining:
                if (p.get("group") or p.get("material")) != grp:
                    keep.append(p); continue
                b = bbox(p)
                near = any(not (b[2] < m[0] - gap or b[0] > m[2] + gap or b[3] < m[1] - gap or b[1] > m[3] + gap) for m in mb)
                # ANTI-SPRAWL: a weld may never grow into a page-spanning blob — one wall face
                # is wide OR tall, never half the sheet in BOTH directions
                nb = (min(cb[0], b[0]), min(cb[1], b[1]), max(cb[2], b[2]), max(cb[3], b[3]))
                if near and (nb[2] - nb[0]) > 0.55 and (nb[3] - nb[1]) > 0.55:
                    near = False
                if near:
                    members.append(p); grew = True; cb = nb
                else:
                    keep.append(p)
            remaining = keep
        # weld the cluster (also smooths a single piece's stair-noise)
        try:
            S = raster; SH = raster
            m2 = np.zeros((SH, S), np.uint8)
            for p in members:
                cnt = np.array([[int(x * S), int(y * SH)] for x, y in p["points"]], np.int32)
                _cv.fillPoly(m2, [cnt.reshape(-1, 1, 2)], 1)
            k = max(3, int(gap * S * 0.8))
            m2 = _cv.morphologyEx(m2, _cv.MORPH_CLOSE, np.ones((k, k), np.uint8))
            # shave thin SPURS (leader lines poking out of the face) — outline only, SF untouched
            ko = max(3, k // 2)
            m2 = _cv.morphologyEx(m2, _cv.MORPH_OPEN, np.ones((ko, ko), np.uint8))
            cnts, _ = _cv.findContours(m2, _cv.RETR_EXTERNAL, _cv.CHAIN_APPROX_SIMPLE)
            if not cnts:
                out.extend(members); continue
            c = max(cnts, key=_cv.contourArea)
            ap = _cv.approxPolyDP(c, 0.008 * _cv.arcLength(c, True), True).reshape(-1, 2)   # straighter, estimator-looking edges
            if len(ap) < 3:
                out.extend(members); continue
            # drop NEEDLE vertices (sharp slivers poking into text/dimension areas)
            import math as _math
            def _ang(a, b, cpt):
                v1 = (a[0] - b[0], a[1] - b[1]); v2 = (cpt[0] - b[0], cpt[1] - b[1])
                d1 = _math.hypot(*v1); d2 = _math.hypot(*v2)
                if d1 * d2 == 0:
                    return 180.0
                cs = max(-1, min(1, (v1[0] * v2[0] + v1[1] * v2[1]) / (d1 * d2)))
                return _math.degrees(_math.acos(cs))
            for _ in range(3):
                if len(ap) <= 4:
                    break
                keepv = [i for i in range(len(ap))
                         if _ang(ap[i - 1], ap[i], ap[(i + 1) % len(ap)]) > 22]
                if len(keepv) == len(ap) or len(keepv) < 3:
                    break
                ap = ap[keepv]
            face = dict(members[0])
            face["points"] = [[round(float(x) / S, 5), round(float(y) / SH, 5)] for x, y in ap]
            face["area_sf"] = round(sum(p.get("area_sf", 0) for p in members), 1)
            holes = []
            for p in members:
                holes += (p.get("holes") or [])
            face["holes"] = holes[:24]
            face["cx"] = round(sum(pt[0] for pt in face["points"]) / len(face["points"]), 5)
            face["cy"] = round(sum(pt[1] for pt in face["points"]) / len(face["points"]), 5)
            if len(members) > 1:
                face["sf_exact"] = True       # summed nets: protect from shoelace recompute
                face["merged_pieces"] = len(members)
            face["label"] = f"~{round(face['area_sf']):,} SF"
            out.append(face)
        except Exception:
            out.extend(members)
    out.sort(key=lambda p: -p.get("area_sf", 0))
    return out
