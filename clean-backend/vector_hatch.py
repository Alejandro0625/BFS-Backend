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
MAX_REGIONS = 120     # 80 capped out on 113-wall multifamily sheets (Avita p3/p4: real walls
                      # were budget-starved); sprawl-era risks are held by the anti-sprawl
                      # guard + per-reader gates, not the cap
SMALL_MIN_SF = 30     # small faces (30-100 SF) survive ONLY via the discriminator below:
SMALL_ADOPT_GAP = 0.06  # same pattern as a BIG face AND bbox within 6% of the sheet of that
                        # big face DIRECTLY (never small-to-small — no stepping-stone chains)


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


def _train_regions(pg, axis, chain=False):
    W, H = pg.rect.width, pg.rect.height
    rot = pg.rotation_matrix
    lines, cross = _collect_axis(pg, axis)
    c1 = fitz.Point(0, 0) * rot
    c2 = fitz.Point(W, H) * rot
    RX = max(abs(c1.x), abs(c2.x))
    RY = max(abs(c1.y), abs(c2.y))
    lines.sort()
    # (Duplicate-stroke collapse was TRIED 2026-07-10 for 26-191A's 1pt double-stroke
    # courses and REVERTED: it moved the Fleet canary (7749->7736/6452->6466) and cost
    # a Danbury money wall while NOT capturing the target band — the band's failure is
    # downstream of line clustering. Off-by-one-band capture needs a different design.)
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
            # FRAGMENTED COURSES (26-088 view 3: lap lines drawn as ~32pt dashes — they
            # never overlap, so no cluster ever formed): fragments of the SAME course
            # line (within 3pt) chain across gaps ≤18pt. Area stays honest — the strip
            # integral sums only actual ink, never the gaps (the old window-bridging
            # lesson stays enforced).
            elif chain and abs(cj - ci) <= 3 and ov > -18:
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
    # clamp outline to the DOMINANT band — dimension/leader verticals that slipped into the
    # train can't spike the display shape (SF comes from the strip integral, untouched)
    if len(runs) >= 3:
        topsP = _pct([r[2] for r in runs], 15) - 10
        botsP = _pct([r[3] for r in runs], 85) + 10
        runs = [(ca, cb, max(t, topsP), min(b, botsP)) for (ca, cb, t, b) in runs if min(b, botsP) - max(t, topsP) > 4]
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


def view_boxes(geo, W, H):
    """Partition a multi-view sheet into its drawing-view blocks by clustering the ink.
    Band/story logic must run PER VIEW — story lines stitched across separate elevation
    views created phantom walls (band-detector v1 lesson)."""
    try:
        import numpy as np, cv2
        GS = 48.0
        gw, gh = int(W / GS) + 2, int(H / GS) + 2
        m = np.zeros((gh, gw), np.uint8)
        for (x1, y1, x2, y2) in (geo.get("segs") or [])[:30000]:
            mx, my = int((x1 + x2) / 2 / GS), int((y1 + y2) / 2 / GS)
            if 0 <= my < gh and 0 <= mx < gw:
                m[my, mx] = 1
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
        out = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < 10:
                continue                 # note blocks / stray ticks
            out.append((x * GS, y * GS, (x + w) * GS, (y + h) * GS))
        return out
    except Exception:
        return []


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
    # STRICT no-regression construction: no-chain regions are the canon (identical to
    # prior behavior); chain-formed regions (fragmented courses, 26-088 view 3) are
    # added ONLY where no canon region exists — new coverage is pure gain.
    def _bbox_of_train(t):
        part = t["lines"]
        c0 = min(v[0] for v in part); c1_ = max(v[0] for v in part)
        a0 = min(v[1] for v in part); a1 = max(v[2] for v in part)
        return (a0, c0, a1, c1_) if t["axis"] == "h" else (c0, a0, c1_, a1)
    trains = []
    for ax in ("v", "h"):
        base = _train_regions(pg, ax, chain=False)
        extra = []
        bbs = [_bbox_of_train(t) for t in base]
        for t in _train_regions(pg, ax, chain=True):
            tb = _bbox_of_train(t)
            ta = max(1e-6, (tb[2] - tb[0]) * (tb[3] - tb[1]))
            ov = 0.0
            for bb in bbs:
                ix = max(0, min(tb[2], bb[2]) - max(tb[0], bb[0]))
                iy = max(0, min(tb[3], bb[3]) - max(tb[1], bb[1]))
                ov += ix * iy
            if ov / ta < 0.3:
                extra.append(t)
        trains += base + extra
    # structural snap targets: the page's long vertical Xs and horizontal Ys
    _vl, _hc = _collect_axis(pg, "v")
    snapx = sorted(set(c for (c, _, _) in _vl))[:800]
    snapy = sorted(set(y for (_, _, y) in _hc))[:800]
    geo = page_geometry(pg)   # rectangles + segments for window/door detection (display coords)
    rot_for_sig = pg.rotation_matrix
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
        if sf < SMALL_MIN_SF:
            return
        # single-region anti-sprawl: one region bigger than half the sheet in BOTH dims
        # is never one wall (Fenn: an 8,617sf 'train' spanning 0.95x0.69 of the sheet —
        # the weld-cluster guard never applied to single regions)
        _dx = [x for x, _ in disp_pts]; _dy = [y for _, y in disp_pts]
        if (max(_dx) - min(_dx)) > 0.55 * W and (max(_dy) - min(_dy)) > 0.55 * H:
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

    # DIAGONAL HATCH — the industry's most common material convention (masonry/EIFS/
    # stone drawn as parallel 45-deg strokes). Benchmark-found blind spot (26-145's
    # storefront band = 513 diagonal segs the v/h readers never see). THE GATE that
    # makes it safe: true hatch is PARALLEL — one angle family; flattened text is
    # angle-soup. A region only qualifies when >=70% of the diagonal ink inside it
    # belongs to ONE angle bucket, so glyphs/arrowheads can never form a region.
    try:
        import math as _m
        import os as _os
        if _os.environ.get("VH_NO_DIAG"):
            raise RuntimeError("diagonal reader disabled by env")
        import numpy as np, cv2
        doc_d = fitz.open(stream=pdf_bytes, filetype="pdf")
        pg_d = doc_d[page_index]
        rot_d = pg_d.rotation_matrix
        # ROOF pages are LAYERED by convention (membrane over crickets/pads = separate
        # pay items): a cricket's diagonal hatch must never claim territory FROM the
        # membrane host — patch-on-host stays wall-only (26-045 lost its money wall
        # to a cricket patch on the first gate run).
        try:
            _txt_d = (pg_d.get_text() or "").upper()
            roof_page_d = sum(1 for t in _ROOF_TERMS if t in _txt_d) >= 2
        except Exception:
            roof_page_d = False
        diag = []                       # (x0,y0,x1,y1,angle_deg,len) display coords
        for d in pg_d.get_drawings():
            for it in d.get("items") or []:
                if it[0] != "l":
                    continue
                p0 = fitz.Point(it[1]) * rot_d
                p1 = fitz.Point(it[2]) * rot_d
                dx, dy = p1.x - p0.x, p1.y - p0.y
                L = (dx * dx + dy * dy) ** 0.5
                # hatch strokes: short-to-medium; X-braces/leader arrows are LONG
                if L < 7 or L > 0.22 * max(W, H):
                    continue
                ang = _m.degrees(_m.atan2(dy, dx)) % 180.0
                if 24.0 <= ang <= 66.0 or 114.0 <= ang <= 156.0:
                    diag.append((p0.x, p0.y, p1.x, p1.y, ang, L))
        doc_d.close()
        # dominant angle FAMILIES (8-deg buckets); only their segs may form regions
        fam_len = {}
        for (_, _, _, _, ang, L) in diag:
            k = int(ang // 8)
            fam_len[k] = fam_len.get(k, 0.0) + L
        strong = sorted([k for k, v in fam_len.items() if v > 400], key=lambda k: -fam_len[k])[:3]
        diag_by_fam = {k: [(x0, y0, x1, y1, ang, L) for (x0, y0, x1, y1, ang, L) in diag
                           if int(ang // 8) == k] for k in strong}
        diag_all = diag
        if strong and len(diag) >= 40:  # a hatched area is DENSE; scattered ticks are not
            DW = 900
            DH = max(1, int(DW * H / max(1, W)))
            sx, sy = DW / W, DH / H
            # NEVER re-count a wall another reader already measured: hatch regions may
            # only claim VIRGIN territory (Fleet's brick is already train-measured — a
            # hatch echo there would double the money).
            taken = np.zeros((DH, DW), np.uint8)
            for p0_ in polys:
                try:
                    cnt0 = np.array([[int(x * W * sx), int(y * H * sy)] for x, y in p0_["points"]], np.int32)
                    cv2.fillPoly(taken, [cnt0], 1)
                except Exception:
                    pass
            _pm_buf = np.zeros((DH, DW), np.uint8)   # reusable host-mask buffer
            for fam_k in strong:
              fam = diag_by_fam.get(fam_k) or []
              if len(fam) < 25:
                continue
              m = np.zeros((DH, DW), np.uint8)
              for (x0d, y0d, x1d, y1d, _a, _l) in fam:
                cv2.line(m, (int(x0d * sx), int(y0d * sy)), (int(x1d * sx), int(y1d * sy)), 1, 2)
              # bridge the gaps BETWEEN hatch strokes, then drop thin strays
              m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
              m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
              cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
              for ci, cnt in enumerate(cnts or []):
                if hier is not None and hier[0][ci][3] != -1:
                    continue            # holes handled via find_openings below
                a_px = cv2.contourArea(cnt)
                area_pt2 = a_px / (sx * sy)
                sf = area_pt2 * ft_pt * ft_pt
                if sf < MIN_SF:
                    continue
                bx, by, bw, bh = cv2.boundingRect(cnt)
                if bw > 0.55 * DW and bh > 0.55 * DH:
                    continue            # anti-sprawl: site-plan / whole-sheet junk
                cm = np.zeros((DH, DW), np.uint8)
                cv2.drawContours(cm, [cnt], -1, 1, -1)
                cm_area = max(1, int(cm.sum()))
                inter = int((cm & taken).sum())
                _dbg = _os.environ.get("VH_DIAG_DEBUG")
                if _dbg:
                    print(f"[diag] fam{fam_k} cand sf={sf:.0f} px={cm_area} ov={inter/cm_area:.2f}", flush=True)
                patch_hosts = None
                if inter > 0.2 * cm_area:
                    # PATCH-ON-HOST (26-235's renovation overlays): a dense single-family
                    # hatch region drawn ON a much larger measured piece is the drawing
                    # SAYING "this sub-area is a different material" — the hatch boundary
                    # IS his wall. Claim it only when EVERY overlapping piece is >=3x the
                    # candidate (true host, not a sibling double-read); the hosts get
                    # shaved by the stolen fraction so the page total stays honest.
                    if roof_page_d:
                        continue        # layered roof convention: crickets ride ON the
                                        # membrane, both are paid — never steal territory
                    hosts = []
                    ok_patch = True
                    for pidx_ in range(len(polys)):
                        try:
                            _pm_buf[:] = 0
                            cnt0 = np.array([[int(x * W * sx), int(y * H * sy)]
                                             for x, y in polys[pidx_]["points"]], np.int32)
                            cv2.fillPoly(_pm_buf, [cnt0], 1)
                            ov0 = int((cm & _pm_buf).sum())
                            if ov0 > 0.05 * cm_area:
                                pa0 = int(_pm_buf.sum())
                                if _dbg:
                                    print(f"[diag]   vs piece[{pidx_}] '{(polys[pidx_].get('material') or '?')[:22]}' "
                                          f"sf={polys[pidx_].get('area_sf')} px={pa0} ratio={pa0/cm_area:.1f}", flush=True)
                                if pa0 >= 3 * cm_area:
                                    hosts.append((pidx_, ov0, pa0))
                                elif ov0 >= 0.5 * pa0:
                                    # the candidate covers MOST of this piece: it is a
                                    # backing/partial read of the same zone (26-235's
                                    # 518sf gray fill under his 275 patch) — it yields
                                    # at the piece-level dedup, it does not block.
                                    hosts.append((pidx_, ov0, pa0))
                                else:
                                    ok_patch = False
                                    break
                        except Exception:
                            pass
                    if not ok_patch or not hosts:
                        if _dbg:
                            print(f"[diag]   REJECT: ok_patch={ok_patch} hosts={len(hosts)}", flush=True)
                        continue        # comparable-size overlap = double-read, reject
                    patch_hosts = hosts
                # THE PURITY GATE: >=70% of the diagonal ink inside this region must be
                # this one angle family. True hatch is parallel; text/arrow zones carry
                # every angle and can never pass.
                x0r, y0r = bx / sx, by / sy
                x1r, y1r = (bx + bw) / sx, (by + bh) / sy
                fam_ink = tot_ink = 0.0
                for (xa, ya, xb, yb, ang2, L2) in diag_all:
                    mx, my = (xa + xb) / 2, (ya + yb) / 2
                    if x0r <= mx <= x1r and y0r <= my <= y1r and \
                       cm[min(DH - 1, max(0, int(my * sy))), min(DW - 1, max(0, int(mx * sx)))]:
                        tot_ink += L2
                        if int(ang2 // 8) == fam_k:
                            fam_ink += L2
                if tot_ink < 300 or fam_ink < 0.7 * tot_ink:
                    if _dbg:
                        print(f"[diag]   PURITY kill: tot_ink={tot_ink:.0f} fam_frac={fam_ink/max(tot_ink,1):.2f}", flush=True)
                    continue
                eps = 0.008 * max(bw, bh)
                ap = cv2.approxPolyDP(cnt, max(2.0, eps), True)
                if len(ap) < 3:
                    continue
                disp_pts = [(float(p[0][0]) / sx, float(p[0][1]) / sy) for p in ap]
                holes_norm = []
                try:
                    op_rects, op_pt2 = find_openings(geo, disp_pts, ft_pt)
                    for (ox0, oy0, ox1, oy1) in op_rects:
                        holes_norm.append([[round(ox0 / W, 5), round(oy0 / H, 5)],
                                           [round(ox1 / W, 5), round(oy0 / H, 5)],
                                           [round(ox1 / W, 5), round(oy1 / H, 5)],
                                           [round(ox0 / W, 5), round(oy1 / H, 5)]])
                    if op_pt2 > 0:
                        area_pt2 = max(0.0, area_pt2 - op_pt2)
                        sf = area_pt2 * ft_pt * ft_pt
                except Exception:
                    pass
                if sf < MIN_SF:
                    continue
                norm = [[round(x / W, 5), round(y / H, 5)] for (x, y) in disp_pts]
                cx = round(sum(p[0] for p in norm) / len(norm), 5)
                cy = round(sum(p[1] for p in norm) / len(norm), 5)
                rec = {"points": norm, "area_sf": round(sf, 1), "cx": cx, "cy": cy,
                       "fill_color": [0.85, 0.45, 0.25], "source": "vector",
                       "material": "Hatched area (masonry/EIFS)",
                       "category": "Hatched area (masonry/EIFS)",
                       "group": "Hatched area (masonry/EIFS)", "sf_exact": True,
                       "holes": holes_norm[:24],
                       "label": f"~{round(sf):,} SF"}
                if patch_hosts:
                    # NO host shave here — downstream split/trim recompute host SF
                    # geometrically (a shave would be erased, or double-charged on the
                    # no-split path). Territory accounting happens ONCE, at the piece-
                    # level dedup: patches take priority there and hosts get shaved by
                    # the actual final overlap.
                    rec["patch"] = True
                    taken |= cm         # patched territory is spoken for
                polys.append(rec)
    except Exception:
        pass

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
            if p.get("patch"):
                # patch-on-host piece: its host was ALREADY shaved by exactly this
                # overlap when the patch claimed it — dropping or re-shaving here
                # would double-charge the same territory.
                covered |= m
                keep.append(p)
                continue
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
    polys = [p for p in polys if p["area_sf"] >= SMALL_MIN_SF]
    # pattern fingerprint per piece — the weld may only join pieces whose PATTERNS match
    # (the estimator's rule: fill the material, and materials change where patterns change)
    for p in polys:
        try:
            disp_pts = [(x * W, y * H) for x, y in p["points"]]
            p["psig"] = piece_signature(geo.get("segs") or [], disp_pts, W, H)
        except Exception:
            p["psig"] = "plain"
    # SMALL-FACE DISCRIMINATOR (the safe version of "lower MIN_SF" — that regressed once):
    # a face under 100 SF is kept ONLY when its pattern matches a BIG face's group AND its
    # bbox sits within SMALL_ADOPT_GAP of that big face DIRECTLY. Junk specks have no big
    # same-pattern neighbor; real small wall pieces (a return, a knee wall) always do.
    def _bbox(p):
        xs = [x for x, _ in p["points"]]; ys = [y for _, y in p["points"]]
        return (min(xs), min(ys), max(xs), max(ys))
    big = [p for p in polys if p["area_sf"] >= MIN_SF]
    kept_small = []
    for p in polys:
        if p["area_sf"] >= MIN_SF:
            continue
        grp = (p.get("group") or p.get("material") or "")
        # the group name of a TRAIN region encodes its measured pattern ("Vertical panel -
        # 2.0' seams") — that IS the fingerprint. The generic gray-fill group is where the
        # sprawl junk lived (title-block boxes, specks) — those stay dropped.
        if grp.startswith("Panel wall"):
            continue
        b0 = _bbox(p)
        for q in big:
            if (q.get("group") or q.get("material")) != grp:
                continue
            b1 = _bbox(q)
            gx = max(0.0, max(b0[0], b1[0]) - min(b0[2], b1[2]))
            gy = max(0.0, max(b0[1], b1[1]) - min(b0[3], b1[3]))
            if gx <= SMALL_ADOPT_GAP and gy <= SMALL_ADOPT_GAP:
                kept_small.append(p)
                break
    polys = (big + kept_small)[:MAX_REGIONS]
    polys = weld_faces(polys)   # FIRST IMPRESSION = clean faces: same-pattern pieces welded into one
    # FIRST PAINT = HIS TAKEOFF: split each welded face at structural joints (the same
    # senior-estimator reader the bucket uses) so the auto preview shows one shape PER
    # WALL, not one blob per band. MONEY-SAFE: ratio mode — each face's pieces carry
    # exactly the face's SF (sum preserved to the tenth), so page totals cannot move.
    try:
        import snap_fill as _sf
        doc2 = fitz.open(stream=pdf_bytes, filetype="pdf")
        pg2 = doc2[page_index]
        out_split = []
        for p in polys:
            if len(out_split) >= MAX_REGIONS - 4:
                out_split.append(p)      # region budget guard: keep remaining faces whole
                continue
            is_train = not ((p.get("group") or p.get("material") or "").startswith("Panel wall"))
            sp = None
            try:
                # GEOMETRIC-NET piece SF when the scale is trusted (his 664 band: ratio
                # mode gave 371 for a fully-covered wall — parents' strip-integral nets
                # spread unevenly). The piece-level dedup below keeps totals honest.
                sp = _sf.split_face_at_joints(pg2, p["points"], p.get("holes") or [],
                                              p.get("area_sf", 0), W, H, None,
                                              ft_pt=(ft_pt if conf else None),
                                              filter_empty=is_train)
            except Exception:
                sp = None
            if not sp or not sp[1] or len(sp[1]) < 2:
                out_split.append(p)
                continue
            for piece in sp[1]:
                q = dict(p)
                q["points"] = piece["points"]
                q["area_sf"] = piece["area_sf"]
                q["holes"] = piece["holes"]
                q["label"] = f"~{round(piece['area_sf']):,} SF"
                xs = [x for x, _ in piece["points"]]; ys = [y for _, y in piece["points"]]
                q["cx"] = round(sum(xs) / len(xs), 5); q["cy"] = round(sum(ys) / len(ys), 5)
                out_split.append(q)
        doc2.close()
        polys = out_split[:MAX_REGIONS]
    except Exception:
        pass
    # PATTERN-EXTENT TRIM (the owner's rule: "the overlay goes over the PATTERN"): the
    # estimator's highlight hugs where the courses/seams actually stop; our pieces can
    # include blank margin past the last course (664/989 bands ran +11-13% over). Clip
    # each pattern piece to its own hairline-ink extent (+4pt). TRIM-ONLY — the SF can
    # shrink toward the drawn pattern, never grow.
    if conf:
        try:
            from shapely.geometry import Polygon as _P2, Point as _Pt2, box as _box
            hair_v, hair_h = [], []
            all_v, all_h = [], []        # ANY weight — door jambs are drawn heavier than hairlines
            _dtrim = fitz.open(stream=pdf_bytes, filetype="pdf")   # AUDIT FIX: was leaked
            for d in _dtrim[page_index].get_drawings():
                wd = d.get("width") or 0
                col = d.get("color")
                if col is not None and len(col) >= 3 and sum(col[:3]) / 3 >= 0.66:
                    continue
                for it in d.get("items") or []:
                    if it[0] != "l":
                        continue
                    p0 = fitz.Point(it[1]) * rot; p1 = fitz.Point(it[2]) * rot
                    dx, dy = abs(p1.x - p0.x), abs(p1.y - p0.y)
                    if dy >= 6 and dx <= 1.5:
                        all_v.append(((p0.x + p1.x) / 2, min(p0.y, p1.y), max(p0.y, p1.y)))
                        if wd < 0.7:
                            hair_v.append(all_v[-1])
                    elif dx >= 6 and dy <= 1.5:
                        all_h.append(((p0.y + p1.y) / 2, min(p0.x, p1.x), max(p0.x, p1.x)))
                        if wd < 0.7:
                            hair_h.append(all_h[-1])
            _dtrim.close()
            for p in polys:
                grp = (p.get("group") or p.get("material") or "")
                if grp.startswith("Panel wall"):
                    continue             # plain fills have no pattern to trim to
                try:
                    poly = _P2([(x * W, y * H) for x, y in p["points"]]).buffer(0)
                    if poly.is_empty:
                        continue
                    xs, ys = [], []
                    for (x, lo, hi) in hair_v:
                        if poly.contains(_Pt2(x, (lo + hi) / 2)):
                            xs.append(x); ys += [lo, hi]
                    for (y, lo, hi) in hair_h:
                        if poly.contains(_Pt2((lo + hi) / 2, y)):
                            ys.append(y); xs += [lo, hi]
                    if len(xs) < 4 or len(ys) < 4:
                        continue
                    ext = _box(min(xs) - 4, min(ys) - 4, max(xs) + 4, max(ys) + 4)
                    clipped = poly.intersection(ext)
                    if clipped.geom_type == "MultiPolygon":
                        clipped = max(clipped.geoms, key=lambda g: g.area)
                    if clipped.geom_type == "Polygon" and not clipped.is_empty and \
                       0.5 * poly.area <= clipped.area < 0.995 * poly.area:
                        ring = list(clipped.exterior.coords)[:-1]
                        p["points"] = [[round(px / W, 5), round(py / H, 5)] for px, py in ring]
                        poly = clipped
                    # PATTERN-INTERRUPTION OPENINGS (how SHE deducts): courses/seams STOP
                    # at a door — an aligned gap recurring across ≥5 pattern lines is an
                    # opening even with no drawn frame (garage/man doors, flattened sets).
                    gap_holes, gap_pt2 = _pattern_gap_openings(
                        poly, hair_v, hair_h, p.get("holes") or [], W, H, ft_pt,
                        jamb_v=all_v, jamb_h=all_h)
                    if gap_holes:
                        p["holes"] = (p.get("holes") or []) + gap_holes
                    hole_pt2 = 0.0
                    for h in (p.get("holes") or []):
                        try:
                            hole_pt2 += abs(_P2([(qq[0] * W, qq[1] * H) for qq in h]).area)
                        except Exception:
                            pass
                    p["area_sf"] = round(max(0.0, poly.area - hole_pt2) * ft_pt * ft_pt, 1)
                    p["label"] = f"~{round(p['area_sf']):,} SF"
                    # TRUST = SHOW THE ARITHMETIC: every wall carries its own math so
                    # the estimator can verify any number in seconds, like checking a
                    # colleague's takeoff. gross (drawn geometry) − openings = net.
                    bx0, by0, bx1, by1 = poly.bounds
                    p["sf_calc"] = {
                        "gross_sf": round(poly.area * ft_pt * ft_pt, 1),
                        "openings_sf": round(hole_pt2 * ft_pt * ft_pt, 1),
                        "net_sf": p["area_sf"],
                        "n_openings": len(p.get("holes") or []),
                        "w_ft": round((bx1 - bx0) * ft_pt, 1),
                        "h_ft": round((by1 - by0) * ft_pt, 1),
                        "basis": "drawing geometry @ 1\"=%g'" % round(ft_pt * 72, 2),
                    }
                except Exception:
                    pass
        except Exception:
            pass
    # MATERIAL-TAG SPLIT (benchmark finding, 26-088: side-1|mt-5 walls are both drawn as
    # 3in courses — geometry CANNOT split same-pattern different-material walls, but the
    # architect TAGS each wall with its finish code and the estimator names walls BY those
    # tags). Read short codes (MT-5, SIDE-1, B-1...) inside each face; ≥2 distinct tag
    # territories → split at midlines snapped to structural lines, name pieces by tag.
    # Flattened sets (Fleet) have no text → automatic no-op → canary safe.
    try:
        polys = _tag_split(pdf_bytes, page_index, polys, snapx, W, H)
    except Exception:
        pass
    # PIECE-LEVEL overlap dedup: with geometric-net piece SF, overlapping pieces from
    # two parent faces would double-count — same honest shave as the face-level pass
    # (biggest/fill first; drop >55% covered; shave partial overlaps' SF).
    try:
        import numpy as np, cv2
        MW = 900
        MH = max(1, int(MW * H / max(1, W)))
        # PATCH pieces (hatch overlays accepted on a host — 26-235's MP-1) go FIRST:
        # the drawn hatch boundary is the drawing's own statement of a sub-area wall,
        # so the patch claims its territory and the host pieces get shaved by the
        # actual overlap here (the ONE place territory is charged).
        order = sorted(range(len(polys)), key=lambda i: (
            0 if polys[i].get("patch") else
            (1 if (polys[i].get("material") or "").startswith("Panel wall") else 2),
            -polys[i]["area_sf"]))
        covered = np.zeros((MH, MW), np.uint8)
        patch_cov = None                 # territory claimed by patch pieces
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
            if ov > 0.9:
                continue                 # true duplicate
            if patch_cov is not None and not p.get("patch"):
                grp0 = (p.get("group") or p.get("material") or "")
                if (grp0.startswith("Panel wall") or grp0.startswith("Wall area")) and \
                   int((m & patch_cov).sum()) >= 0.5 * a:
                    continue             # generic backing mostly under a patch: yields
            if ov > 0.05:
                # shave the SF, keep the piece — dropping cost a matched wall cross-job
                p = dict(p); p["area_sf"] = round(p["area_sf"] * (1 - ov), 1)
                p["label"] = f"~{round(p['area_sf']):,} SF"
            covered |= m
            if p.get("patch"):
                if patch_cov is None:
                    patch_cov = np.zeros((MH, MW), np.uint8)
                patch_cov |= m
            p["_mid_dedup"] = True       # early pieces: the final pass must never re-shave
            keep.append(p)
        polys = keep
    except Exception:
        pass
    # COLOR-FILL READER (Avita's real language, seen in the gold overlay: siding drawn
    # as SATURATED COLOR FILLS — teal=SFC-1, lavender=SFC-2; her rectangles hug those
    # areas). Gray fills were always read; colored ones were ignored. Rasterize fill
    # paths per color → connected regions = walls, grouped by color (bucket siblings).
    if conf:
        try:
            import numpy as np, cv2
            from collections import defaultdict
            dcf = fitz.open(stream=pdf_bytes, filetype="pdf")
            pgc = dcf[page_index]
            rotc = pgc.rotation_matrix
            CW = 1200
            CH = max(1, int(CW * H / max(1, W)))
            sxc, syc = CW / W, CH / H
            by_col = {}
            chips = {}                                # color key -> legend-chip display pos
            for d in pgc.get_drawings():
                f = d.get("fill")
                if not f or len(f) < 3:
                    continue
                if max(f[:3]) - min(f[:3]) < 0.10:
                    continue                          # gray/white/black — not a color fill
                key = "%.2f,%.2f,%.2f" % (f[0], f[1], f[2])
                if key not in by_col:
                    by_col[key] = (np.zeros((CH, CW), np.uint8), list(f[:3]))
                m = by_col[key][0]
                for it in d.get("items") or []:
                    if it[0] == "re":
                        r = it[1]
                        # tiny color rect = a LEGEND CHIP — the sheet's own color->material map
                        if (r.x1 - r.x0) < 24 and (r.y1 - r.y0) < 24 and key not in chips:
                            cpt = fitz.Point((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2) * rotc
                            chips[key] = (cpt.x, cpt.y)
                        cs = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)]
                        pp = [fitz.Point(X, Y) * rotc for X, Y in cs]
                        cv2.fillPoly(m, [np.array([[int(p.x * sxc), int(p.y * syc)] for p in pp], np.int32)], 1)
                cur = []
                for it in d.get("items") or []:
                    if it[0] == "l":
                        p1 = fitz.Point(it[1]) * rotc
                        cur.append((p1.x, p1.y))
                if len(cur) >= 3:
                    cv2.fillPoly(m, [np.array([[int(x * sxc), int(y * syc)] for x, y in cur], np.int32)], 1)
            # read the legend row to the right of each chip = HER material name
            col_names = {}
            try:
                wlist = []
                for w in pgc.get_text("words"):
                    wp = fitz.Point((w[0] + w[2]) / 2, (w[1] + w[3]) / 2) * rotc
                    wlist.append((wp.x, wp.y, (w[4] or "").strip()))
                for key, (cx_, cy_) in chips.items():
                    row = sorted([(x, t) for (x, y, t) in wlist
                                  if abs(y - cy_) < 9 and 0 < x - cx_ < 420 and t], key=lambda q: q[0])
                    nm = " ".join(t for _, t in row)[:64].strip()
                    if len(nm) >= 4:
                        col_names[key] = nm
            except Exception:
                pass
            dcf.close()
            for key, (m, rgb) in by_col.items():
                cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in cnts:
                    a_pt2 = cv2.contourArea(c) / (sxc * syc)
                    sfc_ = a_pt2 * ft_pt * ft_pt
                    if sfc_ < 40 or sfc_ > 20000:
                        continue                      # legend chips are tiny; junk giants out
                    eps = 0.006 * cv2.arcLength(c, True)
                    ap = cv2.approxPolyDP(c, max(1.5, eps), True).reshape(-1, 2)
                    if len(ap) < 3:
                        continue
                    normc = [[round(float(x) / sxc / W, 5), round(float(y) / syc / H, 5)] for x, y in ap]
                    # virgin territory: never double-count a wall another reader measured —
                    # EXCEPT anonymous junk floods ('Wall area (confirm)'): a drawn color
                    # fill is the sheet's own statement of the material extent and beats
                    # an unnamed flood (26-179: datum-named floods blocked the keynote
                    # bands the reader had already built). Same rule the rc reader ships.
                    nx0 = min(q[0] for q in normc); nx1 = max(q[0] for q in normc)
                    ny0 = min(q[1] for q in normc); ny1 = max(q[1] for q in normc)
                    clash = False
                    repl_c = []
                    ca = max(1e-9, (nx1 - nx0) * (ny1 - ny0))
                    for pex in polys:
                        exs = [q[0] for q in pex["points"]]; eys = [q[1] for q in pex["points"]]
                        ex0, ex1 = min(exs), max(exs); ey0, ey1 = min(eys), max(eys)
                        ix = min(nx1, ex1) - max(nx0, ex0)
                        iy = min(ny1, ey1) - max(ny0, ey0)
                        if ix <= 0 or iy <= 0 or ix * iy <= 0.3 * ca:
                            continue
                        mat_e = pex.get("material") or ""
                        ea = max(1e-9, (ex1 - ex0) * (ey1 - ey0))
                        mutual = (ix * iy > 0.7 * ca) and (ix * iy > 0.7 * ea)
                        if mat_e == "Wall area (confirm)" or \
                           (mat_e.startswith("Panel wall") and mutual):
                            # the gray fill is the BACKING of this colored band (mutual
                            # cover) — the color IS the material statement, gray yields.
                            # Gray fills without a color twin (Fleet) are never touched.
                            repl_c.append(pex)
                        else:
                            clash = True
                            break
                    if clash:
                        continue
                    for pex in repl_c:
                        try:
                            polys.remove(pex)
                        except ValueError:
                            pass
                    cxc = round(sum(q[0] for q in normc) / len(normc), 5)
                    cyc = round(sum(q[1] for q in normc) / len(normc), 5)
                    cname = col_names.get(key) or f"Color fill ({key})"
                    polys.append({"points": normc, "area_sf": round(sfc_, 1), "cx": cxc, "cy": cyc,
                                  "fill_color": rgb, "source": "vector",
                                  "material": cname, "category": cname,
                                  "group": cname, "sf_exact": True, "_colorpiece": True,
                                  "holes": [], "named_by_tag": bool(col_names.get(key)),
                                  "label": f"~{round(sfc_):,} SF"})
        except Exception:
            pass
        # HER WALLS ARE PER-STORY: split color pieces at structural joints (the same
        # senior-estimator splitter) — a two-story teal area becomes two walls, geometric SF
        try:
            import snap_fill as _sf4
            d3 = fitz.open(stream=pdf_bytes, filetype="pdf")
            pg3 = d3[page_index]
            out_c = []
            for p in polys:
                if not p.pop("_colorpiece", False):
                    out_c.append(p)
                    continue
                try:
                    sp = _sf4.split_face_at_joints(pg3, p["points"], [], p.get("area_sf", 0),
                                                   W, H, None, ft_pt=ft_pt, filter_empty=False)
                except Exception:
                    sp = None
                if not sp or not sp[1] or len(sp[1]) < 2:
                    out_c.append(p)
                    continue
                for piece in sp[1]:
                    q = dict(p)
                    q["points"] = piece["points"]
                    q["area_sf"] = piece["area_sf"]
                    q["label"] = f"~{round(piece['area_sf']):,} SF"
                    xs4 = [x for x, _ in piece["points"]]; ys4 = [y for _, y in piece["points"]]
                    q["cx"] = round(sum(xs4) / len(xs4), 5); q["cy"] = round(sum(ys4) / len(ys4), 5)
                    out_c.append(q)
            d3.close()
            polys = out_c
        except Exception:
            pass
    # BAND DETECTOR v2 — PER VIEW (v1 lesson: story lines stitch across separate views).
    # Within one drawing view: the zone between consecutive story lines, between outline
    # verticals, CONTAINING a window (junk gate) and not already covered = a wall band.
    if conf:
        try:
            import snap_fill as _sf5
            db = fitz.open(stream=pdf_bytes, filetype="pdf")
            pgb = db[page_index]
            vsh, hsh, _vh5, _hh5, _o5 = _sf5._heavy_lines(pgb)
            db.close()
            win_rects = [(r[0], r[1], r[2], r[3]) for r in (geo.get("rects") or [])
                         if 8 <= (r[2] - r[0]) <= 200 and 8 <= (r[3] - r[1]) <= 200]
            def _covered_b(x0b, y0b, x1b, y1b):
                for pex in polys:
                    exs = [q[0] * W for q in pex["points"]]; eys = [q[1] * H for q in pex["points"]]
                    ix = min(x1b, max(exs)) - max(x0b, min(exs))
                    iy = min(y1b, max(eys)) - max(y0b, min(eys))
                    if ix > 0 and iy > 0 and ix * iy > 0.25 * (x1b - x0b) * (y1b - y0b):
                        return True
                return False
            added_b = 0
            for (vx0, vy0, vx1, vy1) in view_boxes(geo, W, H):
                vw, vh_ = vx1 - vx0, vy1 - vy0
                if vw < 120 or vh_ < 80 or vx0 > 0.82 * W:
                    continue             # skip note blocks and the title-block strip
                # residential story lines are drawn LIGHT (trim boards) — admit light
                # stitched horizontals as story candidates too (window+covered gates and
                # the 3-16ft band height filter keep lap courses and junk out)
                vhs = [(c, lo, hi, wd) for (c, lo, hi, wd) in hsh if vy0 <= c <= vy1]
                vhs += [(c, lo, hi, 0.9) for (c, lo, hi) in _hh5
                        if vy0 <= c <= vy1 and (hi - lo) >= 0.35 * vw]
                vvs = [(c, lo, hi, wd) for (c, lo, hi, wd) in vsh if vx0 <= c <= vx1]
                vvs += [(c, lo, hi, 0.9) for (c, lo, hi) in _vh5
                        if vx0 <= c <= vx1 and (hi - lo) >= 0.30 * vh_]
                story = sorted(y for (y, sp, wd, cov) in
                               _sf5._joint_positions(vhs, vx0, vx1, 0.35 * vw))
                outl = sorted(x for (x, sp, wd, cov) in
                              _sf5._joint_positions(vvs, vy0, vy1, 0.35 * vh_))
                for ya, yb in zip(story, story[1:]):
                    bh = yb - ya
                    if not (3.0 <= bh * ft_pt <= 16.0):
                        continue
                    for xa, xb in zip(outl, outl[1:]):
                        bw = xb - xa
                        if bw * ft_pt < 6.0 or added_b >= 12:
                            continue
                        wins = [w for w in win_rects
                                if xa < (w[0] + w[2]) / 2 < xb and ya < (w[1] + w[3]) / 2 < yb]
                        # ALTERNATE CORROBORATION (26-191A off-by-one class): a band with
                        # NO windows still counts when it is DENSE with course lines
                        # (his 684sf upper band: ~95 stitched light rows; junk gaps have none)
                        dense_b = sum(1 for (c5, lo5, hi5) in _hh5
                                      if ya < c5 < yb and min(hi5, xb) - max(lo5, xa)
                                      >= 0.5 * (xb - xa)) >= 6
                        if (not wins and not dense_b) or _covered_b(xa, ya, xb, yb):
                            continue
                        sfb = bw * bh * ft_pt * ft_pt
                        if not (40 <= sfb <= 2500):
                            continue
                        normb = [[round(xa / W, 5), round(ya / H, 5)], [round(xb / W, 5), round(ya / H, 5)],
                                 [round(xb / W, 5), round(yb / H, 5)], [round(xa / W, 5), round(yb / H, 5)]]
                        polys.append({"points": normb, "area_sf": round(sfb, 1),
                                      "cx": round((xa + xb) / 2 / W, 5), "cy": round((ya + yb) / 2 / H, 5),
                                      "fill_color": [0.55, 0.75, 0.95], "source": "vector",
                                      "material": "Wall band (confirm)", "category": "Wall band (confirm)",
                                      "group": "Wall band (confirm)", "sf_exact": True,
                                      "holes": [], "label": f"~{round(sfb):,} SF"})
                        added_b += 1
        except Exception:
            pass
    # TAG-SEEDED AUTO-BUCKET (Avita convention: siding bands drawn BLANK, material only
    # a repeated legend tag). Flood from each repeated tag within structural barriers →
    # the labeled band. Virgin territory only; textless pages no-op (Fleet canary safe).
    if conf:
        try:
            import snap_fill as _sf3
            newp = _sf3.tag_seed_fill(pdf_bytes, page_index,
                                      [p["points"] for p in polys],
                                      max_new=max(0, MAX_REGIONS - len(polys)))
            _STOPN = {"FLOOR", "LEGEND", "BLDG", "SET", "OF", "GT", "TYP", "SIM", "REF",
                      "LOWER", "LEVEL", "ROOF", "RIDGE", "MAX", "MIN", "GRADE", "TO",
                      "SCALE", "PLAN", "ELEV", "NOTE", "NOTES", "THE", "OD", "ID", "AND", "FOR", "PER",
                      "SEE", "ALL", "MAY", "TRIM", "FASC", "HORIZ", "VERT", "SIM", "TYPE"}
            import re as _re6
            for p in (newp or []):
                mat6 = (p.get("material") or "").upper()
                # a drafting word seeded the fill — the AREA may be real, the NAME is not.
                # Single-letter+digit tags are SCHEDULE marks (W2=window, S2=storefront,
                # L3=louver, grid bubbles A1/B2 — 26-191A named whole bands "W2"), never
                # material names; real material keys carry 2+ letters (SF-1, MP_1, SS-2, EF1).
                if mat6 in _STOPN or _re6.match(r"^[A-Z][-–_]?\d{1,2}$", mat6) \
                        or _re6.search(r"\b[BT]\.?O\.?\b", mat6):
                    # datum/level phrases ("3 B.O. DECK - LOW", "T.O. WALL") are
                    # elevation markers, never materials (26-179 named whole bands by them)
                    p["material"] = p["category"] = p["group"] = "Wall area (confirm)"
                    p.pop("named_by_tag", None)
            if newp:
                # tag-seeded floods can swallow several walls (26-183: an EP12 flood of
                # 1,218sf contains his 291sf ACM soffit) — run the same structural joint
                # splitter color pieces get, geometric SF per piece
                try:
                    d6 = fitz.open(stream=pdf_bytes, filetype="pdf")
                    pg6 = d6[page_index]
                    split6 = []
                    for p in newp:
                        sp6 = None
                        try:
                            sp6 = _sf3.split_face_at_joints(pg6, p["points"], [],
                                                            p.get("area_sf", 0), W, H, None,
                                                            ft_pt=ft_pt, filter_empty=False)
                        except Exception:
                            sp6 = None
                        if not sp6 or not sp6[1] or len(sp6[1]) < 2:
                            split6.append(p)
                            continue
                        for piece in sp6[1]:
                            q = dict(p)
                            q["points"] = piece["points"]
                            q["area_sf"] = piece["area_sf"]
                            q["label"] = f"~{round(piece['area_sf']):,} SF"
                            xs6 = [x for x, _ in piece["points"]]; ys6 = [y for _, y in piece["points"]]
                            q["cx"] = round(sum(xs6) / len(xs6), 5); q["cy"] = round(sum(ys6) / len(ys6), 5)
                            split6.append(q)
                    d6.close()
                    newp = split6
                except Exception:
                    pass
                polys += newp
        except Exception:
            pass
    # RENDERED-ELEVATION COLOR READER (Avita p4/p5 language: sheets are tiled-JPEG
    # underlays — the siding exists ONLY as image pixels, invisible to get_drawings
    # forever. Segment saturated hue families in the RENDER; her convention measured
    # from her gold: polygon = gross module box, SF = net of window-shaped holes).
    # Gates: page must carry a raster underlay (>=25% image coverage) — pure-vector
    # sheets (Fleet/Danbury) no-op; virgin territory only; scale-confirmed pages only.
    # (Un-gating for unconfirmed photo sheets was TRIED 2026-07-09 and reverted: real
    # photos don't segment into clean hue families — 26-195 stayed 0/19 while adding
    # junk risk. Photo-sheet takeoff = its own future detector class; his convention
    # there = 1/8" default scale, walls traced on the photo itself.)
    if conf:
        try:
            polys += _rendered_color_regions(pdf_bytes, page_index, polys, W, H, ft_pt,
                                             max_new=max(0, MAX_REGIONS - len(polys)))
        except Exception:
            pass
    # EXTEND-TO-GRADE — DISABLED after failing its gates (2026-07-08). The convention is
    # real (his west wall: fill stops at the base drip y1381, he measures to grade
    # y1408) but the west grade line is mostly LIGHT/hidden ink (~200pt heavy of 607
    # needed) so no safe evidence bar fires there, while story-line slivers made 664
    # regress +7→+17%. Next attempt needs: 'no piece BELOW the candidate line' gate +
    # accepting mixed-weight stitched grade ink. Reverted per never-worse.
    if False and conf:
        try:
            import snap_fill as _sf2
            dg = fitz.open(stream=pdf_bytes, filetype="pdf")
            _vsx, _hsx, _vhx, _hhx, _oth = _sf2._heavy_lines(dg[page_index])
            dg.close()
            from shapely.geometry import Polygon as _P4, box as _box4
            all_p4 = []
            for p in polys:
                try:
                    all_p4.append((p, _P4([(x * W, y * H) for x, y in p["points"]]).buffer(0)))
                except Exception:
                    all_p4.append((p, None))
            for p, poly4 in all_p4:
                try:
                    if poly4 is None or poly4.is_empty:
                        continue
                    bx0, by0, bx1, by1 = poly4.bounds
                    pw = bx1 - bx0
                    best = None
                    # ground lines are drawn in segments (broken at bollards/stairs) —
                    # the joint stitcher handles gaps and rejects dashes
                    for (yy, sp, wdp, cov) in _sf2._joint_positions(_hsx, bx0 - 40, bx1 + 40,
                                                                    0.9 * pw):
                        if wdp < 1.0 or not (by1 + 2 < yy <= by1 + 36):
                            continue                       # within ~4ft below the piece
                        if best is None or yy < best:
                            best = yy
                    if best is None:
                        continue
                    gap_box = _box4(bx0 + 0.02 * pw, by1 - 2, bx1 - 0.02 * pw, best)
                    # GRADE has nothing below/inside the strip; a STORY line has the next
                    # wall there. Any other piece in the strip → this is not grade, skip.
                    other_wall = False
                    for p2, q4 in all_p4:
                        if p2 is p or q4 is None or q4.is_empty:
                            continue
                        try:
                            if q4.intersection(gap_box).area > 0.05 * gap_box.area:
                                other_wall = True
                                break
                        except Exception:
                            pass
                    if other_wall:
                        continue
                    ext = poly4.union(gap_box)
                    ext = ext.buffer(0)
                    if ext.geom_type != "Polygon" or ext.area <= poly4.area:
                        continue
                    if ext.area > 1.35 * poly4.area:
                        continue                           # never grow more than 35%
                    ring = list(ext.exterior.coords)[:-1]
                    p["points"] = [[round(px / W, 5), round(py / H, 5)] for px, py in ring]
                    hp4 = 0.0
                    for h in (p.get("holes") or []):
                        try:
                            hp4 += abs(_P4([(q[0] * W, q[1] * H) for q in h]).area)
                        except Exception:
                            pass
                    p["area_sf"] = round(max(0.0, ext.area - hp4) * ft_pt * ft_pt, 1)
                    p["label"] = f"~{round(p['area_sf']):,} SF"
                    p.pop("sf_calc", None)
                except Exception:
                    pass
        except Exception:
            pass
    # every piece carries its arithmetic (gross − openings = net) — trust is showing
    # the math, not asking for belief
    if conf:
        try:
            from shapely.geometry import Polygon as _P3
            for p in polys:
                if p.get("sf_calc"):
                    continue
                try:
                    poly3 = _P3([(x * W, y * H) for x, y in p["points"]]).buffer(0)
                    if poly3.is_empty:
                        continue
                    hp = 0.0
                    for h in (p.get("holes") or []):
                        try:
                            hp += abs(_P3([(q[0] * W, q[1] * H) for q in h]).area)
                        except Exception:
                            pass
                    bx0, by0, bx1, by1 = poly3.bounds
                    p["sf_calc"] = {
                        "gross_sf": round(poly3.area * ft_pt * ft_pt, 1),
                        "openings_sf": round(hp * ft_pt * ft_pt, 1),
                        "net_sf": p.get("area_sf", 0),
                        "n_openings": len(p.get("holes") or []),
                        "w_ft": round((bx1 - bx0) * ft_pt, 1),
                        "h_ft": round((by1 - by0) * ft_pt, 1),
                        "basis": "drawing geometry @ 1\"=%g'" % round(ft_pt * 72, 2),
                    }
                except Exception:
                    pass
        except Exception:
            pass
    # ROOF-PLAN split (vocab-gated to pages speaking real roofing language)
    try:
        polys = _roof_split(pdf_bytes, page_index, polys, W, H, ft_pt)
    except Exception:
        pass
    # PER-VIEW SCALES (owner directive: never measure at a scale the drawing didn't
    # state). Multi-view sheets print each view's own scale under its title (Q1
    # BUILDING ELEVATION / 1/8"=1'-0"; details at 3/4"). Rule measured on Avita p5:
    # a piece rescales ONLY when EVERY scale anchor below it agrees — any ambiguity
    # (a neighboring detail's anchor in range) leaves the piece at page scale with a
    # scale_risk flag instead of silently inventing a number. Perspective views
    # ("3D VIEW") are never measurable — their pieces are dropped.
    try:
        polys = _apply_view_scales(pdf_bytes, page_index, polys, W, H, sc)
    except Exception:
        pass
    # FINAL OVERLAP DEDUP — LATE PIECES ONLY (26-204 corpus finding: the mid-pipeline
    # shave runs before the color/rc/tag readers append, so late pieces could STACK —
    # a 73sf wall collected a flood + two identical color pieces = 172sf). Early
    # pieces (_mid_dedup) are the untouchable baseline: re-shaving them double-counts
    # the shave (first attempt moved the Fleet canary — reverted to this form). Late
    # pieces dedup against the baseline AND each other: >90% covered = duplicate.
    try:
        import numpy as np, cv2
        MW9 = 900
        MH9 = max(1, int(MW9 * H / max(1, W)))
        covered9 = np.zeros((MH9, MW9), np.uint8)
        early9 = [p for p in polys if p.get("_mid_dedup")]
        late9 = [p for p in polys if not p.get("_mid_dedup")]
        for p in early9:
            cnt9 = np.array([[int(x * MW9), int(y * MH9)] for x, y in p["points"]], np.int32)
            cv2.fillPoly(covered9, [cnt9], 1)
        keep9 = list(early9)
        kept_late9 = []                    # (group, bbox) of admitted late pieces
        def _bb9(p):
            xs = [x for x, _ in p["points"]]; ys = [y for _, y in p["points"]]
            return (min(xs), min(ys), max(xs), max(ys))
        for p in sorted(late9, key=lambda q: -q.get("area_sf", 0)):
            cnt9 = np.array([[int(x * MW9), int(y * MH9)] for x, y in p["points"]], np.int32)
            m9 = np.zeros((MH9, MW9), np.uint8)
            cv2.fillPoly(m9, [cnt9], 1)
            a9 = int(m9.sum())
            if a9 == 0:
                continue
            bb = _bb9(p)
            # TRUE DUPLICATE = a TWIN: same group, near-identical bbox (26-204's
            # doubled color fill). A nested module inside its parent band is NOT a
            # duplicate — dropping those lost 2 Avita walls. Twins only.
            twin = False
            for (g9, ob) in kept_late9:
                if g9 != (p.get("group") or ""):
                    continue
                ix = min(bb[2], ob[2]) - max(bb[0], ob[0])
                iy = min(bb[3], ob[3]) - max(bb[1], ob[1])
                if ix <= 0 or iy <= 0:
                    continue
                inter = ix * iy
                union = (bb[2]-bb[0])*(bb[3]-bb[1]) + (ob[2]-ob[0])*(ob[3]-ob[1]) - inter
                if inter / max(union, 1e-12) >= 0.8:
                    twin = True
                    break
            if twin:
                continue
            ov9 = int((m9 & covered9).sum()) / a9
            if 0.25 < ov9 <= 0.9:
                p = dict(p)
                p["area_sf"] = round(p.get("area_sf", 0) * (1 - ov9), 1)
                p["label"] = f"~{round(p['area_sf']):,} SF"
            covered9 |= m9
            kept_late9.append(((p.get("group") or ""), bb))
            keep9.append(p)
        polys = keep9
    except Exception:
        pass
    # JUNK-YIELDS-TO-NAMED (26-204 tail: a 73sf wall collected an anonymous flood PLUS
    # the color piece that owns it — the flood must yield when NAMED color/rendered
    # pieces cover >=60% of it; anonymous area is a guess, a drawn color is a statement)
    try:
        import numpy as np, cv2
        MWj = 700
        MHj = max(1, int(MWj * H / max(1, W)))
        named_m = np.zeros((MHj, MWj), np.uint8)
        n_named = 0
        for p in polys:
            mt = (p.get("material") or "")
            if mt.startswith("Color fill") or mt.startswith("Rendered"):
                cv2.fillPoly(named_m, [np.array([[int(x * MWj), int(y * MHj)]
                                                 for x, y in p["points"]], np.int32)], 1)
                n_named += 1
        if n_named:
            import re as _rej
            def _junky(mt):
                mtu = mt.upper()
                return (mt == "Wall area (confirm)"
                        or bool(_rej.match(r"^[A-Z][-–_]?\d{1,2}$", mtu))
                        or bool(_rej.search(r"\b[BT]\.?O\.?\b", mtu)))
            keepj = []
            for p in polys:
                if _junky(p.get("material") or ""):
                    mj = np.zeros((MHj, MWj), np.uint8)
                    cv2.fillPoly(mj, [np.array([[int(x * MWj), int(y * MHj)]
                                                for x, y in p["points"]], np.int32)], 1)
                    aj = int(mj.sum())
                    if aj and int((mj & named_m).sum()) > 0.6 * aj:
                        continue          # the color reading owns this territory
                keepj.append(p)
            polys = keepj
    except Exception:
        pass
    # SOURCE-AGNOSTIC junk-name pass: whatever reader named a piece, a name that is a
    # schedule mark (W2), a grid bubble (A1), or a datum phrase (3 B.O. DECK - LOW,
    # T.O. WALL) is never a material — keep the area, make the name honest (which also
    # makes the piece replaceable by color/rc readers).
    try:
        import re as _re9
        for p in polys:
            m9 = (p.get("material") or "").upper()
            if _re9.match(r"^[A-Z][-–_]?\d{1,2}$", m9) or _re9.search(r"\b[BT]\.?O\.?\b", m9):
                p["material"] = p["category"] = p["group"] = "Wall area (confirm)"
                p.pop("named_by_tag", None)
    except Exception:
        pass
    for i, p in enumerate(polys):
        p.pop("_mid_dedup", None)
        p["id"] = i
    return polys, W, H, {"ft_per_in": round(sc, 3), "scale_confirmed": conf}


_ROOF_TERMS = ("MEMBRANE", "TPO", "EPDM", "CRICKET", "WALKWAY", "WALKING PAD",
               "STANDING SEAM", "TAPERED INSULATION", "FULLY ADHERED", "BALLAST")


def _roof_split(pdf_bytes, page_index, polys, W, H, ft_pt):
    """ROOF-PLAN mode (26-045/26-183 p6 class): his roof zones = plan areas bounded by
    ridge/valley/edge lines of ANY weight (measured: every membrane edge carries
    378-11,112pt of aligned long ink, wd 1.0-5.0). On pages speaking ROOFING (>=2
    distinct roof terms — 'ROOFING' alone appears in siding legends, not enough),
    split large flood pieces recursively at long straight runs. SF = plan shoelace
    (his convention, proven: implied scale exactly 8.00, NO pitch factor)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        pg = doc[page_index]
        txt = (pg.get_text() or "").upper()
        if sum(1 for t in _ROOF_TERMS if t in txt) < 2:
            return polys
        h_lines, _ = _collect_axis(pg, "h", stitch=8)
        v_lines, _ = _collect_axis(pg, "v", stitch=8)
        rot = pg.rotation_matrix
        # roofing PHRASES (row-assembled) for piece naming — Budget/Excel honesty
        rows9 = {}
        for w in pg.get_text("words"):
            p9 = fitz.Point((w[0] + w[2]) / 2, (w[1] + w[3]) / 2) * rot
            rows9.setdefault(round(p9.y / 6), []).append((p9.x, p9.y, (w[4] or "").strip()))
        phrases = []
        for k9 in sorted(rows9):
            ws9 = sorted(rows9[k9])
            t9 = " ".join(t for _, _, t in ws9)
            if any(term in t9.upper() for term in _ROOF_TERMS):
                xs9 = [x for x, _, _ in ws9]
                phrases.append((sum(xs9) / len(xs9),
                                sum(y for _, y, _ in ws9) / len(ws9), t9[:44]))
    finally:
        doc.close()
    from shapely.geometry import Polygon as _P, LineString as _LS
    from shapely.ops import split as _split_op

    def parts_of(poly, depth):
        if depth >= 4 or poly.area * ft_pt * ft_pt < 800:
            return [poly]
        bx0, by0, bx1, by1 = poly.bounds
        best = None
        for (c, a, b) in h_lines:
            if not (by0 + 12 < c < by1 - 12):
                continue
            ov = min(b, bx1) - max(a, bx0)
            if ov >= 0.6 * (bx1 - bx0):
                d = abs(c - (by0 + by1) / 2)
                if best is None or d < best[0]:
                    best = (d, "h", c)
        for (c, a, b) in v_lines:
            if not (bx0 + 12 < c < bx1 - 12):
                continue
            ov = min(b, by1) - max(a, by0)
            if ov >= 0.6 * (by1 - by0):
                d = abs(c - (bx0 + bx1) / 2)
                if best is None or d < best[0]:
                    best = (d, "v", c)
        if best is None:
            return [poly]
        _, ax, c = best
        cutter = _LS([(bx0 - 5, c), (bx1 + 5, c)]) if ax == "h" else _LS([(c, by0 - 5), (c, by1 + 5)])
        try:
            pieces9 = [g for g in _split_op(poly, cutter).geoms
                       if g.geom_type == "Polygon" and g.area * ft_pt * ft_pt >= 60]
        except Exception:
            return [poly]
        if len(pieces9) < 2:
            return [poly]
        out9 = []
        for g in pieces9:
            out9.extend(parts_of(g, depth + 1))
        return out9

    out = []
    for p in polys:
        mat9 = (p.get("material") or "")
        if p.get("area_sf", 0) < 400 or not (mat9 == "Wall area (confirm)" or p.get("named_by_tag")):
            out.append(p)
            continue
        try:
            poly = _P([(x * W, y * H) for x, y in p["points"]]).buffer(0)
            if poly.is_empty or poly.geom_type != "Polygon":
                out.append(p)
                continue
            parts = parts_of(poly, 0)
        except Exception:
            out.append(p)
            continue
        if len(parts) < 2:
            out.append(p)
            continue
        for g in parts:
            q = dict(p)
            ring = list(g.exterior.coords)[:-1]
            q["points"] = [[round(px / W, 5), round(py / H, 5)] for px, py in ring]
            q["area_sf"] = round(g.area * ft_pt * ft_pt, 1)
            q["label"] = f"~{round(q['area_sf']):,} SF"
            q["holes"] = []
            q.pop("sf_calc", None)
            xs9 = [c[0] for c in ring]; ys9 = [c[1] for c in ring]
            cx9 = sum(xs9) / len(xs9); cy9 = sum(ys9) / len(ys9)
            q["cx"] = round(cx9 / W, 5)
            q["cy"] = round(cy9 / H, 5)
            # name from the nearest roofing phrase (the drawing's own vocabulary)
            if phrases:
                d9, best9 = min(((ph[0] - cx9) ** 2 + (ph[1] - cy9) ** 2, ph[2])
                                for ph in phrases)
                if d9 ** 0.5 <= 0.3 * max(W, H):
                    nm9 = best9.strip() + " (confirm)"
                    q["material"] = q["category"] = q["group"] = nm9
            out.append(q)
    return out


def _apply_view_scales(pdf_bytes, page_index, polys, W, H, page_scale):
    import texture as _tx
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        pg = doc[page_index]
        rot = pg.rotation_matrix
        rows = {}
        for w in pg.get_text("words"):
            p = fitz.Point((w[0] + w[2]) / 2, (w[1] + w[3]) / 2) * rot
            rows.setdefault(round(p.y / 6), []).append((p.x, p.y, (w[4] or "").strip()))
    finally:
        doc.close()
    if not rows:
        return polys
    anchors = []
    for k in sorted(rows):
        ws = sorted(rows[k])
        txt = " ".join(t for _, _, t in ws)
        cands = _tx._parse_scale_text(txt)
        if not cands:
            continue
        best, confd = _tx._consensus(cands)
        if not confd:
            continue
        xs = [x for x, _, _ in ws]
        ay = sum(y for _, y, _ in ws) / len(ws)
        # view title often sits on the row above the scale note — carry it for the
        # perspective test
        above = " ".join(t for _, _, t in rows.get(k - 1, [])) + " " + \
                " ".join(t for _, _, t in rows.get(k - 2, [])) + " " + txt
        anchors.append((sum(xs) / len(xs), ay, best, above.upper()))
    if not anchors:
        return polys
    diff = any(abs(s - page_scale) > 0.01 * page_scale for _, _, s, _ in anchors)
    persp = any(("3D" in t or "PERSPECTIVE" in t or "ISOMETRIC" in t or "AXONOMET" in t)
                for _, _, _, t in anchors)
    if not diff and not persp:
        return polys                       # single-scale sheet, nothing to do
    keep = []
    for p in polys:
        xs = [x * W for x, _ in p["points"]]; ys = [y * H for _, y in p["points"]]
        px0, px1, py1 = min(xs), max(xs), max(ys)
        pw = px1 - px0
        below = [(ay - py1, s, t) for (ax, ay, s, t) in anchors
                 if ay >= py1 - 8 and (px0 - 0.6 * pw - 60) <= ax <= (px1 + 0.6 * pw + 60)
                 and (ay - py1) <= 0.35 * H]
        if not below:
            keep.append(p)
            continue
        below.sort()
        vals = {round(s, 4) for _, s, _ in below}
        if len(vals) > 1:
            p["scale_risk"] = True         # ambiguous view zone — verify, never invent
            keep.append(p)
            continue
        d0, s0, t0 = below[0]
        if "3D" in t0 or "PERSPECTIVE" in t0 or "ISOMETRIC" in t0 or "AXONOMET" in t0:
            continue                       # perspective views are not measurable — drop
        if abs(s0 - page_scale) > 0.01 * page_scale:
            f2 = (s0 / page_scale) ** 2
            p["area_sf"] = round(p["area_sf"] * f2, 1)
            p["label"] = f"~{round(p['area_sf']):,} SF"
            p["view_scale"] = s0
            if p.get("sf_calc"):
                for kk in ("gross_sf", "openings_sf", "net_sf"):
                    if kk in p["sf_calc"]:
                        p["sf_calc"][kk] = round(p["sf_calc"][kk] * f2, 1)
                p["sf_calc"]["basis"] = "view scale 1\"=%g'" % round(s0, 2)
        keep.append(p)
    return keep


def _rendered_color_regions(pdf_bytes, page_index, polys, W, H, ft_pt, max_new=40):
    """Read siding drawn as RENDERED color (tiled-image underlay sheets, Avita profile).
    HSV-mask saturated pixels -> 15-degree hue families -> connected components ->
    white-column module splitting -> piece shape = hole-filled outline (her gross box),
    SF = filled minus WINDOW-SHAPED holes only (her net; trim/text islands are wall).
    Measured against her Avita gold: p3 41/42, p4 47/51, p5 18/20 found."""
    import numpy as np, cv2
    if max_new <= 0:
        return []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        pg = doc[page_index]
        # gate: the sheet must BE a raster underlay — pure-vector CAD no-ops here
        img_area = 0.0
        try:
            for im in pg.get_images(full=True):
                for r in pg.get_image_rects(im[0]):
                    img_area += max(0, r.width) * max(0, r.height)
        except Exception:
            pass
        if img_area < 0.25 * W * H:
            return []
        z = 4000.0 / max(W, H)
        pix = pg.get_pixmap(matrix=fitz.Matrix(z, z))
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif pix.n == 1:
            return []
        img = img[:, :, :3]
    finally:
        doc.close()
    Hp, Wp = img.shape[:2]
    ftpx = ft_pt / z                     # ft per point / px per point = ft per px
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    Hc = hsv[:, :, 0].astype(np.int16)
    S = hsv[:, :, 1].astype(np.float32) / 255.0
    V = hsv[:, :, 2].astype(np.float32) / 255.0
    # V floor 0.28 (was 0.45): woodtone/dark-stained sidings render dark (26-162 lap
    # V med 0.30, S 0.35) — saturation still fences out grays/shadows/linework
    colored = ((S >= 0.10) & (S <= 0.75) & (V >= 0.28) & (V <= 0.99)).astype(np.uint8)
    # GRAY value-band families (Regdate giant class: FC-2A panels are deliberate
    # mid-gray S=0.00 V 0.44-0.66 — 35 of p34's 59 unfound walls). Saturation-zero
    # pixels banded by VALUE; paper-white (V>0.85) and linework-black stay excluded.
    # GRAY FAMILY v3 — EVIDENCE-BASED (debug dump 2026-07-12, Regdate p34 + Avita p3):
    # true gray walls are SMALL modules 26-200sf at V-median 0.5-0.73; junk is DARK
    # (V<=0.46 — all 15 of Avita p3's junk grays), BIG (447-1779sf roofs), or THIN
    # lines (aspect>50). One mask, V 0.47-0.80; per-component gates below (size cap,
    # aspect); gray pieces get NO replacement power (v1's eviction cost Avita walls).
    fams = [("h", h0, ((colored > 0) & (Hc >= h0) & (Hc < h0 + 15)))
            for h0 in range(0, 180, 15)]
    fams.append(("g", 0, ((S < 0.08) & (V >= 0.47) & (V <= 0.80))))
    del hsv, S, V
    out = []
    for _fk, h0, fam_b in fams:
        if len(out) >= max_new:
            break
        fam = fam_b.astype(np.uint8)
        if int(fam.sum()) < 0.0004 * Hp * Wp:
            continue
        # representative color of the family (for the UI fill)
        ys_f, xs_f = np.where(fam > 0)
        samp = img[ys_f[::max(1, len(ys_f) // 500)], xs_f[::max(1, len(xs_f) // 500)]]
        rgb = [round(float(c) / 255.0, 3) for c in samp.mean(axis=0)] if len(samp) else [0.6, 0.6, 0.4]
        m = cv2.morphologyEx(fam, cv2.MORPH_CLOSE,
                             np.ones((7, 7) if _fk == "g" else (3, 3), np.uint8))
        ncc, lbl, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for ci in range(1, ncc):
            if len(out) >= max_new:
                break
            a = stats[ci, cv2.CC_STAT_AREA]
            if not (25 <= a * ftpx * ftpx <= 3500):
                continue
            x, y, w, h = stats[ci, 0], stats[ci, 1], stats[ci, 2], stats[ci, 3]
            if _fk == "g":
                # gray evidence gates: thin lines are borders/leaders; the size test
                # moved to the hole test below (walls have window holes, roofs don't)
                if max(w, h) > 20 * max(1, min(w, h)):
                    continue
            # title-block / margin strip: not building
            if x / Wp > 0.86 or y / Hp > 0.92:
                continue
            comp = (lbl[y:y + h, x:x + w] == ci).astype(np.uint8)
            subs = [comp]
            if w > h * 1.6:
                # split merged modules at white trim columns (her walls end at trim boards)
                colf = comp.sum(axis=0) / float(h)
                gaps = colf < 0.04
                cuts = []
                i = 0
                while i < w:
                    if gaps[i]:
                        j = i
                        while j < w and gaps[j]:
                            j += 1
                        if j - i >= 4 and i > 6 and w - j > 6:
                            cuts.append((i, j))
                        i = j
                    else:
                        i += 1
                if cuts:
                    subs = []
                    prev = 0
                    for (i, j) in cuts:
                        seg = np.zeros_like(comp); seg[:, prev:i] = comp[:, prev:i]
                        if seg.sum():
                            subs.append(seg)
                        prev = j
                    seg = np.zeros_like(comp); seg[:, prev:] = comp[:, prev:]
                    if seg.sum():
                        subs.append(seg)
            for sub in subs:
                if len(out) >= max_new:
                    break
                if float(sub.sum()) * ftpx * ftpx < 25:
                    continue
                ff = sub.copy()
                msk = np.zeros((h + 2, w + 2), np.uint8)
                cv2.floodFill(ff, msk, (0, 0), 2)
                filled = (sub | ((ff != 2) & (sub == 0)).astype(np.uint8))
                holes_m = (filled & (sub == 0)).astype(np.uint8)
                nh, hl, hstats, _ = cv2.connectedComponentsWithStats(holes_m, connectivity=8)
                ded = 0.0
                holes_norm = []
                for hi in range(1, nh):
                    ha = hstats[hi, cv2.CC_STAT_AREA]
                    hw, hh2 = hstats[hi, 2], hstats[hi, 3]
                    hsf = ha * ftpx * ftpx
                    if hsf >= 5 and ha >= 0.35 * hw * hh2 and 0.15 <= hw / max(1, hh2) <= 6:
                        ded += hsf
                        hx, hy = hstats[hi, 0] + x, hstats[hi, 1] + y
                        holes_norm.append([[round(hx / Wp, 5), round(hy / Hp, 5)],
                                           [round((hx + hw) / Wp, 5), round(hy / Hp, 5)],
                                           [round((hx + hw) / Wp, 5), round((hy + hh2) / Hp, 5)],
                                           [round(hx / Wp, 5), round((hy + hh2) / Hp, 5)]])
                gross = float(filled.sum()) * ftpx * ftpx
                sf = gross - ded
                if not (35 <= sf <= 2500):
                    continue
                if _fk == "g" and sf > 300 and len(holes_norm) < 2:
                    continue   # big gray with no window holes = ROOF, not wall
                cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cnts:
                    continue
                c = max(cnts, key=cv2.contourArea)
                ap = cv2.approxPolyDP(c, 0.008 * cv2.arcLength(c, True), True).reshape(-1, 2)
                if len(ap) < 3:
                    continue
                norm = [[round((float(px) + x) / Wp, 5), round((float(py) + y) / Hp, 5)] for px, py in ap]
                # virgin territory: never re-measure a wall another reader owns — EXCEPT
                # anonymous junk floods ('Wall area (confirm)' = tag seeds whose name was
                # a drafting word): a color-measured module with her conventions beats an
                # anonymous flood, so the rc piece REPLACES those (never named/color/train
                # /soffit pieces).
                nx0 = min(q[0] for q in norm); nx1 = max(q[0] for q in norm)
                ny0 = min(q[1] for q in norm); ny1 = max(q[1] for q in norm)
                clash = False
                replaceable = []
                for pex in polys + out:
                    exs = [q[0] for q in pex["points"]]; eys = [q[1] for q in pex["points"]]
                    ix = min(nx1, max(exs)) - max(nx0, min(exs))
                    iy = min(ny1, max(eys)) - max(ny0, min(eys))
                    if ix > 0 and iy > 0 and ix * iy > 0.3 * max(1e-9, (nx1 - nx0) * (ny1 - ny0)):
                        if _fk == "h" and (pex.get("material") == "Wall area (confirm)") \
                                and pex not in out:
                            replaceable.append(pex)   # gray pieces never evict (v1 lesson)
                        else:
                            clash = True
                            break
                if clash:
                    continue
                for pex in replaceable:
                    try:
                        polys.remove(pex)
                    except ValueError:
                        pass
                gname = (f"Rendered color {h0}-{h0+15}deg" if _fk == "h"
                         else f"Rendered gray V{h0}-{h0+14}")
                out.append({"points": norm, "area_sf": round(sf, 1),
                            "cx": round((nx0 + nx1) / 2, 5), "cy": round((ny0 + ny1) / 2, 5),
                            "fill_color": rgb, "source": "vector",
                            "material": "Rendered siding (confirm)",
                            "category": "Rendered siding (confirm)",
                            "group": gname, "sf_exact": True,
                            "holes": holes_norm,
                            "label": f"~{round(sf):,} SF",
                            "sf_calc": {"gross_sf": round(gross, 1),
                                        "openings_sf": round(ded, 1),
                                        "net_sf": round(sf, 1),
                                        "n_openings": len(holes_norm),
                                        "w_ft": round(w * ftpx, 1),
                                        "h_ft": round(h * ftpx, 1),
                                        "basis": "rendered colors @ 1\"=%g'" % round(ft_pt * 72, 2)}})
    return out


def _pattern_gap_openings(poly, hair_v, hair_h, existing_holes, W, H, ft_pt,
                          jamb_v=None, jamb_h=None):
    """Openings the way the estimator sees them: pattern lines STOP at doors/windows.
    An aligned gap recurring across ≥5 consecutive pattern lines = an opening — no
    drawn frame required. Returns (holes_norm, total_pt2). Conservative: plausible
    door/window sizes only, skips anything overlapping an already-found hole, and
    total deduction capped at 40% of the piece."""
    from shapely.geometry import Point as _Pt3
    from collections import defaultdict
    x0b, y0b, x1b, y1b = poly.bounds
    # dominant pattern orientation inside the piece
    vlen = sum(min(hi, y1b) - max(lo, y0b) for (x, lo, hi) in hair_v
               if x0b <= x <= x1b and min(hi, y1b) > max(lo, y0b))
    hlen = sum(min(hi, x1b) - max(lo, x0b) for (y, lo, hi) in hair_h
               if y0b <= y <= y1b and min(hi, x1b) > max(lo, x0b))
    horiz = hlen >= vlen                 # courses run horizontally → gaps are vertical slots
    rows = defaultdict(list)
    if horiz:
        for (y, lo, hi) in hair_h:
            if y0b < y < y1b and poly.contains(_Pt3(max(lo, x0b) / 2 + min(hi, x1b) / 2, y)):
                rows[round(y / 4)].append((max(lo, x0b), min(hi, x1b)))
    else:
        for (x, lo, hi) in hair_v:
            if x0b < x < x1b and poly.contains(_Pt3(x, max(lo, y0b) / 2 + min(hi, y1b) / 2)):
                rows[round(x / 4)].append((max(lo, y0b), min(hi, y1b)))
    clusters = []                        # [ [g0,g1,rows_set] ]
    for rk, segs in rows.items():
        segs = sorted(s for s in segs if s[1] > s[0])
        for (a0, a1), (b0, b1) in zip(segs, segs[1:]):
            g = b0 - a1
            if not (12 <= g <= 220):
                continue
            placed = False
            for cl in clusters:
                ovl = min(cl[1], b0) - max(cl[0], a1)
                if ovl > 0.5 * min(g, cl[1] - cl[0]):
                    cl[0] = (cl[0] + a1) / 2; cl[1] = (cl[1] + b0) / 2
                    cl[2].add(rk); placed = True
                    break
            if not placed:
                clusters.append([a1, b0, {rk}])
    holes, tot = [], 0.0
    max_ded = 0.4 * poly.area
    jambs = (jamb_v if jamb_v is not None else hair_v) if horiz else \
            (jamb_h if jamb_h is not None else hair_h)
    def _jamb_at(c, lo, hi):
        need = 0.6 * (hi - lo)
        cov = sum(min(jhi, hi) - max(jlo, lo) for (jc, jlo, jhi) in jambs
                  if abs(jc - c) <= 8 and min(jhi, hi) > max(jlo, lo))
        return cov >= need
    for g0, g1, rks in clusters:
        if len(rks) < 5:
            continue
        span = (max(rks) - min(rks) + 1) * 4.0
        wft = (g1 - g0) * ft_pt
        hft = span * ft_pt
        if not (1.2 <= wft <= 22 and hft >= 2.0) or wft * hft < 8:
            continue
        # a REAL opening has JAMBS: lines at both gap edges spanning it. Text/leader
        # interruptions don't — this is what stops false deductions (1987 regressed
        # -0% -> -10% without it; she deducts doors, not noise).
        rlo, rhi = min(rks) * 4 - 2, max(rks) * 4 + 2
        if not (_jamb_at(g0, rlo, rhi) and _jamb_at(g1, rlo, rhi)):
            continue
        if horiz:
            r = (g0, min(rks) * 4 - 2, g1, max(rks) * 4 + 2)
        else:
            r = (min(rks) * 4 - 2, g0, max(rks) * 4 + 2, g1)
        # never double-deduct a window the rect-detector already found
        dup = False
        for h in existing_holes:
            hx = [q[0] * W for q in h]; hy = [q[1] * H for q in h]
            ix = min(r[2], max(hx)) - max(r[0], min(hx))
            iy = min(r[3], max(hy)) - max(r[1], min(hy))
            if ix > 0 and iy > 0 and ix * iy > 0.3 * (r[2] - r[0]) * (r[3] - r[1]):
                dup = True
                break
        if dup:
            continue
        a = (r[2] - r[0]) * (r[3] - r[1])
        if tot + a > max_ded:
            continue
        tot += a
        holes.append([[round(r[0] / W, 5), round(r[1] / H, 5)], [round(r[2] / W, 5), round(r[1] / H, 5)],
                      [round(r[2] / W, 5), round(r[3] / H, 5)], [round(r[0] / W, 5), round(r[3] / H, 5)]])
    return holes[:12], tot


_TAG_RE = None


def _tag_split(pdf_bytes, page_index, polys, snapx, W, H):
    """Split faces at material-tag territories. Tags = short letter+digit codes
    ("MT-5", "SIDE-1", "B2") — grid bubbles (single chars) and dimensions never match."""
    global _TAG_RE
    import re as _re
    if _TAG_RE is None:
        _TAG_RE = _re.compile(r"^[A-Za-z]{1,6}[-–]?\d{1,2}$")
    doc_t = fitz.open(stream=pdf_bytes, filetype="pdf")
    pg_t = doc_t[page_index]
    words = []
    try:
        for w in pg_t.get_text("words"):
            t = (w[4] or "").strip().rstrip(".,;:")
            if 2 <= len(t) <= 8 and _TAG_RE.match(t) and any(c.isalpha() for c in t) \
               and any(c.isdigit() for c in t):
                words.append(((w[0] + w[2]) / 2, (w[1] + w[3]) / 2, t.upper()))
    except Exception:
        pass
    doc_t.close()
    if len(words) < 2:
        return polys
    from shapely.geometry import Polygon as _P, Point as _Pt, LineString as _LS
    from shapely.ops import split as _split
    out = []
    for p in polys:
        try:
            poly = _P([(x * W, y * H) for x, y in p["points"]]).buffer(0)
            if poly.is_empty:
                out.append(p); continue
            inside = [(x, y, t) for (x, y, t) in words if poly.contains(_Pt(x, y))]
            tags = sorted(set(t for _, _, t in inside))
            if len(tags) < 2:
                if len(tags) == 1 and not p.get("named_by_tag"):
                    q = dict(p); q["material"] = q["category"] = q["group"] = tags[0]
                    q["named_by_tag"] = True
                    out.append(q); continue
                out.append(p); continue
            # territories along x: cluster tag words by tag, cut at midpoints between
            # adjacent DIFFERENT-tag clusters, snapped to the nearest structural line
            cl = {}
            for (x, y, t) in inside:
                cl.setdefault(t, []).append(x)
            marks = sorted((sum(v) / len(v), t) for t, v in cl.items())
            cuts = []
            for (xa, ta), (xb, tb) in zip(marks, marks[1:]):
                if ta == tb or xb - xa < 40:
                    continue
                mid = (xa + xb) / 2
                near = [s for s in snapx if xa + 8 < s < xb - 8]
                cut = min(near, key=lambda s: abs(s - mid)) if near else mid
                cuts.append(cut)
            if not cuts:
                out.append(p); continue
            pieces = [poly]
            x0b, y0b, x1b, y1b = poly.bounds
            for cx_ in cuts[:6]:
                nxt = []
                for q_ in pieces:
                    try:
                        nxt.extend(list(_split(q_, _LS([(cx_, y0b - 5), (cx_, y1b + 5)])).geoms))
                    except Exception:
                        nxt.append(q_)
                pieces = nxt
            pieces = [q_ for q_ in pieces if q_.geom_type == "Polygon" and q_.area >= 0.04 * poly.area]
            if len(pieces) < 2:
                out.append(p); continue
            total = sum(q_.area for q_ in pieces)
            for q_ in pieces:
                ring = list(q_.exterior.coords)[:-1]
                pts = [[round(px / W, 5), round(py / H, 5)] for px, py in ring]
                # name each piece by ITS tags (majority inside)
                mine = [t for (x, y, t) in inside if q_.contains(_Pt(x, y))]
                name = max(set(mine), key=mine.count) if mine else p.get("material", "")
                holes = [h for h in (p.get("holes") or [])
                         if q_.contains(_Pt(sum(a[0] for a in h) / len(h) * W,
                                             sum(a[1] for a in h) / len(h) * H))]
                nq = dict(p)
                nq["points"] = pts
                nq["area_sf"] = round(p.get("area_sf", 0) * q_.area / total, 1)
                nq["holes"] = holes
                nq["material"] = nq["category"] = nq["group"] = name
                nq["named_by_tag"] = True
                nq["label"] = f"~{round(nq['area_sf']):,} SF"
                xs = [a for a, _ in pts]; ys = [b for _, b in pts]
                nq["cx"] = round(sum(xs) / len(xs), 5); nq["cy"] = round(sum(ys) / len(ys), 5)
                out.append(nq)
        except Exception:
            out.append(p)
    return out


def piece_signature(segs_disp, pts_disp, W, H):
    """Pattern fingerprint of ONE piece from the segments inside it: dominant axis @ median
    spacing bucket. Two wall pieces are the SAME material only if their patterns match —
    a terra-cotta lap band and a metal-panel band must never weld even when they touch."""
    try:
        from shapely.geometry import Polygon as _P, Point as _Pt
        poly = _P(pts_disp).buffer(0)
        if poly.is_empty:
            return "plain"
        minx, miny, maxx, maxy = poly.bounds
        vs, hs = [], []
        for (x1, y1, x2, y2) in segs_disp:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            if not (minx <= mx <= maxx and miny <= my <= maxy):
                continue
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            L = (dx * dx + dy * dy) ** 0.5
            if L < 8:
                continue
            if not poly.contains(_Pt(mx, my)):
                continue
            if dx < 0.15 * L:
                vs.append(round(mx))
            elif dy < 0.15 * L:
                hs.append(round(my))
        def spacing(cs):
            cs = sorted(set(cs))
            gaps = [b - a for a, b in zip(cs, cs[1:]) if b - a > 1]
            return sorted(gaps)[len(gaps) // 2] if len(gaps) >= 3 else None
        sv, sh = spacing(vs), spacing(hs)
        if sv and (not sh or len(vs) >= len(hs)):
            return f"v{int(round(sv / 6))}"      # 6pt buckets: same product, tolerant of noise
        if sh:
            return f"h{int(round(sh / 6))}"
        return "plain"
    except Exception:
        return "plain"


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
        grp = ((seed.get("group") or seed.get("material")), seed.get("psig", "plain"))
        members = [seed]
        grew = True
        while grew:
            grew = False
            mb = [bbox(m) for m in members]
            keep = []
            cb = (min(m[0] for m in mb), min(m[1] for m in mb), max(m[2] for m in mb), max(m[3] for m in mb))
            for p in remaining:
                if ((p.get("group") or p.get("material")), p.get("psig", "plain")) != grp:
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
        # weld the cluster — EXACT geometry first (shapely union with mitred closing: square
        # corners by construction, no raster stair-noise, real holes); raster fallback second
        try:
            from shapely.geometry import Polygon as _SPoly, MultiPolygon as _SMulti
            from shapely.ops import unary_union as _sunion
            parts = []
            for p in members:
                try:
                    sp = _SPoly([(x, y) for x, y in p["points"]]).buffer(0)
                    if not sp.is_empty:
                        parts.append(sp)
                except Exception:
                    pass
            if parts:
                u = _sunion(parts)
                g2 = gap / 2.0
                u = u.buffer(g2, join_style=2).buffer(-g2, join_style=2)   # close butt joints, mitred = square
                if isinstance(u, _SMulti):
                    u = max(u.geoms, key=lambda q: q.area)
                u = u.simplify(0.0022, preserve_topology=True)
                ext = list(u.exterior.coords)
                if len(ext) >= 4:
                    face = dict(members[0])
                    face["points"] = [[round(float(x), 5), round(float(y), 5)] for x, y in ext[:-1]]
                    face["area_sf"] = round(sum(p.get("area_sf", 0) for p in members), 1)
                    holes = []
                    for p in members:
                        holes += (p.get("holes") or [])
                    for ring in list(u.interiors)[:12]:      # true geometric holes from the union
                        rc = list(ring.coords)
                        if len(rc) >= 4:
                            holes.append([[round(float(x), 5), round(float(y), 5)] for x, y in rc[:-1]])
                    face["holes"] = holes[:24]
                    face["cx"] = round(sum(pt[0] for pt in face["points"]) / len(face["points"]), 5)
                    face["cy"] = round(sum(pt[1] for pt in face["points"]) / len(face["points"]), 5)
                    if len(members) > 1:
                        face["sf_exact"] = True
                        face["merged_pieces"] = len(members)
                    face["label"] = f"~{round(face['area_sf']):,} SF"
                    out.append(face)
                    continue
        except Exception:
            pass
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
