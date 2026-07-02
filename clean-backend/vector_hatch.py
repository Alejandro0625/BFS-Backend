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

MIN_SF = 100          # ignore specks — estimator noise floor
MAX_REGIONS = 40


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


def _train_polygon(reg):
    """Rectilinear outline from runs of similar extents (an estimator-looking border)."""
    part = reg["lines"]
    cols = {}
    for (c, a, b) in part:
        e = cols.setdefault(c, [a, b])
        e[0] = min(e[0], a); e[1] = max(e[1], b)
    cs = sorted(cols)
    runs = []
    for c in cs:
        t, b = cols[c]
        if runs and abs(runs[-1][2] - t) < 8 and abs(runs[-1][3] - b) < 8:
            runs[-1] = (runs[-1][0], c, min(runs[-1][2], t), max(runs[-1][3], b))
        else:
            runs.append((c, c, t, b))
    pts = []
    for (ca, cb, t, b) in runs:
        pts += [(ca, t), (cb, t)]
    for (ca, cb, t, b) in reversed(runs):
        pts += [(cb, b), (ca, b)]
    if reg["axis"] == "h":
        pts = [(y, x) for (x, y) in pts]
    return pts


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

    def add(pts_page, area_pt2, material, color, sf_exact):
        sf = area_pt2 * ft_pt * ft_pt
        if sf < MIN_SF:
            return
        disp = [fitz.Point(x, y) * rot for (x, y) in pts_page]
        norm = [[round(p.x / W, 5), round(p.y / H, 5)] for p in disp]
        cx = round(sum(p[0] for p in norm) / len(norm), 5)
        cy = round(sum(p[1] for p in norm) / len(norm), 5)
        polys.append({"points": norm, "area_sf": round(sf, 1), "cx": cx, "cy": cy,
                      "fill_color": color, "source": "vector", "material": material,
                      "category": material, "group": material, "sf_exact": sf_exact,
                      "label": f"~{round(sf):,} SF"})

    for f in fills:
        add(f["pts"], f["area_pt2"], "Panel wall (drawn fill)", [0.35, 0.55, 0.85], False)
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
        add(_train_polygon(t), _train_area_pt2(t), mat, col, True)

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
    for i, p in enumerate(polys):
        p["id"] = i
    return polys, W, H, {"ft_per_in": round(sc, 3), "scale_confirmed": conf}
