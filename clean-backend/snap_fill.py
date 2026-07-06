"""Coloring-book BUCKET fill + corner-snap — click a wall, get the exact polygon + SF from the VECTOR
geometry (no model, no guessing). Additive: this does NOT touch the digitize-markup path; it's a new
assist layer. The estimator names the material (we can't tell metal from lap reliably; she can in a click).

Two operations, both exact-from-geometry:
  * bucket(point)  — flood-fill the region enclosed by the drawing's linework at the click. If the fill
    "leaks" (region larger than a sane wall), we say so and the caller asks for corners instead.
  * corners(pts)   — snap each rough corner click to the nearest strong vector line → exact polygon.

SF = pixel_area * (feet_per_pixel)^2, feet_per_pixel from the drawing's own scale note."""
import numpy as np
import cv2
import fitz
import texture

RENDER_LS = 4000          # high-res so the linework is crisp
DARK = 170                # a pixel darker than this is "line/ink"
LEAK_FRAC = 0.10          # a single wall on a multi-view sheet shouldn't exceed ~10% → treat as a leak (tightened: at 0.18 a fill could leak into an adjacent elevation and still pass)
SNAP_MARGIN = 0.012       # corner snap search window, fraction of the long side


def _render(doc, page_index):
    pg = doc[page_index]
    z = RENDER_LS / max(pg.rect.width, pg.rect.height)
    pix = pg.get_pixmap(matrix=fitz.Matrix(z, z))
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif pix.n == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    sc, conf = texture._read_scale(doc, page_index)
    ft_per_px = sc / (72 * z)
    return img[:, :, :3], ft_per_px, sc, conf, z


def _rot_pt(pg, x, y, z):
    """Drawing coords are UNROTATED; the render is rotated. Map through the page rotation."""
    p = fitz.Point(x, y) * pg.rotation_matrix
    return int(p.x * z), int(p.y * z)


def _vector_struct(pg, z, H, W, min_len=8, width=3, collect_ends=False, end_min=15):
    """Rasterize the page's REAL vector linework (rotation-aware). Skips strokes shorter than
    min_len — text glyphs, arrowheads, dim ticks — so notes on top of a wall aren't barriers."""
    struct = np.zeros((H, W), np.uint8)
    ends = []
    try:
        for d in pg.get_drawings():
            # barrier rule matches what the ESTIMATOR SEES: only strokes that render visibly
            # block a fill — dark AND heavier than a hairline. Panel seams/ribs are hairlines
            # (or light gray): texture, not walls. Wall outlines are heavier ink.
            col = d.get("color")
            wd = d.get("width") or 0
            dark_stroke = (col is not None and len(col) >= 3
                           and (col[0] + col[1] + col[2]) / 3 < 0.66
                           and wd * z >= 1.0)
            for it in d.get("items", []):
                if it[0] == "l":
                    p1, p2 = it[1], it[2]
                    L = ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5
                    if L >= min_len and dark_stroke:
                        cv2.line(struct, _rot_pt(pg, p1.x, p1.y, z), _rot_pt(pg, p2.x, p2.y, z), 1, width)
                    if collect_ends and L >= end_min:
                        ends.append(_rot_pt(pg, p1.x, p1.y, z)); ends.append(_rot_pt(pg, p2.x, p2.y, z))
                elif it[0] == "re":
                    r = it[1]
                    cs = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)]
                    rp = [_rot_pt(pg, X, Y, z) for (X, Y) in cs]
                    if dark_stroke:
                        for k in range(4):
                            cv2.line(struct, rp[k], rp[(k + 1) % 4], 1, width)
                    if collect_ends:
                        ends += rp
    except Exception:
        pass
    return struct, ends


def _linework(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lines = (gray < DARK).astype(np.uint8)
    lines = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return lines


def _all_segs(pg, z, min_len=6):
    """All visible line segments in raster coords (rotation-aware) — for pattern signatures."""
    out = []
    try:
        for d in pg.get_drawings():
            for it in d.get("items", []):
                if it[0] == "l":
                    p1, p2 = it[1], it[2]
                    if ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5 >= min_len:
                        a = _rot_pt(pg, p1.x, p1.y, z); b = _rot_pt(pg, p2.x, p2.y, z)
                        out.append((a[0], a[1], b[0], b[1]))
    except Exception:
        pass
    return out[:20000]


def _pattern_sig(segs, poly, W, H, ftpx):
    """Signature of the pattern INSIDE a region: dominant line axis + median spacing in feet.
    Same signature = same material pattern -> regions group across tools and pages."""
    try:
        pts = np.array([[int(x * W), int(y * H)] for x, y in poly], np.int32)
        vs = []; hs = []
        for (x1, y1, x2, y2) in segs:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            if cv2.pointPolygonTest(pts, (float(mx), float(my)), False) < 0:
                continue
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            L = (dx * dx + dy * dy) ** 0.5
            if L < 8:
                continue
            if dx < 0.15 * L:
                vs.append(round((x1 + x2) / 2))
            elif dy < 0.15 * L:
                hs.append(round((y1 + y2) / 2))
        def spacing(cs):
            cs = sorted(set(cs))
            gaps = [b - a for a, b in zip(cs, cs[1:]) if b - a > 1]
            return sorted(gaps)[len(gaps) // 2] if len(gaps) >= 3 else None
        sv, sh = spacing(vs), spacing(hs)
        if sv and (not sh or len(vs) >= len(hs)):
            return f"v@{sv * ftpx:.1f}ft"
        if sh:
            return f"h@{sh * ftpx:.1f}ft"
        return "plain"
    except Exception:
        return "plain"


def _poly_from_region(region, W, H):
    cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    ap = cv2.approxPolyDP(c, 0.004 * cv2.arcLength(c, True), True).reshape(-1, 2)
    if len(ap) < 3:
        return None
    return [[round(float(x) / W, 5), round(float(y) / H, 5)] for x, y in ap]


def _pip_norm(pt, poly):
    """point-in-polygon on normalized [nx,ny] coords."""
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i - 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
    return inside


def bucket(pdf_bytes, page_index, point):
    """Click a wall -> exact polygon + SF. point = [nx, ny] normalized (DISPLAY coords).

    PRIMARY: return the VECTOR-ENGINE region under the click — the same reader that produces the
    auto-detect regions (reads the panel/seam pattern, spans the whole wall, nets openings, snaps
    borders). This fixes the flood-fill failures on patterned walls: diagonal hatch and dimension
    lines no longer chop the fill into a jagged strip, and windows are already netted out.
    FALLBACK: the classic flood-fill for pages/spots the vector engine didn't cover."""
    try:
        import vector_hatch
        vpolys, VW, VH, vinfo = vector_hatch.detect(pdf_bytes, page_index)
        hits = [p for p in (vpolys or []) if len(p.get("points", [])) >= 3 and _pip_norm(point, p["points"])]
        if hits:
            best = max(hits, key=lambda p: p.get("area_sf", 0))   # if regions overlap, the largest wall wins
            # THE FACE IS ONE PIECE (estimator's rule): the engine stores a wall as several
            # drawn pieces — gather every SAME-PATTERN piece touching the clicked one (BFS over
            # near-adjacent bboxes) and weld them into ONE shape. SF = sum of the pieces' own
            # (already net-of-openings) SF — the weld never invents area.
            group = best.get("group") or best.get("material")
            fam = [p for p in vpolys if (p.get("group") or p.get("material")) == group and len(p.get("points", [])) >= 3]
            def bbox(p):
                xs = [q[0] for q in p["points"]]; ys = [q[1] for q in p["points"]]
                return (min(xs), min(ys), max(xs), max(ys))
            GAP = 0.015    # pieces within ~1.5% of the sheet touch/butt-join = same face
            members = [best]; rest = [p for p in fam if p is not best]
            grew = True
            while grew and rest:
                grew = False
                mb = [bbox(m) for m in members]
                keep2 = []
                for p in rest:
                    b = bbox(p)
                    near = any(not (b[2] < m[0] - GAP or b[0] > m[2] + GAP or b[3] < m[1] - GAP or b[1] > m[3] + GAP) for m in mb)
                    if near:
                        members.append(p); grew = True
                    else:
                        keep2.append(p)
                rest = keep2
            if len(members) == 1:
                return {"status": "ok", "points": best["points"], "area_sf": best.get("area_sf", 0),
                        "holes": best.get("holes", []), "material": best.get("material", ""),
                        "scale_confirmed": bool(vinfo.get("scale_confirmed")), "source": "bucket-vector",
                        "width": VW, "height": VH}
            # weld: rasterize members, close the butt-joint gaps, take the outer contour
            S = 1600
            SH = max(200, int(S * VH / max(1.0, VW)))
            m2 = np.zeros((SH, S), np.uint8)
            for p in members:
                cnt = np.array([[int(x * S), int(y * SH)] for x, y in p["points"]], np.int32)
                cv2.fillPoly(m2, [cnt.reshape(-1, 1, 2)], 1)
            k = max(3, int(GAP * S))
            m2 = cv2.morphologyEx(m2, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
            cnts, _ = cv2.findContours(m2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                ap = cv2.approxPolyDP(c, 0.004 * cv2.arcLength(c, True), True).reshape(-1, 2)
                if len(ap) >= 3:
                    pts = [[round(float(x) / S, 5), round(float(y) / SH, 5)] for x, y in ap]
                    holes = []
                    for p in members:
                        holes += (p.get("holes") or [])
                    return {"status": "ok", "points": pts,
                            "area_sf": round(sum(p.get("area_sf", 0) for p in members), 1),
                            "holes": holes[:40], "material": best.get("material", ""),
                            "merged_pieces": len(members),
                            "scale_confirmed": bool(vinfo.get("scale_confirmed")), "source": "bucket-vector",
                            "width": VW, "height": VH}
            # weld failed → clicked piece alone (never block the estimator)
            return {"status": "ok", "points": best["points"], "area_sf": best.get("area_sf", 0),
                    "holes": best.get("holes", []), "material": best.get("material", ""),
                    "scale_confirmed": bool(vinfo.get("scale_confirmed")), "source": "bucket-vector",
                    "width": VW, "height": VH}
    except Exception:
        pass
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    img, ftpx, sc, conf, z = _render(doc, page_index)
    H, W = img.shape[:2]
    struct, _ = _vector_struct(doc[page_index], z, H, W, min_len=4, width=3)
    page_segs = _all_segs(doc[page_index], z)
    doc.close()
    if not struct.any():                                     # scanned/no-vector page → old pixel barriers
        struct = _linework(img)
    free = (1 - struct).astype(np.uint8)
    px, py = int(point[0] * W), int(point[1] * H)
    px = min(max(px, 0), W - 1); py = min(max(py, 0), H - 1)
    if free[py, px] == 0:                                   # clicked on a line → nudge to nearest free px
        y0, x0 = max(0, py - 10), max(0, px - 10)
        ys, xs = np.where(free[y0:py + 10, x0:px + 10] > 0)
        if len(xs) == 0:
            return {"status": "empty"}
        px, py = x0 + xs[0], y0 + ys[0]
    ff = free.copy(); mask = np.zeros((H + 2, W + 2), np.uint8)
    cv2.floodFill(ff, mask, (px, py), 2)
    region = (ff == 2).astype(np.uint8)
    area = int(region.sum()); frac = area / (H * W)
    if frac > LEAK_FRAC or frac < 1e-5:
        return {"status": "leak", "fill_frac": round(frac, 3)}   # caller should switch to corner mode
    poly = _poly_from_region(region, W, H)
    if poly is None:
        return {"status": "empty"}
    # interior HOLES: keep as deductions ONLY the window/door-shaped ones (clean rectangles of
    # real size). Text callouts, leader arrows and dimension boxes sitting ON the wall are
    # annotations, not holes — their area goes BACK into the wall SF (estimator's rule).
    holes = []
    add_back_px = 0
    cnts, hier = cv2.findContours(region, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is not None and len(cnts):
        hier = hier[0]
        ext = [i for i in range(len(cnts)) if hier[i][3] == -1]
        if ext:
            big = max(ext, key=lambda i: cv2.contourArea(cnts[i]))
            for i in range(len(cnts)):
                if hier[i][3] != big:
                    continue
                a = cv2.contourArea(cnts[i])
                if a <= 0.002 * area:
                    add_back_px += a          # crumbs (letter counters, tick gaps) = wall
                    continue
                x, y, w2, h2 = cv2.boundingRect(cnts[i])
                rectish = a / (w2 * h2 + 1e-6) > 0.6 and 0.15 < w2 / (h2 + 1e-6) < 8
                if rectish and a * ftpx ** 2 >= 4:   # a real opening: window/door/louver
                    ap = cv2.approxPolyDP(cnts[i], 0.02 * cv2.arcLength(cnts[i], True), True).reshape(-1, 2)
                    if len(ap) >= 3:
                        holes.append([[round(float(px) / W, 5), round(float(py) / H, 5)] for px, py in ap])
                else:                                 # text/arrow-shaped island: not a hole
                    add_back_px += a
    return {"status": "ok", "points": poly, "area_sf": round((area + add_back_px) * ftpx ** 2, 1),
            "holes": holes[:40], "pattern_sig": _pattern_sig(page_segs, poly, W, H, ftpx),
            "scale_confirmed": conf, "source": "bucket"}


def corners(pdf_bytes, page_index, pts, min_opening_sf=0.0):
    """pts = list of [nx, ny] rough corner clicks (in order). Each click snaps to the nearest REAL
    CORNER of the drawing — a vector line ENDPOINT (eave corner, gable PEAK, rake end — any angle),
    falling back to the nearest long horizontal/vertical line. So gables and sloped walls measure
    exactly, not just rectangles. min_opening_sf: openings smaller than this are NOT deducted
    (many shops don't deduct small windows — labor around them costs the same)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    img, ftpx, sc, conf, z = _render(doc, page_index)
    H, W = img.shape[:2]
    struct, ends = _vector_struct(doc[page_index], z, H, W, min_len=20, width=2, collect_ends=True, end_min=15)
    doc.close()
    ends = np.array(ends, np.float32) if ends else np.zeros((0, 2), np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY); dark = (gray < DARK).astype(np.uint8)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, W // 60), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, H // 60)))
    Hl = cv2.morphologyEx(dark, cv2.MORPH_OPEN, hk)
    Vl = cv2.morphologyEx(dark, cv2.MORPH_OPEN, vk)
    m = int(SNAP_MARGIN * max(W, H))
    mp = int(0.011 * max(W, H))                              # point-snap radius (a bit tighter than line snap)
    snapped = []
    for nx, ny in pts:
        px, py = int(nx * W), int(ny * H)
        sx = sy = None
        if len(ends):                                        # 1) nearest true drawing corner within radius (handles gable peaks/slopes)
            dd = np.hypot(ends[:, 0] - px, ends[:, 1] - py)
            j = int(dd.argmin())
            if dd[j] <= mp:
                sx, sy = float(ends[j, 0]), float(ends[j, 1])
        if sx is None:                                       # 2) fallback: nearest long H/V line per axis (old behavior)
            yb = Hl[:, max(0, px - 4):px + 4].max(1); xb = Vl[max(0, py - 4):py + 4, :].max(0)
            yi = np.where(yb[max(0, py - m):py + m] > 0)[0]
            xi = np.where(xb[max(0, px - m):px + m] > 0)[0]
            sy = (max(0, py - m) + yi[np.argmin(np.abs(max(0, py - m) + yi - py))]) if len(yi) else py
            sx = (max(0, px - m) + xi[np.argmin(np.abs(max(0, px - m) + xi - px))]) if len(xi) else px
        snapped.append([round(min(max(sx, 0), W - 1) / W, 5), round(min(max(sy, 0), H - 1) / H, 5)])
    poly_px = np.array([[int(x * W), int(y * H)] for x, y in snapped], np.int32)
    gross = abs(cv2.contourArea(poly_px)) * ftpx ** 2
    op_sf, op_polys, review = _detect_openings(struct, poly_px, ftpx, W, H, min_opening_sf)
    net = max(0.0, gross - op_sf)
    return {"status": "ok", "points": snapped, "area_sf": round(net, 1), "gross_sf": round(gross, 1),
            "opening_sf": round(op_sf, 1), "openings": op_polys, "n_openings": len(op_polys),
            "openings_review": review, "scale_confirmed": conf, "source": "corners"}


def refine_group(pdf_bytes, page_index, patches, min_opening_sf=0.0):
    """#3 EXACT-ON-SELECT: turn a texture-preview group (blocky patches [[nx,ny,nw,nh],...]) into EXACT
    shapes — each patch-blob's corners snap to the drawing's real vector endpoints (any angle, same
    primitive as the gable-verified corner tool), openings deducted. Output shapes feed the normal
    exact pipeline, so the estimator's one click on a group yields a bid-grade SF, not an estimate."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    img, ftpx, sc, conf, z = _render(doc, page_index)
    H, W = img.shape[:2]
    struct, ends = _vector_struct(doc[page_index], z, H, W, min_len=20, width=2, collect_ends=True, end_min=15)
    doc.close()
    ends = np.array(ends, np.float32) if ends else np.zeros((0, 2), np.float32)
    mask = np.zeros((H, W), np.uint8)
    for p in patches or []:
        x0, y0 = int(p[0] * W), int(p[1] * H)
        cv2.rectangle(mask, (x0, y0), (min(W - 1, x0 + int(p[2] * W)), min(H - 1, y0 + int(p[3] * H))), 1, -1)
    if not mask.any():
        return {"status": "empty", "shapes": []}
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))   # bridge patch gaps within one wall
    mp = int(0.015 * max(W, H))                                                    # blob corners are ±patch off → slightly wider snap radius
    shapes = []
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] * ftpx ** 2 < 8:                                # skip crumbs (<8 SF)
            continue
        cnts, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        ap = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True).reshape(-1, 2)
        if len(ap) < 3:
            continue
        snapped = []
        for px, py in ap:                                                          # snap each blob corner to the nearest TRUE drawing corner
            if len(ends):
                dd = np.hypot(ends[:, 0] - px, ends[:, 1] - py)
                j = int(dd.argmin())
                if dd[j] <= mp:
                    px, py = float(ends[j, 0]), float(ends[j, 1])
            snapped.append([round(min(max(px, 0), W - 1) / W, 5), round(min(max(py, 0), H - 1) / H, 5)])
        poly_px = np.array([[int(x * W), int(y * H)] for x, y in snapped], np.int32)
        gross = abs(cv2.contourArea(poly_px)) * ftpx ** 2
        if gross < 8:
            continue
        op_sf, op_polys, review = _detect_openings(struct, poly_px, ftpx, W, H, min_opening_sf)
        shapes.append({"points": snapped, "area_sf": round(max(0.0, gross - op_sf), 1), "gross_sf": round(gross, 1),
                       "opening_sf": round(op_sf, 1), "openings": op_polys, "openings_review": review})
    return {"status": "ok", "shapes": shapes, "total_sf": round(sum(s["area_sf"] for s in shapes), 1),
            "scale_confirmed": conf, "source": "group-refined"}


def _detect_openings(struct, poly_px, ftpx, W, H, min_opening_sf=0.0):
    """Inside a wall polygon, find rectangular window/door openings to deduct → (opening_sf, polys, review).
    MONEY-SAFE: the dangerous direction is a FALSE POSITIVE — panel reveal/joint grids, louvers and
    signage are also rectangles, and deducting them under-states SF → under-bid → real loss. So we:
      * SKIP repeating same-size rectangle grids (that's a panel layout = cladding, not fenestration) —
        deduct nothing, just flag for review;
      * CAP total deduction at 35% of the wall;
      * always return the opening polygons + a review flag so the estimator vetoes them (never silent).
    Under-deducting (over-bid = lose the bid) is acceptable; over-deducting (lose money) is not."""
    H, W = struct.shape[:2]
    wall = np.zeros((H, W), np.uint8); cv2.fillPoly(wall, [poly_px], 1); gross_px = int(wall.sum())
    if gross_px < 100:
        return 0.0, [], False
    s = cv2.dilate(struct & wall, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(s, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cand = []
    min_px = max(0.012 * gross_px, (min_opening_sf / (ftpx ** 2)) if min_opening_sf > 0 else 0)
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_px or a > 0.6 * gross_px:                # window/door-sized only + shop's no-deduct-under rule
            continue
        x, y, w2, h2 = cv2.boundingRect(c)
        if a / (w2 * h2 + 1e-6) > 0.72 and 0.2 < w2 / (h2 + 1e-6) < 6:   # a filled rectangle
            cand.append((c, a))
    if not cand:
        return 0.0, [], False
    def _poly(c):
        ap = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True).reshape(-1, 2)
        return [[round(float(px) / W, 5), round(float(py) / H, 5)] for px, py in ap]
    # PANEL-GRID GUARD: many rectangles of similar size = a panel reveal/joint grid (cladding), NOT windows
    areas = sorted(a for _, a in cand)
    med = areas[len(areas) // 2]
    similar = sum(1 for a in areas if 0.6 * med <= a <= 1.6 * med)
    if len(cand) >= 4 and similar >= 0.7 * len(cand):
        return 0.0, [_poly(c) for c, _ in cand[:12]], True   # looks like a panel grid → deduct NOTHING, flag
    cand.sort(key=lambda t: -t[1])
    used = np.zeros((H, W), np.uint8); op_px = 0; polys = []; cap = int(0.35 * gross_px)
    for c, a in cand:
        mm = np.zeros((H, W), np.uint8); cv2.drawContours(mm, [c], -1, 1, -1)
        if (mm & used).sum() > 0.3 * mm.sum():              # skip overlaps (nested mullions etc.)
            continue
        if op_px + int(mm.sum()) > cap:                     # cap total deduction → never eat the wall
            break
        used |= mm; op_px += int(mm.sum()); polys.append(_poly(c))
    review = op_px >= 0.25 * gross_px                        # a big deduction → flag for a glance
    return op_px * ftpx ** 2, polys, review
