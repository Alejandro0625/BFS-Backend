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


def _linework(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    lines = (gray < DARK).astype(np.uint8)
    lines = cv2.morphologyEx(lines, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return lines


def _poly_from_region(region, W, H):
    cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    ap = cv2.approxPolyDP(c, 0.004 * cv2.arcLength(c, True), True).reshape(-1, 2)
    if len(ap) < 3:
        return None
    return [[round(float(x) / W, 5), round(float(y) / H, 5)] for x, y in ap]


def bucket(pdf_bytes, page_index, point):
    """point = [nx, ny] normalized. Returns dict with status 'ok' (polygon+area_sf) or 'leak'."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    img, ftpx, sc, conf, z = _render(doc, page_index)
    doc.close()
    H, W = img.shape[:2]
    lines = _linework(img)
    free = (1 - lines).astype(np.uint8)
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
    return {"status": "ok", "points": poly, "area_sf": round(area * ftpx ** 2, 1),
            "scale_confirmed": conf, "source": "bucket"}


def corners(pdf_bytes, page_index, pts):
    """pts = list of [nx, ny] rough corner clicks (in order). Each snaps to the nearest strong vector
    line in x and y → exact polygon + SF. Robust on hatched fields (the estimator supplies the shape)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    img, ftpx, sc, conf, z = _render(doc, page_index)
    H, W = img.shape[:2]
    struct = np.zeros((H, W), np.uint8)                     # medium+long vector lines (window/door outlines; NOT hatch)
    try:
        for d in doc[page_index].get_drawings():
            for it in d.get("items", []):
                if it[0] == "l":
                    p1, p2 = it[1], it[2]
                    if ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5 >= 20:
                        cv2.line(struct, (int(p1.x * z), int(p1.y * z)), (int(p2.x * z), int(p2.y * z)), 1, 2)
                elif it[0] == "re":
                    r = it[1]; cv2.rectangle(struct, (int(r.x0 * z), int(r.y0 * z)), (int(r.x1 * z), int(r.y1 * z)), 1, 2)
    except Exception:
        pass
    doc.close()
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY); dark = (gray < DARK).astype(np.uint8)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, W // 60), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, H // 60)))
    Hl = cv2.morphologyEx(dark, cv2.MORPH_OPEN, hk)
    Vl = cv2.morphologyEx(dark, cv2.MORPH_OPEN, vk)
    m = int(SNAP_MARGIN * max(W, H))
    snapped = []
    for nx, ny in pts:
        px, py = int(nx * W), int(ny * H)
        yb = Hl[:, max(0, px - 4):px + 4].max(1); xb = Vl[max(0, py - 4):py + 4, :].max(0)
        yi = np.where(yb[max(0, py - m):py + m] > 0)[0]
        xi = np.where(xb[max(0, px - m):px + m] > 0)[0]
        sy = (max(0, py - m) + yi[np.argmin(np.abs(max(0, py - m) + yi - py))]) if len(yi) else py
        sx = (max(0, px - m) + xi[np.argmin(np.abs(max(0, px - m) + xi - px))]) if len(xi) else px
        snapped.append([round(sx / W, 5), round(sy / H, 5)])
    poly_px = np.array([[int(x * W), int(y * H)] for x, y in snapped], np.int32)
    gross = abs(cv2.contourArea(poly_px)) * ftpx ** 2
    op_sf, op_polys, review = _detect_openings(struct, poly_px, ftpx, W, H)
    net = max(0.0, gross - op_sf)
    return {"status": "ok", "points": snapped, "area_sf": round(net, 1), "gross_sf": round(gross, 1),
            "opening_sf": round(op_sf, 1), "openings": op_polys, "n_openings": len(op_polys),
            "openings_review": review, "scale_confirmed": conf, "source": "corners"}


def _detect_openings(struct, poly_px, ftpx, W, H):
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
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 0.012 * gross_px or a > 0.6 * gross_px:      # window/door-sized only
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
