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


# ── JOINT SPLIT: read the drawing like a senior estimator ──────────────────────────
# A wall PIECE ends where a HEAVY structural line crosses the whole face (building
# corner, expansion joint, new/existing line). Panel seams are 0.24pt hairlines every
# few feet; joints are ≥1pt strokes spanning the full height (measured on Fleet: joint
# 1.56pt × full building height vs seams 0.24pt). Splitting the welded face at those
# lines returns walls the way the estimator marks them — one piece per wall.
_JOINT_MIN_WD = 1.0        # stroke weight (pt) that separates structure from seam hairlines
_JOINT_COVER = 0.80        # ink must cover ≥80% of the face's extent at that position
_JOINT_STITCH = 8.0        # collinear gaps ≤8pt bridge (line broken by ticks/labels)
_JOINT_MIN_PIECE_FRAC = 0.02  # drop split slivers under 2% of the face


def _heavy_lines(pg):
    """H/V strokes in display coords, split HEAVY (structure) vs HAIRLINE (pattern).
    Returns (vs_heavy, hs_heavy, vs_hair, hs_hair) as [(const, lo, hi)]."""
    rot = pg.rotation_matrix
    vs, hs, vh, hh = [], [], [], []
    for d in pg.get_drawings():
        wd = d.get("width") or 0
        col = d.get("color")
        if col is not None and len(col) >= 3 and (col[0] + col[1] + col[2]) / 3 >= 0.66:
            continue                      # light strokes are shading, not structure/pattern
        heavy = wd >= _JOINT_MIN_WD
        for it in d.get("items") or []:
            if it[0] != "l":
                continue
            p0 = fitz.Point(it[1]) * rot
            p1 = fitz.Point(it[2]) * rot
            dx, dy = abs(p1.x - p0.x), abs(p1.y - p0.y)
            if dy >= 8 and dx <= 1.5:
                (vs if heavy else vh).append(((p0.x + p1.x) / 2, min(p0.y, p1.y), max(p0.y, p1.y), wd))
            elif dx >= 8 and dy <= 1.5:
                (hs if heavy else hh).append(((p0.y + p1.y) / 2, min(p0.x, p1.x), max(p0.x, p1.x), wd))
    return vs, hs, [(c, a, b) for c, a, b, _ in vh], [(c, a, b) for c, a, b, _ in hh]


def _content_differs(vs_hair, hs_hair, axis, c, b0, b1, lo, hi):
    """THE HUMAN TEST: does the drawn content actually CHANGE across this line? Sample the
    hairline pattern in a window on each side — a wall boundary flips orientation or density
    (panel seams end, lap courses start, or blank); a downspout/jamb has the SAME pattern on
    both sides. axis='v': c is x, [lo,hi] is the piece's y-range (b0,b1 unused pad)."""
    WIN = 34.0
    def prof(w0, w1):
        vlen = hlen = 0.0
        if axis == "v":
            for (x, y0, y1) in vs_hair:
                if w0 <= x <= w1:
                    vlen += max(0.0, min(y1, hi) - max(y0, lo))
            for (y, x0, x1) in hs_hair:
                if lo <= y <= hi:
                    hlen += max(0.0, min(x1, w1) - max(x0, w0))
        else:
            for (y, x0, x1) in hs_hair:
                if w0 <= y <= w1:
                    hlen += max(0.0, min(x1, hi) - max(x0, lo))
            for (x, y0, y1) in vs_hair:
                if lo <= x <= hi:
                    vlen += max(0.0, min(y1, w1) - max(y0, w0))
        return vlen, hlen
    def dom(v, h):
        if v > 1.6 * h and v > 15:
            return "v"
        if h > 1.6 * v and h > 15:
            return "h"
        return "m" if (v + h) > 15 else "0"
    def differs(w0a, w1a, w0b, w1b):
        va, ha = prof(w0a, w1a)
        vb, hb = prof(w0b, w1b)
        ta, tb = va + ha, vb + hb
        if dom(va, ha) != dom(vb, hb):
            return True                   # orientation flips (or pattern → blank)
        if max(ta, tb) > 40 and (min(ta, tb) + 1) / (max(ta, tb) + 1) < 0.4:
            return True                   # same orientation but density regime changes
        return False
    if not differs(c - WIN, c - 3, c + 3, c + WIN):
        return False
    # ZOOM OUT (how a human decides): a louver/trim strip flips the pattern locally but
    # 6 ft past it the wall reads the SAME on both sides — feature IN the wall, not a
    # boundary. A real material change differs in the far field too.
    if b1 - b0 > 170 and c - 82 > b0 and c + 82 < b1:
        return differs(c - 82, c - 42, c + 42, c + 82)
    return True


def _joint_positions(lines, lo_clip, hi_clip, need):
    """Agglomerative-cluster collinear heavy segments (consts within 3pt = one line —
    fixed buckets split real joints across boundaries), stitch gaps. Returns
    [(const, full_span, wd_max)] for lines whose covered span within [lo_clip,hi_clip]
    reaches `need`; full_span = total unclipped reach (building-scale test)."""
    ent = []
    for c, lo, hi, wd in lines:
        l, h = max(lo, lo_clip), min(hi, hi_clip)
        if h > l:
            ent.append((c, l, h, hi - lo, wd))
    ent.sort()
    out = []
    i = 0
    while i < len(ent):
        j = i + 1
        while j < len(ent) and ent[j][0] - ent[j - 1][0] <= 3.0:
            j += 1
        items = ent[i:j]
        i = j
        # DASHED lines are hidden/reference geometry, never wall boundaries. A real
        # joint has at least one LONG stroke (ground lines mix long runs with short
        # ticks — median fails them; pure dashes have no stroke ≥25pt).
        if max(h - l for _, l, h, _, _ in items) < 25:
            continue
        ivs = sorted((l, h) for _, l, h, _, _ in items)
        cov = 0.0
        cs, ce = ivs[0]
        for l, h in ivs[1:]:
            if l <= ce + _JOINT_STITCH:
                ce = max(ce, h)
            else:
                cov += ce - cs
                cs, ce = l, h
        cov += ce - cs
        if cov >= need:
            out.append((sum(c for c, _, _, _, _ in items) / len(items),
                        max(f for _, _, _, f, _ in items),
                        max(w for _, _, _, _, w in items),
                        cov))
    return sorted(out)


def split_face_at_joints(pg, pts_norm, holes_norm, area_sf, W, H, click_norm, ft_pt=None,
                         filter_empty=False):
    """Split a welded face at heavy full-span structural lines. Returns
    (primary, others) piece dicts {points, area_sf, holes}. SF per piece: GEOMETRIC
    net of its holes when the sheet scale is confirmed (ft_pt given) — the way the
    estimator measures a wall; parent-area-ratio fallback otherwise. Returns None
    when no joint crosses the face."""
    try:
        from shapely.geometry import Polygon, Point as SPt, LineString, box
        from shapely.ops import split as shsplit, unary_union
        poly = Polygon([(x * W, y * H) for x, y in pts_norm]).buffer(0)
        if poly.is_empty or poly.area <= 0:
            return None
        vs, hs, vs_hair, hs_hair = _heavy_lines(pg)

        # fenestration edges: window heads/sills/jambs are HOLE boundaries, not wall
        # boundaries — a joint candidate hugging one is the window row, skip it
        hole_xs, hole_ys = [], []
        for hole in (holes_norm or []):
            hxs = [q[0] * W for q in hole]; hys = [q[1] * H for q in hole]
            hole_xs += [min(hxs), max(hxs)]
            hole_ys += [min(hys), max(hys)]

        def _on_hole_edge(vals, c):
            return any(abs(c - v) <= 8 for v in vals)

        def _height_at(p_, x):
            try:
                seg = p_.intersection(LineString([(x, p_.bounds[1] - 10), (x, p_.bounds[3] + 10)]))
                return seg.length
            except Exception:
                return 0.0

        def _width_at(p_, y):
            try:
                seg = p_.intersection(LineString([(p_.bounds[0] - 10, y), (p_.bounds[2] + 10, y)]))
                return seg.length
            except Exception:
                return 0.0

        # HIERARCHICAL, like a human: split at joints spanning THIS piece, then re-examine
        # each piece with its OWN extents (a band line often exists only on one side of a
        # building joint — it spans that piece fully even though it never crossed the face).
        def _try_line(p_, line):
            """Split by one joint line. Accept when it separates ≥2 REAL walls (each ≥15%
            of the piece and ≥18pt thick). Sliver crumbs from weld spurs are dropped — SF
            is later divided by area ratio over kept pieces, so no money is lost."""
            try:
                parts = [q for q in shsplit(p_, line).geoms if q.geom_type == "Polygon" and q.area > 0]
            except Exception:
                return None
            def _thick(q):
                qb = q.bounds
                return min(qb[2] - qb[0], qb[3] - qb[1]) >= 18
            # ≥18pt thick kills parapet/trim strips; the area floor only filters crumbs —
            # a small REAL wall (parapet screen on an L-shaped face) can be 5-15% of it.
            big = [q for q in parts if q.area >= 0.05 * p_.area and _thick(q)]
            if len(big) < 2:
                return None
            return big

        def _split_piece(p_, depth):
            if depth >= 6:      # stacked bands need 4-5 levels; evidence gates stop runaway
                return [p_]
            x0, y0, x1, y1 = p_.bounds
            if (x1 - x0) < 12 or (y1 - y0) < 12:
                return [p_]
            # pre-filter at HALF the bar against the bbox (weld spurs inflate it), then
            # verify coverage against the face's TRUE extent at that exact position —
            # a joint spanning the building must not fail because a leader spur
            # stretched the bbox.
            pvx = [(x, sp, w, cov) for x, sp, w, cov in
                   _joint_positions(vs, y0, y1, 0.5 * _JOINT_COVER * (y1 - y0))
                   if x0 + 4 < x < x1 - 4]
            phy = [(y, sp, w, cov) for y, sp, w, cov in
                   _joint_positions(hs, x0, x1, 0.5 * _JOINT_COVER * (x1 - x0))
                   if y0 + 4 < y < y1 - 4]
            for x, _sp, w, cov in pvx[:10]:
                ext = _height_at(p_, x)
                if ext < 18 or cov < _JOINT_COVER * ext:
                    continue
                # WHAT A HUMAN CHECKS at a heavy vertical line: outline-class ink (the
                # weight buildings are drawn with, ≥1.5pt) IS a boundary; girt-class ink
                # (1.0-1.5) needs evidence — the outline steps there, or the drawn content
                # changes. A downspout: girt-class, same pattern both sides → never splits.
                outline_class = w >= 1.5
                if not outline_class and _on_hole_edge(hole_xs, x):
                    continue          # window jamb line, not a wall boundary
                step = abs(_height_at(p_, x - 8) - _height_at(p_, x + 8)) > 0.12 * (y1 - y0)
                if not outline_class and not step and not _content_differs(vs_hair, hs_hair, "v", x, x0, x1, y0, y1):
                    continue
                parts = _try_line(p_, LineString([(x, y0 - 10), (x, y1 + 10)]))
                if parts:
                    out_ = []
                    for q in parts:
                        out_.extend(_split_piece(q, depth + 1))
                    return out_
            for y, sp, w, cov in phy[:10]:
                ext = _width_at(p_, y)
                if ext < 18 or cov < _JOINT_COVER * ext:
                    continue
                # buildings STACK: outline-class ink, or a line running far beyond this
                # piece (roof/story line), splits even when the pattern continues
                # (parapet screen over wall). Girt-class needs step/content evidence.
                outline_class = w >= 1.5
                building_scale = sp >= 2.2 * (x1 - x0)
                if not outline_class and not building_scale and _on_hole_edge(hole_ys, y):
                    continue          # window head/sill row, not a wall boundary
                step = abs(_width_at(p_, y - 8) - _width_at(p_, y + 8)) > 0.12 * (x1 - x0)
                if not outline_class and not building_scale and not step and \
                   not _content_differs(vs_hair, hs_hair, "h", y, y0, y1, x0, x1):
                    continue
                parts = _try_line(p_, LineString([(x0 - 10, y), (x1 + 10, y)]))
                if parts:
                    out_ = []
                    for q in parts:
                        out_.extend(_split_piece(q, depth + 1))
                    return out_
            return [p_]

        pieces = _split_piece(poly, 0)
        pieces = [p_ for p_ in pieces if p_.geom_type == "Polygon" and
                  p_.area >= _JOINT_MIN_PIECE_FRAC * poly.area]
        if filter_empty and len(pieces) >= 2:
            # weld spurs drag TRAIN faces past the building (above the roof, below grade
            # into the dimension zone). Those phantom pieces carry no drawn pattern —
            # a real clad wall is covered in seam/course hairlines. Drop the blanks so
            # they never become phantom SF. (Plain gray-FILL walls are legitimately
            # blank — this filter only runs for pattern-train faces.)
            def _pattern_len(p_):
                x0, y0, x1, y1 = p_.bounds
                tot = 0.0
                for (x, l, h) in vs_hair:
                    if x0 <= x <= x1:
                        seg = min(h, y1) - max(l, y0)
                        if seg > 0 and p_.contains(SPt(x, (max(l, y0) + min(h, y1)) / 2)):
                            tot += seg
                for (y, l, h) in hs_hair:
                    if y0 <= y <= y1:
                        seg = min(h, x1) - max(l, x0)
                        if seg > 0 and p_.contains(SPt((max(l, x0) + min(h, x1)) / 2, y)):
                            tot += seg
                return tot
            kept = [p_ for p_ in pieces if _pattern_len(p_) >= 0.004 * p_.area]
            if kept:
                pieces = kept
        if len(pieces) < 2:
            return None
        total = sum(p_.area for p_ in pieces)
        click_pt = SPt(click_norm[0] * W, click_norm[1] * H) if click_norm is not None else None
        out = []
        for p_ in pieces:
            # shave spur fingers (leader-line tails thinner than ~2ft) before measuring —
            # they're weld artifacts, not wall; mitre buffer keeps corners square
            try:
                q2 = p_.buffer(-9, join_style=2).buffer(9, join_style=2)
                if q2.geom_type == "MultiPolygon":
                    q2 = max(q2.geoms, key=lambda g: g.area)
                if q2.geom_type == "Polygon" and not q2.is_empty and q2.area >= 0.5 * p_.area:
                    p_ = q2
            except Exception:
                pass
            ring = list(p_.exterior.coords)[:-1]
            pts = [[round(px / W, 5), round(py / H, 5)] for px, py in ring]
            hl = []
            hole_pt2 = 0.0
            for hole in (holes_norm or []):
                hx = sum(q[0] for q in hole) / len(hole) * W
                hyc = sum(q[1] for q in hole) / len(hole) * H
                if p_.contains(SPt(hx, hyc)):
                    hl.append(hole)
                    try:
                        hole_pt2 += abs(Polygon([(q[0] * W, q[1] * H) for q in hole]).area)
                    except Exception:
                        pass
            if ft_pt and ft_pt > 0:
                # GEOMETRIC net — how the estimator measures: this wall's own area minus
                # its own windows. Immune to the parent's overlap-shave/spur artifacts.
                sf = max(0.0, (p_.area - hole_pt2)) * ft_pt * ft_pt
            else:
                sf = area_sf * p_.area / total
            out.append({"points": pts, "area_sf": round(sf, 1), "holes": hl,
                        "contains_click": bool(click_pt is not None and
                                               (p_.contains(click_pt) or p_.distance(click_pt) < 2))})
        if not (ft_pt and ft_pt > 0):
            # ratio mode: rounding drift goes to the largest piece (sum EXACTLY preserved)
            drift = round(area_sf - sum(o["area_sf"] for o in out), 1)
            if abs(drift) >= 0.1:
                max(out, key=lambda o: o["area_sf"])["area_sf"] = round(
                    max(out, key=lambda o: o["area_sf"])["area_sf"] + drift, 1)
        if click_norm is None:              # first-paint mode: caller wants every piece
            for o in out:
                o.pop("contains_click", None)
            return None, out
        prim = next((o for o in out if o.pop("contains_click", False)), None)
        others = [o for o in out if o is not prim]
        for o in others:
            o.pop("contains_click", None)
        if prim is None:
            return None
        return prim, others
    except Exception:
        return None


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
            # faces are now per-wall pieces; overlapping candidates → the MOST SPECIFIC
            # (smallest real) piece is what the estimator pointed at. Sub-60SF slivers
            # still lose to the wall that contains them.
            real = [p for p in hits if p.get("area_sf", 0) >= 60]
            best = min(real, key=lambda p: p.get("area_sf", 0)) if real \
                else max(hits, key=lambda p: p.get("area_sf", 0))
            # SELECT THE PATTERN, EVERYWHERE (the estimator's click-a-hatch vision): the click
            # picks a pattern — return EVERY face on the page with that same pattern (all
            # elevations), each as its own already-welded, net-of-openings shape. The clicked
            # face is primary; the rest ride along as `siblings` so one click fills them all.
            key = ((best.get("group") or best.get("material")), best.get("psig", "plain"))
            fam = [p for p in vpolys
                   if ((p.get("group") or p.get("material")), p.get("psig", "plain")) == key
                   and len(p.get("points", [])) >= 3]
            sibs = [{"points": p["points"], "area_sf": p.get("area_sf", 0), "holes": p.get("holes", [])}
                    for p in fam if p is not best][:30]
            prim_pts, prim_sf, prim_holes = best["points"], best.get("area_sf", 0), best.get("holes", [])
            n_joints = 0
            # READ IT LIKE THE ESTIMATOR: a wall piece ends at a HEAVY structural line that
            # crosses the whole face (building corner / expansion joint). Split the welded
            # face there; the clicked piece is the wall, the rest ride along as siblings —
            # the pattern total is IDENTICAL, but each wall now carries its own SF.
            try:
                sdoc = fitz.open(stream=pdf_bytes, filetype="pdf")
                ftp = (float(vinfo.get("ft_per_in") or 0) / 72.0) if vinfo.get("scale_confirmed") else None
                is_train = not (best.get("group") or best.get("material") or "").startswith("Panel wall")
                sp = split_face_at_joints(sdoc[page_index], prim_pts, prim_holes, prim_sf,
                                          VW, VH, point, ft_pt=ftp, filter_empty=is_train)
                sdoc.close()
                if sp:
                    prim, rest = sp
                    prim_pts, prim_sf, prim_holes = prim["points"], prim["area_sf"], prim["holes"]
                    sibs = rest + sibs
                    n_joints = len(rest)
                if ftp:
                    # first-paint pieces carry ratio SF (page-sum safety); a bucket CLICK
                    # is the estimator measuring THIS wall — give the geometric net,
                    # the number his takeoff shows for it
                    try:
                        from shapely.geometry import Polygon as _SP
                        pp = _SP([(x * VW, y * VH) for x, y in prim_pts]).buffer(0)
                        hp = sum(abs(_SP([(q[0] * VW, q[1] * VH) for q in h]).area)
                                 for h in (prim_holes or []) if len(h) >= 3)
                        if not pp.is_empty:
                            prim_sf = round(max(0.0, pp.area - hp) * ftp * ftp, 1)
                    except Exception:
                        pass
            except Exception:
                pass
            return {"status": "ok", "points": prim_pts, "area_sf": prim_sf,
                    "holes": prim_holes, "material": best.get("material", ""),
                    "siblings": sibs, "split_pieces": n_joints,
                    "pattern_total_sf": round(prim_sf + sum(s["area_sf"] for s in sibs), 1),
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
