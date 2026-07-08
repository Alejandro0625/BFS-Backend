"""
Texture GROUPING for clean (unmarked) elevation drawings — classical CV, no ML.
Groups the facade by texture: smooth grey (panel-type) in one group, dense grid (brick/lap) in another.
Window/door openings are a different texture -> not in either rule-mask -> excluded from the pixel count,
so each group's SF is NET (openings cut out). Estimator selects a group -> net SF of that texture.
"""
import re
import numpy as np
import cv2
import fitz

# Neutral group names — the estimator renames them in the app (flows to the Excel).
# G1 smooth (panel), G2 looser texture (lap/FC), G3 dense texture (brick) — split so the
# estimator selects only the materials in their scope (e.g. skip brick).
GROUPS = [
    ("Group 1", [0.0, 0.80, 0.90]),   # cyan  — smooth / panel-type
    ("Group 2", [0.35, 0.78, 0.45]),  # green — looser texture (lap / fiber-cement)
    ("Group 3", [0.95, 0.45, 0.55]),  # pink  — dense texture (brick / masonry)
]


def _parse_scale_text(txt):
    """Extract candidate ft/in values from scale-notation text. Shared by the page-text,
    annotation and OCR paths. (?<!\\d ) skips the fraction of a MIXED detail number like
    `1 1/2\"=1'` — a detail scale, not 2.0 as the bare \"1/2\" would parse."""
    t = txt.replace("’", "'").replace("‘", "'").replace("”", '"').replace("“", '"').replace("″", '"').replace("′", "'")
    # OCR quirk: the '1' of \"= 1'-0\"\" reads as I/l/| (Fleet: 'SCALE: 1/8\" = I'-0\"')
    t = re.sub(r"=\s*[Il|]\s*'", "= 1'", t)
    # architectural fraction:  A/B" = 1'  (1/8"=1'-0", 3/16"=1'-0")  -> ft_per_in = B/A
    arch = []
    for m in re.finditer(r"(?<!\d )(\d+)\s*/\s*(\d+)\s*\"?\s*(?:in\.?)?\s*=\s*1\s*'", t, re.I):
        a, b = int(m.group(1)), int(m.group(2))
        if a > 0:
            arch.append(b / a)
    # engineering:  1" = N'   -> ft_per_in = N  (site/civil)
    eng = []
    for m in re.finditer(r"\b1\s*\"\s*=\s*(\d+)\s*'", t, re.I):
        n = int(m.group(1))
        if n > 0:
            eng.append(float(n))
    return arch or eng


def _consensus(cands):
    # confirmed only if UNAMBIGUOUS: one value, or clearly dominant — a wrong scale
    # SQUARES into the SF (a 1/4" read as 1/8" = 4x the money)
    from collections import Counter
    cnt = Counter(round(c, 4) for c in cands); best, bn = cnt.most_common(1)[0]
    others = sum(v for k, v in cnt.items() if k != best)
    confirmed = (len(cnt) == 1) or (bn >= 2 and bn > others)
    return best, confirmed


_SCALE_CACHE = {}


def _read_scale(doc, pi):
    """Read the drawing scale as feet-per-inch, robustly. Returns (ft_per_in, confirmed).
    Source chain — the drawing states its scale three ways, take the first that speaks
    with confidence: (1) page/annot TEXT scale note; (2) the drawing's own DIMENSION
    STRINGS (its built-in ruler — covers sheets whose note lives elsewhere); (3) OCR of
    the title-block strips (flattened sets draw the note as curves). confirmed=False →
    caller must FLAG (never trust a defaulted SF)."""
    txt = ""
    try:
        txt += doc[pi].get_text() or ""
    except Exception:
        pass
    try:
        for a in (doc[pi].annots() or []):
            txt += " " + (a.info.get("content", "") or "")
    except Exception:
        pass
    cands = _parse_scale_text(txt)
    if cands:
        best, confirmed = _consensus(cands)
        if confirmed:
            return best, True
    # cache the expensive fallbacks per page CONTENT (bucket clicks re-enter constantly;
    # content-hash key so two different jobs can never share a cached scale)
    try:
        import hashlib
        key = (pi, hashlib.md5(doc[pi].read_contents()[:4096]).hexdigest())
    except Exception:
        key = None
    if key is not None and key in _SCALE_CACHE:
        return _SCALE_CACHE[key]
    result = None
    # (2) dimension strings: "24'-0\"" drawn over a measurable line IS the scale
    try:
        import dim_scale
        ds = dim_scale.sheet_scale(doc[pi])
        if ds:
            result = (round(ds[0] * 72.0, 4), True)
    except Exception:
        pass
    # (3) OCR the title block (textless/flattened sets)
    if result is None:
        try:
            import ocr_text
            if ocr_text.available():
                oc = _parse_scale_text(ocr_text.read_scale_note(doc[pi]))
                if oc:
                    best, confirmed = _consensus(oc)
                    if confirmed:
                        result = (best, True)
        except Exception:
            pass
    if result is None:
        result = ((_consensus(cands)[0] if cands else 8.0), False)
    if key is not None:
        _SCALE_CACHE[key] = result
        if len(_SCALE_CACHE) > 400:
            _SCALE_CACHE.clear()
    return result


def _fill_holes(mask):
    m = mask.astype(np.uint8)
    h, w = m.shape
    ff = m.copy(); pad = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, pad, (0, 0), 1)
    return (m.astype(bool) | (ff == 0)).astype(np.uint8)


def detect(pdf_bytes, page_index, ft_per_in=8.0, zoom=2.0):
    render_dpi = 72 * zoom
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    sc, sc_confirmed = _read_scale(doc, page_index)
    ft_per_in = sc  # read scale, or 8.0 default (sc_confirmed=False → caller flags it)
    pix = doc[page_index].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    doc.close()
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif pix.n == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    H, W = g.shape

    kd = max(21, (W // 80) | 1)
    edges = cv2.Canny(g, 40, 120).astype(np.float32)
    dens = cv2.boxFilter(edges, -1, (kd, kd)); dens = dens / (dens.max() + 1e-6) * 255
    gf = g.astype(np.float32)
    syy = cv2.boxFilter(np.abs(cv2.Sobel(gf, cv2.CV_32F, 0, 1, 3)), -1, (kd, kd))  # horizontal-line energy

    dark = (g < 165).astype(np.uint8)
    foot = cv2.morphologyEx(cv2.dilate(dark, np.ones((9, 9), np.uint8), 1), cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))
    foot = _fill_holes(foot)
    nn, lab, st, _ = cv2.connectedComponentsWithStats(foot, 8)
    keep = np.zeros_like(foot)
    for i in range(1, nn):
        if st[i, cv2.CC_STAT_AREA] > 0.01 * foot.size:
            keep[lab == i] = 1
    foot = keep.astype(bool)
    foot[int(0.96 * H):, :] = False

    def tidy(m, op=9, cl=41):
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((op, op), np.uint8))
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((cl, cl), np.uint8)).astype(bool)

    smooth = tidy(foot & (g > 180) & (g < 246) & (dens < 45))
    pn, pl, ps, _ = cv2.connectedComponentsWithStats(smooth.astype(np.uint8), 8)
    smooth = np.zeros_like(smooth)
    for i in range(1, pn):
        if ps[i, cv2.CC_STAT_AREA] > 0.003 * smooth.size:
            smooth[pl == i] = True

    # all textured area, then split by LINE DENSITY (looser lap vs denser brick) via adaptive 2-means
    textured = (foot & (dens > 60) & ~smooth)
    tex_lo = np.zeros_like(textured); tex_hi = np.zeros_like(textured)
    ys, xs = np.where(textured)
    if len(xs) > 300:
        vals = dens[ys, xs].reshape(-1, 1).astype(np.float32)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 12, 1.0)
        _, lab, cen = cv2.kmeans(vals, 2, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        lab = lab.flatten(); hi = int(np.argmax(cen.flatten()))
        # only split if the two texture modes are meaningfully apart (else it's one material)
        if abs(cen[0, 0] - cen[1, 0]) > 22:
            tex_lo[ys[lab != hi], xs[lab != hi]] = True
            tex_hi[ys[lab == hi], xs[lab == hi]] = True
        else:
            tex_lo[ys, xs] = True
    else:
        tex_lo[ys, xs] = True
    tex_lo = tidy(tex_lo); tex_hi = tidy(tex_hi) & ~tex_lo

    px2sf = (ft_per_in / render_dpi) ** 2
    polys = []
    pid = 0
    for (name, color), mask in zip(GROUPS, [smooth, tex_lo, tex_hi]):
        col = [round(v, 3) for v in color]
        nc, clab, cst, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
        for i in range(1, nc):
            a = int(cst[i, cv2.CC_STAT_AREA])
            if a < 0.0025 * H * W:
                continue
            cnts, _ = cv2.findContours((clab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            c = max(cnts, key=cv2.contourArea)
            ap = cv2.approxPolyDP(c, 0.008 * cv2.arcLength(c, True), True).reshape(-1, 2)
            if len(ap) < 3:
                continue
            pts = [[round(float(x) / W, 5), round(float(y) / H, 5)] for x, y in ap]
            sf = a * px2sf  # NET pixel count (openings excluded)
            cx = round(float(np.mean([p[0] for p in pts])), 5)
            cy = round(float(np.mean([p[1] for p in pts])), 5)
            polys.append({"id": pid, "points": pts, "area_sf": round(sf, 1), "cx": cx, "cy": cy,
                          "fill_color": col, "source": "texture", "material": name,
                          "category": name, "label": f"~{round(sf):,} SF"})
            pid += 1
    return polys, pix.width, pix.height, {"ft_per_in": round(ft_per_in, 3), "scale_confirmed": sc_confirmed}
