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

GROUPS = [
    ("Smooth zone (panel-type)",   [0.0, 0.80, 0.90]),   # cyan
    ("Textured zone (brick/lap)",  [0.95, 0.45, 0.55]),   # pink/red
]


def _read_scale(doc, pi):
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
    m = re.search(r'(\d+)\s*/\s*(\d+)\s*"?\s*=\s*1\s*\'', txt)
    if m and int(m.group(1)) > 0:
        return int(m.group(2)) / int(m.group(1))
    return None


def _fill_holes(mask):
    m = mask.astype(np.uint8)
    h, w = m.shape
    ff = m.copy(); pad = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, pad, (0, 0), 1)
    return (m.astype(bool) | (ff == 0)).astype(np.uint8)


def detect(pdf_bytes, page_index, ft_per_in=8.0, zoom=2.0):
    render_dpi = 72 * zoom
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    sc = _read_scale(doc, page_index)
    if sc:
        ft_per_in = sc
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
    textured = tidy(foot & (dens > 110)) & ~smooth

    px2sf = (ft_per_in / render_dpi) ** 2
    polys = []
    pid = 0
    for (name, color), mask in zip(GROUPS, [smooth, textured]):
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
    return polys, pix.width, pix.height
