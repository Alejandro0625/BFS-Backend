"""
Texture-based auto-detection for CLEAN (unmarked) elevation drawings.
Classical CV (no ML): smooth grey + vertical seams = metal panel; dense fine grid = masonry/lap.
Runs at full render resolution. Outputs candidate polygons the estimator confirms (assist, not trusted SF).
"""
import numpy as np
import cv2
import fitz

# category -> overlay color (RGB 0-1, like Bluebeam fills) so the frontend groups them
CAT_COLOR = {
    "Metal Wall Panel (auto)": [0.0, 0.80, 0.90],
    "Masonry / Lap (auto)":    [0.95, 0.45, 0.55],
}

def _fill_holes(mask):
    m = mask.astype(np.uint8)
    h, w = m.shape
    ff = m.copy()
    pad = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, pad, (0, 0), 1)   # flood exterior background -> 1
    holes = (ff == 0)                    # enclosed background = holes
    return (m.astype(bool) | holes).astype(np.uint8)

def _contours_to_polys(mask, W, H, category, ft_per_in, render_dpi, min_frac=0.004):
    polys = []
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area_px = cv2.contourArea(c)
        if area_px < min_frac * W * H:
            continue
        eps = 0.01 * cv2.arcLength(c, True)
        ap = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(ap) < 3:
            continue
        sf = area_px * (ft_per_in / render_dpi) ** 2
        pts = [[round(float(x) / W, 5), round(float(y) / H, 5)] for x, y in ap]
        cx = round(float(np.mean([p[0] for p in pts])), 5)
        cy = round(float(np.mean([p[1] for p in pts])), 5)
        polys.append({"points": pts, "area_sf": round(sf, 1), "cx": cx, "cy": cy,
                      "category": category, "material": category,
                      "fill_color": CAT_COLOR.get(category), "source": "texture",
                      "label": f"~{round(sf):,} SF (auto)"})
    return polys

def detect(pdf_bytes, page_index, ft_per_in=8.0, zoom=2.0):
    """Render the page and return candidate material polygons by texture."""
    render_dpi = 72 * zoom
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pix = doc[page_index].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    doc.close()
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    H, W = g.shape

    # local edge-density (fine brick grid survives at full res)
    edges = cv2.Canny(g, 40, 120).astype(np.float32)
    k = max(21, (W // 80) | 1)
    dens = cv2.boxFilter(edges, -1, (k, k)); dens = dens / (dens.max() + 1e-6) * 255

    # building footprint (enclose linework, keep big blobs)
    dark = (g < 165).astype(np.uint8)
    foot = cv2.morphologyEx(cv2.dilate(dark, np.ones((9, 9), np.uint8), 1), cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))
    foot = _fill_holes(foot)
    nn, lab, stats, _ = cv2.connectedComponentsWithStats(foot, 8)
    keep = np.zeros_like(foot)
    for i in range(1, nn):
        if stats[i, cv2.CC_STAT_AREA] > 0.01 * foot.size:
            keep[lab == i] = 1
    foot = keep.astype(bool)
    foot[int(0.95 * H):, :] = False  # drop the ground band

    def tidy(m, op=9, cl=41):
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((op, op), np.uint8))
        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((cl, cl), np.uint8)).astype(bool)

    panel = tidy(foot & (g > 178) & (g < 246) & (dens < 45))   # smooth grey fill (seams allowed)
    masonry = tidy(foot & (dens > 118)) & ~panel                # dense fine grid only
    # keep only large smooth panel blocks
    pn, pl, ps, _ = cv2.connectedComponentsWithStats(panel.astype(np.uint8), 8)
    panel = np.zeros_like(panel)
    for i in range(1, pn):
        if ps[i, cv2.CC_STAT_AREA] > 0.003 * panel.size:
            panel[pl == i] = True

    polys = []
    polys += _contours_to_polys(panel, W, H, "Metal Wall Panel (auto)", ft_per_in, render_dpi)
    polys += _contours_to_polys(masonry, W, H, "Masonry / Lap (auto)", ft_per_in, render_dpi)
    for i, p in enumerate(polys):
        p["id"] = i
    return polys, pix.width, pix.height
