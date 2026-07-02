"""Within-job MATERIAL GROUPING (the estimator's vision + the validated within-job approach):
tile the elevation, read each tile's texture, cluster same-texture tiles → material groups. The
estimator clicks a group → the whole material selects. She confirms; the SF is refined to the exact
boundary (openings subtracted) elsewhere — this module SUGGESTS the grouping + a rough SF, always
flagged approx. Additive; does not touch digitize-markup.

Honest: within-job texture purity measured 62-83% on real multi-material jobs → a strong first-pass
SELECTION, not a trusted SF. `approx_sf` is labeled approximate; the money number comes from the
confirmed/snapped boundary."""
import numpy as np
import cv2
import fitz
import texture

RENDER_LS = 1800
PATCH = 40
DARK = 170


def _elev_mask(pg, img, z):
    """Fence to the ELEVATION drawings: mask out TEXT regions (notes/titleblock/dimensions via the PDF
    text layer) then keep the LARGE connected blobs of drawing ink. Returns a HxW uint8 keep-mask so
    clustering never groups the titleblock/notes as 'materials'."""
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ink = (gray < DARK).astype(np.uint8)
    txt = np.zeros((H, W), np.uint8)
    try:
        for b in pg.get_text("blocks"):
            x0, y0, x1, y1 = [int(v * z) for v in b[:4]]
            cv2.rectangle(txt, (max(0, x0 - 2), max(0, y0 - 2)), (min(W, x1 + 2), min(H, y1 + 2)), 1, -1)
    except Exception:
        pass
    draw = ink & (1 - txt)
    dil = cv2.dilate(draw, np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(dil, 8)
    keep = np.zeros((H, W), np.uint8)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] > 0.02 * H * W:
            keep[lab == i] = 1
    if keep.sum() < 0.02 * H * W:        # fence found nothing → don't over-filter, allow whole sheet
        keep[:] = 1
    return keep


def groups(pdf_bytes, page_index, k=None):
    """Return material groups for a page: [{group, approx_sf, patches:[[nx,ny,size_nx,size_ny]...],
    color}], each a texture cluster the estimator can select in one click."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    sc, conf = texture._read_scale(doc, page_index)
    pg = doc[page_index]
    z = RENDER_LS / max(pg.rect.width, pg.rect.height)
    pix = pg.get_pixmap(matrix=fitz.Matrix(z, z))
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif pix.n == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = img[:, :, :3]
    H, W = img.shape[:2]
    keep = _elev_mask(pg, img, z)                 # fence to the elevations WHILE doc is open (needs text layer)
    doc.close()
    ft_per_px = sc / (72 * z)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0)); gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    feats = []; cells = []
    for y in range(0, H - PATCH, PATCH):
        for x in range(0, W - PATCH, PATCH):
            if keep[y:y + PATCH, x:x + PATCH].mean() < 0.5:   # only patches inside the elevation drawings
                continue
            g = gray[y:y + PATCH, x:x + PATCH]
            edge = (g < DARK).mean()
            if edge < 0.01:                       # blank whitespace = no material
                continue
            exx = gx[y:y + PATCH, x:x + PATCH]; eyy = gy[y:y + PATCH, x:x + PATCH]
            hr = float(exx.mean() / (exx.mean() + eyy.mean() + 1e-3))
            feats.append([edge * 4, hr, g.mean() / 255.0, eyy.mean() / 50.0])
            cells.append((x, y))
    if len(feats) < 8:
        return {"groups": [], "scale_confirmed": conf, "note": "too little drawing content"}
    feats = np.array(feats, np.float32)
    K = k or max(2, min(6, int(round(np.sqrt(len(feats) / 12))) + 1))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, lab, _ = cv2.kmeans(feats, K, None, crit, 5, cv2.KMEANS_PP_CENTERS)
    lab = lab.flatten()
    PAL = [[0.90, 0.24, 0.24], [0.24, 0.63, 0.90], [0.27, 0.78, 0.35], [0.78, 0.47, 0.94],
           [0.94, 0.67, 0.16], [0.59, 0.59, 0.63]]
    out = []
    pnx, pny = PATCH / W, PATCH / H
    for c in range(K):
        idx = [i for i in range(len(lab)) if lab[i] == c]
        if not idx:
            continue
        patches = [[round(cells[i][0] / W, 5), round(cells[i][1] / H, 5), round(pnx, 5), round(pny, 5)] for i in idx]
        approx_sf = round(len(idx) * (PATCH * PATCH) * ft_per_px ** 2, 0)
        out.append({"group": c, "approx_sf": approx_sf, "n_patches": len(idx),
                    "color": PAL[c % len(PAL)], "patches": patches})
    out.sort(key=lambda g: -g["approx_sf"])
    return {"groups": out, "scale_confirmed": conf, "width": pix.width, "height": pix.height,
            "note": "approx SF from texture patches — confirm boundary for exact SF"}
