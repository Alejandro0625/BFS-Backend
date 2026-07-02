"""Within-job MATERIAL GROUPING (the estimator's vision + the validated within-job approach):
tile the elevation, read each tile's texture, cluster same-texture tiles → material groups. The
estimator clicks a group → the whole material selects. She confirms; the SF is refined to the exact
boundary (openings subtracted) elsewhere — this module SUGGESTS the grouping + a rough SF, always
flagged approx. Additive; does not touch digitize-markup.

Honest: within-job texture purity measured 62-83% on real multi-material jobs → a strong first-pass
SELECTION, not a trusted SF. `approx_sf` is labeled approximate; the money number comes from the
confirmed/snapped boundary."""
import re
import numpy as np
import cv2
import fitz
import texture

RENDER_LS = 1800
PATCH = 40
DARK = 170


_KEEP_RE = re.compile(r'\b(elevation|soffit|return)\b', re.I)
_SKIP_RE = re.compile(r'\b(floor plan|roof plan|site plan|detail|section|schedule|keynote|legend|door|window)\b', re.I)


def _elev_mask(pg, img, z):
    """Fence to the ELEVATION/SOFFIT/RETURN drawings only — read the blueprint like a person. Keep the
    large building regions BUILT FROM LONG STRUCTURAL LINES (rooflines/floor lines); that's what tells an
    elevation apart from notes, dimension boxes, callouts, and the titleblock (which have no long lines).
    Cut the far-right titleblock strip, and subtract any region a text title marks as plan/detail/
    section/schedule. Returns a HxW keep-mask so clustering never groups 'other bullshit' as a material."""
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ink = (gray < DARK).astype(np.uint8)
    # long structural lines (buildings have them; text/notes/boxes don't)
    longm = np.zeros((H, W), np.uint8)
    try:
        for d in pg.get_drawings():
            for it in d.get("items", []):
                if it[0] == "l":
                    p1, p2 = it[1], it[2]
                    if ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5 >= 70:
                        cv2.line(longm, (int(p1.x * z), int(p1.y * z)), (int(p2.x * z), int(p2.y * z)), 1, 3)
    except Exception:
        pass
    longd = cv2.dilate(longm, np.ones((25, 25), np.uint8))
    # text-block mask + collect plan/detail/section/schedule titles to subtract
    txt = np.zeros((H, W), np.uint8); skip_boxes = []
    try:
        for b in pg.get_text("blocks"):
            x0, y0, x1, y1 = [int(v * z) for v in b[:4]]; t = (b[4] or "").strip().replace("\n", " ")
            cv2.rectangle(txt, (max(0, x0 - 2), max(0, y0 - 2)), (min(W, x1 + 2), min(H, y1 + 2)), 1, -1)
            if len(t) < 50 and _SKIP_RE.search(t) and not _KEEP_RE.search(t):
                skip_boxes.append((x0, y0, x1, y1))
    except Exception:
        pass
    draw = ink & (1 - txt)
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(draw, np.ones((17, 17), np.uint8)), 8)
    keep = np.zeros((H, W), np.uint8)
    for i in range(1, n):
        area = st[i, cv2.CC_STAT_AREA]; cw = st[i, cv2.CC_STAT_WIDTH]; cx0 = st[i, cv2.CC_STAT_LEFT]
        if area < 0.03 * H * W:                              # drop small stuff (dims/callouts/boxes/text)
            continue
        if (cx0 + cw / 2) > 0.80 * W and cw < 0.22 * W:      # drop far-right titleblock strip
            continue
        comp = (lab == i).astype(np.uint8)
        if (comp & longd).sum() < 0.20 * comp.sum():         # must be built from long structural lines = a real elevation
            continue
        keep |= comp
    for (x0, y0, x1, y1) in skip_boxes:                      # subtract plan/detail/section/schedule view areas
        vw = max(x1 - x0, int(0.12 * W))
        cv2.rectangle(keep, (max(0, (x0 + x1) // 2 - vw), max(0, y0 - int(0.35 * H))),
                      (min(W, (x0 + x1) // 2 + vw), min(H, y1 + int(0.05 * H))), 0, -1)
    if keep.sum() < 0.02 * H * W:                            # fence found nothing → fail-open (don't return empty)
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
            if edge < 0.01 and g.mean() > 245:    # truly blank PAPER only — keep smooth GREY panel fill
                continue                          # (metal/ACM renders as solid grey tone ~190-240, no edges;
                                                  #  the old edge-only skip dropped BFS's core material from the preview)
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
