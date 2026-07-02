"""Run the trained MATERIAL model (ONNX) on a RAW/clean drawing → per-material cladding GROUPS.
This is the core of the product: the estimator uploads a blank elevation, the model paints each
material (metal / lap / fiber-cement / ...) as its own colored group, she just SELECTS the group
she's bidding and reads the SF. Her confirmations/corrections feed the flywheel.

Two things make this accurate and safe:
  * TILED inference — the model was trained on 768px tiles rendered at long-side 2560, so we render
    at that same scale and slide a 768 window (full-image resize would wreck accuracy — measured).
  * memory-safe stitch — we keep only a best-confidence label map (2 HxW arrays), never a 12xHxW
    buffer, so a big elevation can't OOM the backend.

Lightweight: onnxruntime-CPU only (no torch). Auto-detects binary (2-class) vs material (12-class)
models and falls back gracefully (returns nothing → caller uses texture) if no model is present."""
import os
import numpy as np
import cv2
import fitz
import texture  # reuse the robust scale reader

MODEL_PATH = os.environ.get("MODEL_ONNX", "/data/model.onnx")
LS = int(os.environ.get("MODEL_LS", "2560"))    # render long-side (matches training)
TILE = 768                                        # tile size (matches training)
STRIDE = 512                                      # 256px overlap → smoother seams
_SESSION = None
_TRIED = False

# class order MUST match the training classes.json
CLASSES = ["background", "Fiber Cement Panel", "Metal Panel", "Lap Siding", "ACM/Composite",
           "Soffit/Trim", "Board & Batten", "Shingle/Shake", "PVC", "Standing Seam",
           "Brick/Masonry", "Other Cladding"]
# distinct display color per material (RGB 0-1) so groups are visually separable in the UI
MAT_COLORS = {
    "Fiber Cement Panel": [0.00, 0.70, 0.75], "Metal Panel": [0.12, 0.45, 0.95],
    "Lap Siding": [0.20, 0.72, 0.30], "ACM/Composite": [0.55, 0.35, 0.85],
    "Soffit/Trim": [0.95, 0.60, 0.10], "Board & Batten": [0.60, 0.40, 0.20],
    "Shingle/Shake": [0.85, 0.25, 0.25], "PVC": [0.90, 0.45, 0.70],
    "Standing Seam": [0.10, 0.80, 0.85], "Brick/Masonry": [0.70, 0.20, 0.15],
    "Other Cladding": [0.50, 0.50, 0.55],
}

def _session():
    global _SESSION, _TRIED
    if _SESSION is None and not _TRIED:
        _TRIED = True
        try:
            if os.path.exists(MODEL_PATH):
                import onnxruntime as ort
                _SESSION = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
        except Exception:
            _SESSION = None
    return _SESSION

def available():
    return _session() is not None

def reset():
    """Force reload of the ONNX session (call after a new model file is uploaded)."""
    global _SESSION, _TRIED
    _SESSION = None; _TRIED = False

def _in_name(sess):
    try:
        return sess.get_inputs()[0].name
    except Exception:
        return "input"

def _tile_probs(sess, iname, tile):
    """One tile (RGB uint8, TILExTILE) → softmax probabilities (NC, TILE, TILE) float32."""
    inp = (tile.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    logits = sess.run(None, {iname: inp})[0][0]          # (NC, TILE, TILE)
    m = logits.max(0, keepdims=True)
    e = np.exp(logits - m)
    return (e / e.sum(0, keepdims=True)).astype(np.float32)

def _infer_tile(sess, iname, tile):
    """One tile → (label, conf, NC). Kept for diagnostics/consistency tests."""
    p = _tile_probs(sess, iname, tile)
    return p.argmax(0).astype(np.uint8), p.max(0), p.shape[0]

def _run_tiled(sess, img):
    """Slide a 768 window over the full render and AVERAGE softmax probabilities across the
    overlaps (NOT best-confidence — background predictions are near-1.0 in whitespace and would
    erase the moderate-confidence cladding predictions, which was measured). Accumulation is done
    at half resolution so a big elevation stays well under ~60MB and can't OOM the backend.
    Returns (labelmap HxW uint8, n_classes)."""
    iname = _in_name(sess)
    H, W = img.shape[:2]
    H2, W2 = (H + 1) // 2, (W + 1) // 2
    nc = None
    psum = None                                          # (NC, H2, W2) running prob sum
    cnt = np.zeros((H2, W2), np.float32)
    ys = list(range(0, max(1, H - TILE + 1), STRIDE)) or [0]
    xs = list(range(0, max(1, W - TILE + 1), STRIDE)) or [0]
    if ys[-1] != max(0, H - TILE): ys.append(max(0, H - TILE))
    if xs[-1] != max(0, W - TILE): xs.append(max(0, W - TILE))
    for y in ys:
        for x in xs:
            th = min(TILE, H - y); tw = min(TILE, W - x)
            tile = np.full((TILE, TILE, 3), 255, np.uint8)      # pad WHITE (drawings are white paper; the model never saw black padding in training)
            tile[:th, :tw] = img[y:y + th, x:x + tw]
            p = _tile_probs(sess, iname, tile)[:, :th, :tw]     # (NC, th, tw)
            if psum is None:
                nc = p.shape[0]; psum = np.zeros((nc, H2, W2), np.float32)
            # place into the half-res accumulator
            hy, hx = y // 2, x // 2
            ph = cv2.resize(np.transpose(p, (1, 2, 0)), (max(1, tw // 2), max(1, th // 2)),
                            interpolation=cv2.INTER_AREA)
            if ph.ndim == 2:  # nc==1 safety
                ph = ph[..., None]
            ph = np.transpose(ph, (2, 0, 1))
            hh, ww = ph.shape[1], ph.shape[2]
            psum[:, hy:hy + hh, hx:hx + ww] += ph
            cnt[hy:hy + hh, hx:hx + ww] += 1
    cnt = np.maximum(cnt, 1e-6)
    lab2 = (psum / cnt).argmax(0).astype(np.uint8)
    labelmap = cv2.resize(lab2, (W, H), interpolation=cv2.INTER_NEAREST)
    return labelmap, nc

def _polys_from_mask(cm, H, W, px2sf, material, min_frac):
    """Connected components of one material's binary mask → list of polygon dicts."""
    cm = cv2.morphologyEx(cm, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cm = cv2.morphologyEx(cm, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    out = []
    nc, lab, st, _ = cv2.connectedComponentsWithStats(cm, 8)
    color = MAT_COLORS.get(material, [0.0, 0.80, 0.90])
    for i in range(1, nc):
        a = int(st[i, cv2.CC_STAT_AREA])
        if a < min_frac * H * W:
            continue
        cnts, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        ap = cv2.approxPolyDP(c, 0.006 * cv2.arcLength(c, True), True).reshape(-1, 2)
        if len(ap) < 3:
            continue
        pts = [[round(float(px) / W, 5), round(float(py) / H, 5)] for px, py in ap]
        sf = a * px2sf
        cx = round(float(np.mean([p[0] for p in pts])), 5)
        cy = round(float(np.mean([p[1] for p in pts])), 5)
        out.append({"points": pts, "area_sf": round(sf, 1), "cx": cx, "cy": cy,
                    "fill_color": color, "source": "model", "material": material,
                    "category": material, "group": material, "label": f"~{round(sf):,} SF"})
    return out

def detect(pdf_bytes, page_index, zoom=None):
    """Returns (polys, width, height, scale_info). polys carry per-material grouping so the
    estimator can select a whole material at once. Empty polys if the model isn't available."""
    sess = _session()
    if sess is None:
        return [], 0, 0, {"ft_per_in": 8.0, "scale_confirmed": False}
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    sc, conf = texture._read_scale(doc, page_index)
    page = doc[page_index]
    pw, ph = page.rect.width, page.rect.height
    z = LS / max(pw, ph)                                   # render at training long-side
    pix = page.get_pixmap(matrix=fitz.Matrix(z, z))
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    doc.close()
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif pix.n == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    H, W = img.shape[:2]

    labelmap, nc = _run_tiled(sess, img)
    render_dpi = 72 * z
    px2sf = (sc / render_dpi) ** 2

    polys = []
    if nc <= 2:
        # binary extent model → single "Cladding (AI)" group
        cm = (labelmap == 1).astype(np.uint8)
        for p in _polys_from_mask(cm, H, W, px2sf, "Cladding (AI)", 0.002):
            p["material"] = "Cladding (AI)"; p["category"] = "Cladding"; p["group"] = "Cladding (AI)"
            p["fill_color"] = [0.0, 0.80, 0.90]
            polys.append(p)
    else:
        for ci in range(1, min(nc, len(CLASSES))):
            cm = (labelmap == ci).astype(np.uint8)
            if int(cm.sum()) < 0.0015 * H * W:
                continue
            polys.extend(_polys_from_mask(cm, H, W, px2sf, CLASSES[ci], 0.0015))

    for i, p in enumerate(polys):
        p["id"] = i
    return polys, pix.width, pix.height, {"ft_per_in": round(sc, 3), "scale_confirmed": conf}
