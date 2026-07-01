"""Run the trained cladding-EXTENT model (ONNX) on a RAW/clean drawing → suggested cladding regions.
Lightweight: onnxruntime-CPU only (no torch → no OOM). The estimator confirms/names each region; her
corrections feed the flywheel. This is the auto-markup the whole product is for. Falls back gracefully
(returns nothing → caller uses texture) if the model file isn't present."""
import os
import numpy as np
import cv2
import fitz
import texture  # reuse the robust scale reader

MODEL_PATH = os.environ.get("MODEL_ONNX", "/data/model.onnx")
SIZE = 896
_SESSION = None
_TRIED = False

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

def detect(pdf_bytes, page_index, zoom=2.0):
    """Returns (polys, width, height, scale_info). Empty polys if the model isn't available."""
    sess = _session()
    if sess is None:
        return [], 0, 0, {"ft_per_in": 8.0, "scale_confirmed": False}
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    sc, conf = texture._read_scale(doc, page_index)
    pix = doc[page_index].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
    doc.close()
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif pix.n == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    H, W = img.shape[:2]
    inp = cv2.resize(img, (SIZE, SIZE)).astype(np.float32) / 255.0     # match training preprocessing (/255, no mean/std)
    inp = np.transpose(inp, (2, 0, 1))[None]                            # 1x3xSIZExSIZE
    logits = sess.run(None, {"input": inp})[0]                          # 1x2xSIZExSIZE
    mask_s = (logits[0].argmax(0) == 1).astype(np.uint8)               # cladding = class 1
    mask = cv2.resize(mask_s, (W, H), interpolation=cv2.INTER_NEAREST)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))

    render_dpi = 72 * zoom
    px2sf = (sc / render_dpi) ** 2
    polys = []; pid = 0
    nc, lab, st, _ = cv2.connectedComponentsWithStats(mask, 8)
    for i in range(1, nc):
        a = int(st[i, cv2.CC_STAT_AREA])
        if a < 0.002 * H * W:
            continue
        cnts, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        ap = cv2.approxPolyDP(c, 0.008 * cv2.arcLength(c, True), True).reshape(-1, 2)
        if len(ap) < 3:
            continue
        pts = [[round(float(x) / W, 5), round(float(y) / H, 5)] for x, y in ap]
        sf = a * px2sf
        cx = round(float(np.mean([p[0] for p in pts])), 5)
        cy = round(float(np.mean([p[1] for p in pts])), 5)
        polys.append({"id": pid, "points": pts, "area_sf": round(sf, 1), "cx": cx, "cy": cy,
                      "fill_color": [0.0, 0.80, 0.90], "source": "model", "material": "Cladding (AI)",
                      "category": "Cladding", "label": f"~{round(sf):,} SF"})
        pid += 1
    return polys, pix.width, pix.height, {"ft_per_in": round(sc, 3), "scale_confirmed": conf}
