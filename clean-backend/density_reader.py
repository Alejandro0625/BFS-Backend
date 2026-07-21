"""DENSITY READER (suggestions source) — dense-microtexture pages (Callahan class:
materials drawn as thousands of hairline grain/shingle strokes that every classic
reader rejects). Ported from the proven probe ladder (v9r: 19/114 covered standalone,
22/114 in ensemble on 26-171 p20). Pieces are served as suggest_only — the estimator
accepts real walls with one click; SF never counts until accepted.

API: is_dense_page(pg) -> bool  ·  suggest_pieces(pdf_bytes, page_index, W, H) ->
[{points, area_sf, cx, cy, ...}] (normalized coords, caller tags suggest_only)."""
import fitz
import numpy as np
import cv2

CELL = 4.0          # pt (~1ft at 1/8"=1')
FT_PT = 8.0 / 72.0  # default scale; SF is advisory until the estimator accepts


def is_dense_page(pg):
    """Census-derived signature: >=40k hairline strokes and hairline-dominant."""
    lines = hair = 0
    try:
        for d in pg.get_drawings():
            w = d.get("width") or 0
            for it in d.get("items") or []:
                if it[0] == "l":
                    lines += 1
                    if w <= 0.6:
                        hair += 1
            if lines > 200000:
                break
    except Exception:
        return False
    return lines >= 40000 and hair >= 0.8 * max(lines, 1)


def suggest_pieces(pdf_bytes, page_index, W, H, max_new=40):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        pg = doc[page_index]
        rot = pg.rotation_matrix
        gw, gh = int(W / CELL) + 1, int(H / CELL) + 1
        V = np.zeros((gh, gw), np.float32)
        F = np.zeros((gh, gw), np.float32)
        D = np.zeros((gh, gw), np.float32)
        Hh = np.zeros((gh, gw), np.float32)
        for d in pg.get_drawings():
            wdt = d.get("width") or 0
            if wdt > 0.6:
                continue
            for it in d.get("items") or []:
                if it[0] == "re" and d.get("fill"):
                    r0 = it[1]
                    pa = fitz.Point(r0.x0, r0.y0) * rot
                    pb = fitz.Point(r0.x1, r0.y1) * rot
                    tmp = np.zeros((gh, gw), np.uint8)
                    cv2.rectangle(tmp, (int(min(pa.x, pb.x) / CELL), int(min(pa.y, pb.y) / CELL)),
                                  (int(max(pa.x, pb.x) / CELL), int(max(pa.y, pb.y) / CELL)), 1, -1)
                    F += tmp
                    continue
                if it[0] != "l":
                    continue
                p0 = fitz.Point(it[1]) * rot
                p1 = fitz.Point(it[2]) * rot
                dx, dy = abs(p1.x - p0.x), abs(p1.y - p0.y)
                if max(dx, dy) < 2:
                    continue
                a = (int(p0.x / CELL), int(p0.y / CELL))
                b = (int(p1.x / CELL), int(p1.y / CELL))
                tgt = V if dy > 4 * max(dx, 0.1) else (Hh if dx > 4 * max(dy, 0.1) else D)
                tmp = np.zeros((gh, gw), np.uint8)
                cv2.line(tmp, a, b, 1, 1)
                tgt += tmp
        # story datums for band cuts
        lv = []
        for w9 in pg.get_text("words") or []:
            t = (w9[4] or "").strip().upper()
            if t == "LEVEL" or t.startswith("T.O") or t == "ROOF":
                p = fitz.Point((w9[0] + w9[2]) / 2, (w9[1] + w9[3]) / 2) * rot
                lv.append(p.y)
        levels = []
        for y in sorted(lv):
            if not levels or y - levels[-1] > 6:
                levels.append(y)
    finally:
        doc.close()
    cls = np.zeros((gh, gw), np.uint8)
    stk = np.stack([V, D, Hh, F])
    mx = stk.max(0)
    am = stk.argmax(0)
    cls[(mx >= 2)] = (am + 1)[(mx >= 2)]
    cell_sf = (CELL * FT_PT) ** 2
    # view cuts at big vertical whitespace gaps
    occ = ((V + D + Hh + F) > 0).sum(axis=0)
    viewcuts = []
    run = 0
    for x in range(gw):
        if occ[x] == 0:
            run += 1
        else:
            if run >= 8:
                viewcuts.append(x - run // 2)
            run = 0
    out = []
    for c in (1, 2, 3, 4):
        m0 = (cls == c).astype(np.uint8)
        m0 = cv2.morphologyEx(m0, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        m0 &= ((V + D + Hh) > 0).astype(np.uint8)
        for y in levels:
            cy = int(y / CELL)
            if 0 < cy < gh:
                m0[cy, :] = 0
        for xc in viewcuts:
            m0[:, xc] = 0
        n, lab = cv2.connectedComponents(m0)
        for L in range(1, n):
            m = (lab == L)
            a_full = m.sum() * cell_sf
            if a_full < 40:
                continue
            a_core = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).sum() * cell_sf
            sf = (a_full + a_core) / 2.0 if a_core > 0 else a_full * 0.72
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cs:
                continue
            cbig = max(cs, key=cv2.contourArea)
            ap = cv2.approxPolyDP(cbig, 0.02 * cv2.arcLength(cbig, True), True).reshape(-1, 2)
            if len(ap) < 3:
                continue
            norm = [[round(float(px) * CELL / W, 5), round(float(py) * CELL / H, 5)] for px, py in ap]
            cx = round(sum(q[0] for q in norm) / len(norm), 5)
            cy = round(sum(q[1] for q in norm) / len(norm), 5)
            out.append({"points": norm, "area_sf": round(sf, 1), "cx": cx, "cy": cy,
                        "fill_color": [0.16, 0.75, 0.75], "source": "vector",
                        "holes": [], "label": f"~{round(sf):,} SF"})
            if len(out) >= max_new:
                return out
    return out
