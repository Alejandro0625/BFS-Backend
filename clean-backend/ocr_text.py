"""OCR fallback for FLATTENED drawing sets — sheets whose lettering was exported as vector
curves, so PyMuPDF extracts no text. RapidOCR (onnxruntime, NO torch) reads the rendered
image. Used ONLY when a page has no base text AND we found cladding but couldn't name it.

Memory-safe by design (the EasyOCR/torch OOM is why we do NOT use it):
  * lazy: the OCR engine loads only when a flattened page actually needs it;
  * moderate render DPI + one page at a time, image freed immediately;
  * graceful: if RapidOCR isn't installed or errors, callers get [] and behave as before.
"""
import re
import fitz

_ENGINE = None
_TRIED = False
_ERR = ""

# CLADDING head-nouns only (OCR reads whole sentences, so exclude trim/fascia/hardware/doors)
_CLAD_RE = re.compile(r"\b(metl[-\s]?span|standing\s*seam|board\s*&?\s*batten|"
                      r"panel|siding|lap|shake|shingle|clapboard|brick|masonry|stone|"
                      r"stucco|eifs|acm|composite\s*panel|fiber\s*cement|cementitious|"
                      r"hardie|woodtone|nichiha|cladding|veneer)\b", re.I)
# lines that are construction NOTES, not material names — reject even if a material word appears
_NOTE_RE = re.compile(r"\b(existing|building|roof|installed|install|opening|provide|refer|"
                      r"hardware|door|frp|window|frame|fascia|soffit|flashing|reattach|"
                      r"remove|clean|paint|new|sleeper|ladder|typ\b|see\b|per\b|color\s*-)\b", re.I)


def available():
    """Lazy-load RapidOCR. If not pip-installed, extract the VENDORED package (shipped in the
    repo as rapidocr_vendor.zip) — zero pip resolution, so the opencv-python conflict that
    once took the backend down is structurally impossible."""
    global _ENGINE, _TRIED, _ERR
    if _ENGINE is None and not _TRIED:
        _TRIED = True
        try:
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError:
                import sys, zipfile, tempfile, os as _os
                here = _os.path.dirname(_os.path.abspath(__file__))
                vz = _os.path.join(here, "rapidocr_vendor.zip")
                if not _os.path.isfile(vz):
                    _ERR = f"vendor zip missing at {vz}; dir has: {sorted(_os.listdir(here))[:12]}"
                    _ENGINE = None
                    return False
                dest = _os.path.join(tempfile.gettempdir(), "bfs_rapidocr")
                _os.makedirs(dest, exist_ok=True)
                with zipfile.ZipFile(vz) as z:
                    z.extractall(dest)
                if dest not in sys.path:
                    sys.path.insert(0, dest)
                import importlib
                importlib.invalidate_caches()
                try:
                    from rapidocr_onnxruntime import RapidOCR
                except ImportError as e2:
                    _ERR = f"post-extract import failed: {e2} | dest listing: {sorted(_os.listdir(dest))[:8]}"
                    _ENGINE = None
                    return False
            _ENGINE = RapidOCR()
        except Exception as e:
            _ERR = f"{type(e).__name__}: {e}"[:220]
            _ENGINE = None
    return _ENGINE is not None


def last_error():
    return _ERR


def read_materials(pdf_bytes, page_index, long_side=2200, max_lines=120):
    """Return a deduped list of MATERIAL spec lines OCR'd off one flattened page:
    [{text, conf}]. Empty if OCR unavailable or nothing material-like is read."""
    if not available():
        return []
    try:
        import numpy as np
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pg = doc[page_index]
        W, H = pg.rect.width, pg.rect.height
        z = min(3.0, long_side / max(1, max(W, H)))
        pix = pg.get_pixmap(matrix=fitz.Matrix(z, z), alpha=False)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            import cv2
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        doc.close()
        res, _ = _ENGINE(img)
        del img
        out = []
        seen = set()
        for row in (res or [])[:max_lines]:
            txt = (row[1] or "").strip()
            try:
                conf = float(row[2])
            except Exception:
                conf = 0.0
            if conf < 0.6 or not (5 <= len(txt) <= 60):
                continue
            if not _CLAD_RE.search(txt) or _NOTE_RE.search(txt):
                continue
            if re.match(r"^\s*\d", txt) or re.search(r"\bR\.?O\.?\b|\d-?panel", txt, re.I):
                continue                                        # "2-PANEL", "12x52 R.O." = window callout
            nm = re.sub(r"\s+", " ", txt).strip(' .,;:-"')
            key = re.sub(r"[^A-Z]", "", nm.upper())[:12]     # fuzzy key so OCR variants dedupe
            if key in seen:
                continue
            seen.add(key)
            out.append({"text": nm, "conf": round(conf, 2)})
        out.sort(key=lambda m: -m["conf"])
        return out[:8]
    except Exception:
        return []
