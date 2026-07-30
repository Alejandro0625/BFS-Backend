"""Title-block OCR reader — sheet number + drawing title for pages whose title block is
DRAWN (curves/raster) instead of text. The 2026-07-29 flattened re-audit (flattened_audit
.json) proved the 15 page-finding-blind jobs are NOT textless: their gold pages carry
100-3,600 chars of body text while the title block itself extracts nothing — so text-layer
page classification starves exactly where the sheet identity lives only in pixels.

Prototype provenance (ocr_dpi_test.py/2/3, graded on real job 26-222, 60 pages):
  * corner crop (right 12% x bottom 28%) at zoom 2.5 read the sheet number on 60/60 pages,
    ELEVATION-titled sheets 9/9, 0 false positives, ~3.1 s/page CPU;
  * zoom is a SWEET SPOT, not a maximum: at z6-8 the DB detector FRAGMENTED the sheet
    number ("A9.51b" -> "A9" + ".51b") then dropped it — do NOT "improve" this pass by
    raising DPI past ~300 (z4.0); tile instead if micro-text is ever needed;
  * vector-rendered CAD text needs no binarization/deskew — raw RGB renders score 0.8-0.9.

Zero-confidently-wrong contract: a sheet identity below CONF_MIN is never bound —
read_sheet_id returns None (callers flag, never guess). A wrong page binding would
propagate to takeoff; abstention only costs a flag.

Uses the vendored RapidOCR session from ocr_text (zero new dependencies, lazy load).
"""
import os
import re
import time

import fitz

import ocr_text

# ---- frozen parameters (prototype-measured; see module docstring before changing) ----
CONF_MIN = 0.60          # bind gate: sheet/title below this -> abstain (flag, never bind)
ROW_MIN = 0.50           # rows below this are OCR noise — never considered
ZOOM_PRIMARY = 2.5       # 180 DPI — the measured sweet spot for title-block text
ZOOM_RETRY = 4.0         # 288 DPI — retry ceiling; NEVER exceed (fragmentation is real)
CORNER = (0.88, 0.72)    # right 12% x bottom 28% (prototype crop, 60/60 sheet reads)
BOTTOM_STRIP = 0.86      # fallback: bottom 14% full width (bottom-center title blocks)
RIGHT_STRIP = 0.955      # fallback: right 4.5% full height (vertical title blocks)
TIME_BUDGET_S = float(os.environ.get("TITLEBLOCK_PAGE_BUDGET_S", "8.0"))

# sheet-number shapes: A9.51b / A1.01a (dot form, prototype-graded on 26-222),
# A-402 / A-200 (dash form), A301 / A103 (AIA no-separator form — standard US
# discipline-letter + 3-digit convention; present across the archive: A301, A201, A103).
SHEET_RE = re.compile(
    r"^[A-Z]{1,2}\d{1,2}\.\d{1,2}[A-Za-z]?$"   # A9.51b, A4.6D
    r"|^[A-Z]{1,2}-\d{2,3}[A-Za-z]?$"          # A-402, A-310
    r"|^[A-Z]{1,2}\d{3}[A-Za-z]?$"             # A301, A201
)
# elevation-series sheet number (A4xx family: A4.x, A-4xx, A4xx) — SECONDARY candidacy
# signal only; series conventions vary by architect, so this may admit enlarged-plan /
# detail sheets in some sets. Candidacy != confident identity: admitted pages still go
# through wall detection; nothing binds on series alone.
A4_SERIES_RE = re.compile(r"^A-?4(\d|\.)")
ELEV_RE = re.compile(r"ELEVATION", re.I)
# reject note-shaped rows that merely MENTION elevations ("SEE ELEVATION FOR COLOR"):
# corner-crop geometry already excludes body notes, but a note can drift into the crop
# on non-standard layouts — a title row is short and title-shaped.
_NOTE_RE = re.compile(r"\b(SEE|REFER|PER|FOR|TYP)\b", re.I)
_INTERIOR_RE = re.compile(r"\bINTERIOR\b", re.I)

_cache = {}              # (id(doc), page_number) -> result | None
_CACHE_MAX = 4096
_census_cache = {}       # id(doc) -> {page_number: SHEET} from the TEXT layer

# SCHEME-CONSISTENCY VALIDITY GATE (directive item #11 arriving early as a validity
# check, 2026-07-29): before asserting a sheet ID, compare its numbering FAMILY (prefix
# letters + separator form) against the document's own scheme — the other pages'
# text-layer IDs plus this session's confident OCR reads. A family that appears NOWHERE
# else in a document that otherwise shows a real scheme is conf-capped to flag-only
# (scheme_mismatch=True, never asserted). FAIL-OPEN: fewer than _SCHEME_MIN_EVIDENCE
# other-page IDs (e.g. fully flattened sets) -> no gate. Flag-only cost: a legitimate
# one-of-a-kind sheet (the single P sheet in an A set) gets flagged, never mis-bound.
_SCHEME_MIN_EVIDENCE = 4
_SCHEME_CAP = 0.59       # just below CONF_MIN -> flag-never-bind

# title-block field label ("DRAWING NUMBER", "SHEET NO.") — anchors the true ID box.
# Spec paragraphs can carry sheet-shaped product tokens (real 26-169 p65 trap: the
# bottom-most-token rule read "DEARBORN BC-45 CLEANER"'s BC-45 as the sheet while the
# DRAWING NUMBER box says P-003) — the token NEAREST a label outranks the bottom-most.
_LABEL_RE = re.compile(r"^(DRAWING|SHEET|DWG)\.?\s*(NUMBER|NO)?[.:]*$", re.I)


def sheet_family(sheet):
    """Numbering family of a sheet ID: (prefix letters, separator form) or None."""
    m = re.match(r"^([A-Z]{1,2})([-.]?)\d", sheet or "")
    if not m:
        return None
    form = "dash" if m.group(2) == "-" else ("dot" if "." in sheet else "plain")
    return (m.group(1), form)


def text_sheet_of_page(pg):
    """TEXT-LAYER sheet ID of a page, label-anchored: the sheet-shaped token nearest a
    DRAWING/SHEET NUMBER label in the title-block region; falls back to the bottom-most
    token (coverage.py's legacy rule) when no label is readable. None on curve-title
    pages. Also the coverage.py integration point: prefer this, then read_sheet_id."""
    try:
        W, H = pg.rect.width, pg.rect.height
        rot = pg.rotation_matrix
        toks = []
        for w in pg.get_text("words"):
            q = fitz.Point((w[0] + w[2]) / 2, (w[1] + w[3]) / 2) * rot
            if q.x >= W * 0.82 and q.y >= H * 0.80:
                toks.append((q.y, q.x, (w[4] or "").strip()))
    except Exception:
        return None
    toks.sort()
    sheets = [(y, x, t.upper()) for (y, x, t) in toks if SHEET_RE.match(t.upper())]
    if not sheets:
        return None
    labels = [(y, x) for (y, x, t) in toks if _LABEL_RE.match(t)]
    best, bd = None, 1e18
    for (ly, lx) in labels:
        for (sy, sx, s) in sheets:
            d = (sy - ly) ** 2 + (sx - lx) ** 2
            if abs(sy - ly) <= 120 and d < bd:
                best, bd = s, d
    return best or sheets[-1][2]              # label-anchored, else bottom-most (legacy)


def _doc_census(doc):
    """{page_number: text-layer sheet ID} for the whole doc, cached per doc id."""
    key = id(doc)
    if key in _census_cache:
        return _census_cache[key]
    if len(_census_cache) > 64:
        _census_cache.clear()
    census = {}
    try:
        for i in range(doc.page_count):
            s = text_sheet_of_page(doc[i])
            if s:
                census[i] = s
    except Exception:
        pass
    _census_cache[key] = census
    return census


def scheme_consistent(sheet, other_ids):
    """True unless the doc shows a real scheme (>= _SCHEME_MIN_EVIDENCE other-page IDs)
    in which the candidate's family appears ZERO times. Fail-open on thin evidence."""
    fam = sheet_family(sheet)
    if fam is None:
        return True
    others = [f for f in (sheet_family(s) for s in other_ids) if f]
    if len(others) < _SCHEME_MIN_EVIDENCE:
        return True
    return fam in others


def _apply_scheme_gate(pg, best):
    """Cap a scheme-inconsistent sheet bind to flag-only. Mutates + returns best."""
    if not best or not best.get("sheet") or best.get("sheet_conf", 0) < CONF_MIN:
        return best
    try:
        doc = pg.parent
        census = dict(_doc_census(doc))
        census.pop(pg.number, None)           # OWN page never testifies (blind framing)
        other = list(census.values())
        dk = id(doc)
        for (cd, cp), r in _cache.items():    # this session's confident OCR reads
            if cd == dk and cp != pg.number and r and r.get("sheet") \
                    and r.get("sheet_conf", 0) >= CONF_MIN and not r.get("scheme_mismatch"):
                other.append(r["sheet"])
        if not scheme_consistent(best["sheet"], other):
            best["scheme_mismatch"] = True
            best["sheet_conf"] = min(best["sheet_conf"], _SCHEME_CAP)
            best["conf"] = round(min(best["conf"], _SCHEME_CAP), 3)
            best["elev_series"] = False       # a capped ID cannot series-admit either
    except Exception:
        pass
    return best


def _crop_rects(W, H):
    """The 3-crop fallback ladder, in prototype order."""
    return [
        ("corner_z2.5", fitz.Rect(W * CORNER[0], H * CORNER[1], W, H), ZOOM_PRIMARY, False),
        ("corner_z4.0", fitz.Rect(W * CORNER[0], H * CORNER[1], W, H), ZOOM_RETRY, False),
        ("bottom_z2.5", fitz.Rect(0, H * BOTTOM_STRIP, W, H), ZOOM_PRIMARY, False),
        ("right_z2.5", fitz.Rect(W * RIGHT_STRIP, 0, W, H), ZOOM_PRIMARY, True),
    ]


def _ocr_rows(pg, rect, zoom, try_rot90):
    """OCR one crop -> [(text, conf)] rows at ROW_MIN+. Empty on any failure."""
    if not ocr_text.available():
        return []
    try:
        import numpy as np
        pix = pg.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect, alpha=False)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            import cv2
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        res, _ = ocr_text._ENGINE(img)
        rows = [((r[1] or "").strip(), float(r[2])) for r in (res or [])
                if r[1] and float(r[2]) >= ROW_MIN]
        if try_rot90 and not rows:
            import cv2
            res, _ = ocr_text._ENGINE(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
            rows = [((r[1] or "").strip(), float(r[2])) for r in (res or [])
                    if r[1] and float(r[2]) >= ROW_MIN]
        del img
        return rows
    except Exception:
        return []


def _parse(rows):
    """(sheet, sheet_conf, title, title_conf, elev) from one crop's OCR rows.
    Sheet = LAST regex-matching row at CONF_MIN+ (the DRAWING NO. box sits at the very
    bottom of the block — prototype rule). Title = the ELEVATION row when present, else
    the longest all-caps-ish row near the sheet row."""
    sheet, sheet_conf = None, 0.0
    for txt, c in rows:
        t = txt.strip()
        if c >= CONF_MIN and SHEET_RE.match(t):
            sheet, sheet_conf = t.upper(), c
    title, title_conf, elev = None, 0.0, False
    for txt, c in rows:
        if c < CONF_MIN or len(txt) < 5:
            continue
        if ELEV_RE.search(txt) and not _NOTE_RE.search(txt) and not _INTERIOR_RE.search(txt):
            if c > title_conf:
                title, title_conf, elev = txt.upper()[:60], c, True
    if title is None:
        # best non-elevation title candidate: longish letter-heavy row (for coverage's
        # sheet-index reconciliation; NOT an elevation signal)
        for txt, c in rows:
            if c >= CONF_MIN and len(txt) >= 8 and sum(ch.isalpha() for ch in txt) >= 6 \
                    and not SHEET_RE.match(txt.strip()):
                if c > title_conf:
                    title, title_conf = txt.upper()[:60], c
    return sheet, sheet_conf, title, title_conf, elev


def read_sheet_id(pg):
    """OCR the title block of one page -> {sheet, title, conf, elev, elev_series, crop}
    or None when nothing clears CONF_MIN (flag-never-bind: an unreadable identity is a
    flag, not a guess). 'conf' = confidence of the sheet bind when sheet is present,
    else of the title read. ~3 s/page typical; TIME_BUDGET_S caps the ladder."""
    key = (id(pg.parent), pg.number)
    if key in _cache:
        return _cache[key]
    if len(_cache) > _CACHE_MAX:
        _cache.clear()
    W, H = pg.rect.width, pg.rect.height
    t0 = time.time()
    # merge across ladder steps: best sheet read and best title read may come from
    # DIFFERENT crops (corner z2.5 often reads the title, z4.0 the small number box)
    b_sheet, b_sc, b_crop = None, 0.0, None
    b_title, b_tc, b_elev = None, 0.0, False
    for name, rect, zoom, rot in _crop_rects(W, H):
        rows = _ocr_rows(pg, rect, zoom, rot)
        sheet, sc, title, tc, elev = _parse(rows)
        if sheet and sc > b_sc:
            b_sheet, b_sc, b_crop = sheet, sc, name
        if title and (elev > b_elev or (elev == b_elev and tc > b_tc)):
            b_title, b_tc, b_elev = title, tc, elev
            if b_crop is None:
                b_crop = name
        if b_sheet and (b_elev or b_title):
            break                           # full identity read — stop the ladder
        if time.time() - t0 > TIME_BUDGET_S:
            break
    best = None
    if b_sheet or b_title:
        best = {"sheet": b_sheet, "title": b_title,
                "conf": round(b_sc if b_sheet else b_tc, 3),
                "sheet_conf": round(b_sc, 3), "title_conf": round(b_tc, 3),
                "elev": b_elev,
                "elev_series": bool(b_sheet and A4_SERIES_RE.match(b_sheet)),
                "crop": b_crop}
        best = _apply_scheme_gate(pg, best)   # validity gate: flag-only on scheme break
    _cache[key] = best
    return best


def elevation_candidate(pg):
    """Page-finding fallback verdict for a title-block-blind page:
    {admit, why, sheet, title, conf} — admit=True when the OCR'd drawing TITLE carries
    ELEVATION (prototype-measured precision 1.000 on 60 pages), or, weaker, when the
    sheet number sits in the A4xx elevation series. Never binds anything: admission only
    ADDS a page to elevation candidacy; walls must still be found on it."""
    r = read_sheet_id(pg)
    if not r:
        return {"admit": False, "why": "ocr_abstain", "sheet": None, "title": None, "conf": 0.0}
    if r["elev"]:
        return {"admit": True, "why": "title_elevation", "sheet": r["sheet"],
                "title": r["title"], "conf": r["conf"]}
    if r["elev_series"]:
        return {"admit": True, "why": "sheet_series_A4xx", "sheet": r["sheet"],
                "title": r["title"], "conf": r["sheet_conf"]}
    return {"admit": False, "why": "not_elevation", "sheet": r["sheet"],
            "title": r["title"], "conf": r["conf"]}
