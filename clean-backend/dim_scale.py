"""
dim_scale.py — self-calibrate a sheet's scale from its own DIMENSION STRINGS
(blueprint Phase 1b). A dimension "24'-0\"" drawn over a line whose vector length
we can measure IS the scale — stronger than title-block text, works per sheet
even on mixed-scale sheets.

Money-safety design (same as the elevation-marker reader): pair every feet-inches
text with its nearest parallel dimension line, compute the implied ft/pt of each
pair, then trust ONLY the largest agreeing group (>=3 pairs within 3%). Mispaired
strings scatter; the true scale's pairs pile up. When there's no consensus we say
nothing — never a confident wrong scale (SF error = scale error SQUARED).
"""
import re
import fitz

# 24'-0"  ·  3'-4"  ·  12'-6 1/2"  ·  8'  — a measurement, not a level marker
_DIM = re.compile(r"^(\d{1,3})'(?:\s*-?\s*(\d{1,2})(?:\s+(\d{1,2})/(\d{1,2}))?\s*\"?)?$")

_MAX_TEXT_TO_LINE_PT = 30.0    # dim text sits ON/next to its line
_MIN_LINE_PT = 18.0            # ignore tick marks / extension stubs
_PLAUSIBLE_FTPI = (0.5, 40.0)  # ft per inch of paper — arch scales live here
_AGREE = 0.03                  # pairs within 3% = same scale


def _feet(m):
    ft = float(m.group(1))
    if m.group(2):
        inch = float(m.group(2))
        if m.group(3) and m.group(4) and float(m.group(4)) != 0:
            inch += float(m.group(3)) / float(m.group(4))
        ft += inch / 12.0
    return ft


def _dim_words(pg):
    """(cx, cy, feet, horizontal?) for every feet-inches word on the page."""
    out = []
    try:
        words = pg.get_text("words")
    except Exception:
        return out
    for w in words:
        x0, y0, x1, y1, txt = w[0], w[1], w[2], w[3], (w[4] or "").strip()
        m = _DIM.match(txt)
        if not m:
            continue
        ft = _feet(m)
        if ft < 0.5 or ft > 400:
            continue
        out.append(((x0 + x1) / 2, (y0 + y1) / 2, ft, (x1 - x0) >= (y1 - y0)))
    return out


def _segments(pg):
    """Long straight H/V segments in display coords, split by axis."""
    rot = pg.rotation_matrix
    hs, vs = [], []
    try:
        drawings = pg.get_drawings()
    except Exception:
        return hs, vs
    for d in drawings:
        for it in d.get("items") or []:
            if it[0] != "l":
                continue
            p0 = fitz.Point(it[1]) * rot
            p1 = fitz.Point(it[2]) * rot
            dx, dy = abs(p1.x - p0.x), abs(p1.y - p0.y)
            if dx >= _MIN_LINE_PT and dy <= 1.5:
                hs.append((min(p0.x, p1.x), max(p0.x, p1.x), (p0.y + p1.y) / 2, dx))
            elif dy >= _MIN_LINE_PT and dx <= 1.5:
                vs.append((min(p0.y, p1.y), max(p0.y, p1.y), (p0.x + p1.x) / 2, dy))
    return hs, vs


def sheet_scale(pg):
    """Consensus scale from dimension strings. Returns (ft_per_pt, n_agreeing)
    or None when the sheet doesn't give a confident answer.

    Each dim string votes with ALL its plausible line pairings (a dim line often
    has extension/junk segments nearby — nearest-by-distance picks wrong; but the
    TRUE scale is the one many different dims can each reach with SOME pairing,
    while mispairings scatter)."""
    dims = _dim_words(pg)
    if len(dims) < 3:
        return None
    hs, vs = _segments(pg)
    votes = []                 # per dim: list of plausible implied ft/pt values
    for cx, cy, ft, horiz in dims:
        cands = set()
        segs = hs if horiz else vs
        along, perp_at = (cx, cy) if horiz else (cy, cx)
        for (a0, a1, c, ln) in segs:
            if a0 + 0.02 * ln <= along <= a1 - 0.02 * ln and abs(perp_at - c) <= _MAX_TEXT_TO_LINE_PT:
                fpp = ft / ln
                if _PLAUSIBLE_FTPI[0] <= fpp * 72.0 <= _PLAUSIBLE_FTPI[1]:
                    cands.add(fpp)
        if cands:
            votes.append(sorted(cands))
    if len(votes) < 3:
        return None
    # every candidate value is an anchor; a dim supports it if ANY of its own
    # candidates is within 3%. Largest support wins.
    anchors = sorted(v for vs_ in votes for v in vs_)
    best_a, best_sup = None, []
    for a in anchors:
        sup = []
        for vs_ in votes:
            m = min(vs_, key=lambda v: abs(v - a))
            if abs(m - a) / a <= _AGREE:
                sup.append(m)
        if len(sup) > len(best_sup):
            best_a, best_sup = a, sup
    if len(best_sup) < 4 or len(best_sup) < 0.35 * len(votes):
        return None                     # scattered pairings — refuse to guess
    # PHANTOM CHECK: mispairing with nested collinear segments creates a rival
    # cluster at ~half/double scale, built from the SAME dims. Discriminator =
    # EXCLUSIVE voters: dims whose candidates reach only one cluster. The true
    # scale has dims that see only the correct line; the phantom exists purely
    # through dims that also see the true one. No clear exclusive winner → silent.
    def reaches(vs_, a):
        return abs(min(vs_, key=lambda v: abs(v - a)) - a) / a <= _AGREE
    rival_a, rival_sup = None, 0
    for a in anchors:
        if abs(a - best_a) / best_a <= 0.06:
            continue
        sup = sum(1 for vs_ in votes if reaches(vs_, a))
        if sup > rival_sup:
            rival_a, rival_sup = a, sup
    if rival_a is not None and rival_sup * 1.6 > len(best_sup):
        exc_best = sum(1 for vs_ in votes if reaches(vs_, best_a) and not reaches(vs_, rival_a))
        exc_riv = sum(1 for vs_ in votes if reaches(vs_, rival_a) and not reaches(vs_, best_a))
        if exc_riv >= 3 and exc_riv >= 1.6 * exc_best:
            # the rival is the real one — recompute its supporters and return it
            sup = sorted(min(vs_, key=lambda v: abs(v - rival_a)) for vs_ in votes if reaches(vs_, rival_a))
            return (sup[len(sup) // 2], len(sup))
        if not (exc_best >= 3 and exc_best >= 1.6 * exc_riv):
            return None                 # genuinely ambiguous — refuse to guess
    best_sup.sort()
    return (best_sup[len(best_sup) // 2], len(best_sup))
