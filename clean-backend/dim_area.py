# -*- coding: utf-8 -*-
"""dim_area.py -- the GENUINELY INDEPENDENT second area reader (second_reader lane).

WHAT IT IS
----------
For an elevation page, this derives an area estimate DIRECTLY from the printed
W and H DIMENSION STRINGS -- W_ft * H_ft, read off the numbers the architect
drew -- and NEVER from the traced hatch boundary. That is the whole point: the
vector reader's dominant error is a boundary-convention over-read (gross-vs-net,
hatch bleed; measured over-reading on 71.8% of walls at a median 1.275x), and any
reader built from the same traced extent over-reads WITH it -- measured in
SECOND_READER_CORR.json: raster/density err-corr 0.4207, declared rectangle
(w_ft*h_ft) err-corr 0.6545, both REJECTED as non-independent. A number read from
the printed dimension text has a DECORRELATED failure mode (OCR miss / wrong string
pairing), so its AGREEMENT with the vector reader is the first channel this system
has that can catch that over-read. It is the pick the measurement lane justified.

TWO-TIER / MONEY-SAFE (this is a CONFIDENCE SIGNAL, not a measurement)
---------------------------------------------------------------------
app.py attaches area() alongside the vector SF and NEVER replaces it -- no booked
square foot moves. And area() ABSTAINS (returns None) unless BOTH axes carry a
consensus dimension chain, because an area is a PRODUCT of two lengths and a wrong
length squares into the SF (the same money-safety dim_scale is built on: SF error =
scale error squared). A confident wrong area would be worse than silence, so when a
sheet does not give a clean answer this says nothing.

INDEPENDENT OF PIXELS AND OF SCALE, TOO
---------------------------------------
It needs no pixels and no ft/pt scale: the feet are PRINTED. So its errors do not
even share dim_scale's line-pairing failure mode. It consumes only the parsed
feet-inches WORDS (dim_scale._dim_words -> (cx, cy, feet, horizontal?)), splits them
by axis, and asks ONE question per axis: is the largest printed dimension
CORROBORATED by a chain of smaller printed dimensions on the same axis that sums to
it within 3%? An overall that its own sub-dimensions add up to is a closed dimension
chain -- the classic drawing check -- and only then is the overall trusted.

DETERMINISM
-----------
Pure function of pg.get_text("words"); no RNG, no cache, no I/O. Same page in ->
same number out, cold or warm. (The vector reader it is compared against must itself
be warm-invariant before any agreement number is fit -- that is the separate
WARM_INVARIANCE_PASS.json precondition the second_reader lane's Phase B enforces.)
"""
import dim_scale

_AGREE = 0.03            # a chain that closes to within 3% of its overall
_MIN_PANEL_FT = 6.0      # below this it is a detail callout, not an elevation panel
_MAX_PANEL_FT = 400.0    # dim_scale's own feet ceiling; above it is a site/civil dimension
_MIN_PER_AXIS = 3        # need the overall + at least two chain segments to corroborate
_MAX_PARTS = 16          # bound the subset-sum search -> fixed per-page, per-upload cost


def _subset_closes(parts, target):
    """Return the closing subset (a list) iff some subset of `parts` sums to within
    _AGREE*target of target, else None. Bounded, deterministic DP on 0.1-ft buckets
    so the per-page cost is fixed no matter how many dimensions a sheet carries."""
    tol = _AGREE * target
    parts = sorted((p for p in parts if 0 < p <= target + tol), reverse=True)[:_MAX_PARTS]
    if not parts:
        return None
    reach = {0.0: []}                     # reachable sum (0.1-ft bucket) -> the subset
    for p in parts:
        nxt = dict(reach)
        for s, sub in reach.items():
            ns = s + p
            if ns > target + tol:
                continue
            b = round(ns, 1)
            if b not in nxt:
                nxt[b] = sub + [p]
        reach = nxt
    best = min(reach, key=lambda s: abs(s - target))
    if abs(best - target) > tol:
        return None
    return reach[best]


def _axis_overall(vals):
    """vals: [(pos, ft), ...] for ONE axis. Return (overall_ft, basis) when the
    largest printed dimension is corroborated by a closing chain of smaller ones on
    the same axis; else None (abstain)."""
    if len(vals) < _MIN_PER_AXIS:
        return None
    fts = [ft for _pos, ft in vals]
    overall = max(fts)
    if not (_MIN_PANEL_FT <= overall <= _MAX_PANEL_FT):
        return None
    parts = [ft for ft in fts if ft < overall * (1 - _AGREE)]
    if len(parts) < 2:
        return None
    chain = _subset_closes(parts, overall)
    if chain is None:
        return None
    return overall, {"overall_ft": round(overall, 3),
                     "chain_ft": [round(x, 3) for x in chain],
                     "chain_sum_ft": round(sum(chain), 3),
                     "n_axis": len(vals)}


def area(pg):
    """Independent per-page elevation area from printed W/H dimension strings, or None.

    Reuses dim_scale._dim_words -> (cx, cy, feet, horizontal?). Horizontal words
    measure width (W), vertical words measure height (H). Requires a corroborated
    overall on BOTH axes; the returned dict is pure metadata (a confidence signal)
    and is never allowed to move a booked SF.
    """
    try:
        words = dim_scale._dim_words(pg)
    except Exception:
        return None
    if len(words) < 2 * _MIN_PER_AXIS:
        return None
    horiz = [(cx, ft) for (cx, cy, ft, h) in words if h]
    vert = [(cy, ft) for (cx, cy, ft, h) in words if not h]
    w = _axis_overall(horiz)
    h = _axis_overall(vert)
    if not w or not h:
        return None
    w_ft, wb = w
    h_ft, hb = h
    return {"area_sf": w_ft * h_ft, "w_ft": w_ft, "h_ft": h_ft,
            "basis": {"w": wb, "h": hb, "reader": "printed-dimension-strings"}}
