# -*- coding: utf-8 -*-
"""mat_canon.py — FIX #1: clean, job-level material GROUPING + NAMING for select-to-sum.

WHY (audit-measured, 2026-08-10): the estimator sees ~57 cryptic click-groups per job
because (a) the canvas keys groups on per-page fill_color, and (b) vector_hatch emits a
DISTINCT name per course width (":625", 13 "Lap ... Nin courses" on one job), per seam
spacing (":622"), per RGB (":1208 Color fill (r,g,b)"), plus fixed generic fallbacks
("Panel wall", "Wall area (confirm)", ...). ~71% of colored SF lands under generic/OCR-junk
labels. The estimator has nothing clean to click.

WHAT: `canon(name) -> (group, cls)` inverts the reader's name explosion:
  - collapse pattern-variant families (all lap-course widths -> "Lap siding (horizontal)"),
  - fold OCR junk / non-material tokens into ONE "Unclassified — confirm" bucket,
  - keep the architect's real names / codes.
`material_group(material, fill_color, color_name_map=None) -> (group, cls)` is the per-polygon
entry point: it also lets a real drawing color inherit its legend name (guarded so the
reader's SYNTHETIC category palette never cross-contaminates).

MONEY-SAFE BY CONSTRUCTION: this module NEVER touches area_sf. It only relabels/regroups.
The summed SF per job is byte-identical before and after (proven: mat_canon_selftest.py).
Measured on the 8 audited jobs: click-groups 57 -> ~17 (median), %SF under an actionable
material name 28% -> 60%.
"""
import re

# words / tokens that are NOT a material name (OCR noise, notes, structural tags)
JUNK_WORDS = {
    "NO", "DN", "UP", "YES", "NUMBER", "BOSTON", "AND", "THE", "SEE", "TYP", "SHEET",
    "WALL", "ROOM", "BEAM", "STEEL", "WOOD", "HOOD", "EQ", "AT", "BOLTS", "CL", "MF",
    "EXTERIOR", "FCP", "SS3", "HD19", "CS14", "HDU4", "BRICK", "CMU", "STONE", "SLAB",
    "SLABS", "HDU-4", "ANNUNCIATOR PANEL", "FINISHED END PANEL", "PANEL EDGE NAILING",
    "ANNUNCIATOR", "FIRE RATING (MINS.) PANEL TYPE",
    "FLUSH MOUNTED EMERGENCY CALL BOX AND INSTRUCTION PANEL",
    # single-word legend-HEADER fragments the OCR lifts off a legend block as
    # separate "materials" (26-216: COLOR/FIBER/METAL/PANEL/FINISH = one header row;
    # at practice scale ANGLE/MAIN/LIGHT). One bare word is never a cladding identity,
    # so fold to Unclassified. Multi-word real names ("Metal panel", "Fiber cement
    # panel") are untouched -- only the EXACT single token matches this set.
    "COLOR", "FIBER", "METAL", "PANEL", "FINISH", "ANGLE", "MAIN", "LIGHT", "TYPE",
}

UNCLASSIFIED = "Unclassified — confirm"

# SPEC-NOTE PROSE (2026-08-10, measured): the OCR/callout readers lift whole sentences off the
# sheet ("B. PARTITIONS NOT KEYED ON PLANS SHALL BE ASSUMED AS TYP", "JOINTS WITH MEMBRANE TAPE
# AS SPECIFIED. BARR"), and canon() used to hand any long string back as a REAL architect
# material name. Measured over all 103 durable snapshots: 16 such groups / 79,577 SF (0.5% of
# auto SF) were being presented to the estimator as clickable materials, and near-duplicate
# fragments of ONE note split into several groups. Detected by SPEC VOCABULARY (verbs and
# reference words a material name does not contain) and by list markers, NOT by length — the
# real names are long too ("ACM CLADDING ON EXISTING PRECAST CONCRETE PANELS" is 7 words).
NOTE_WORDS = re.compile(
    r"\b(SHALL|SPECIFIED|ASSUMED|PROVIDED?|PROVIDES|REFER|CONTRACTOR|TYPICAL|REQUIRED|"
    r"NOT KEYED|SEE DETAIL|SEE PLAN|NOTE|NOTES|VERIFY|INSTALLED|INSTALLATION|"
    r"MANUFACTURER|AS NEEDED|WHERE OCCURS)\b")
LIST_MARKER = re.compile(r"^([A-Z]|\d{1,2})[\.\)]\s")


def is_note(u, raw=""):
    """True when the string is sheet PROSE (a specification note), not a material name."""
    return bool(NOTE_WORDS.search(u) or LIST_MARKER.match(raw or u))


def _family_norm(u):
    """Light normalization of ONE product line's synonyms (conservative — never merges two
    different products). Uses the engine's own archive vocabulary."""
    if "ACM" in u or "MCM" in u or "ALUCOBOND" in u or ("COMPOSITE" in u and "METAL" in u and "PANEL" in u):
        return "ACM / composite metal panel"
    if ("FIBER CEMENT" in u or "HARDIE" in u or "FCP" in u or "FCL" in u) and ("PANEL" in u):
        return "Fiber cement panel"
    if ("FIBER CEMENT" in u or "HARDIE" in u) and ("LAP" in u or "PLANK" in u or "SIDING" in u):
        return "Fiber cement lap/plank"
    if "FIBER CEMENT" in u and ("SHINGLE" in u or "SHAKE" in u):
        return "Fiber cement shingle"
    if "FIBER CEMENT" in u or "FCL" in u or "FCP" in u:
        return "Fiber cement (other)"
    if "METAL PANEL" in u or "RIBBED METAL" in u or ("METAL" in u and "PANEL" in u and "SOFFIT" not in u):
        return "Metal panel"
    if "BRICK" in u or "MASON" in u:
        return "Brick / masonry (non-BFS)"
    return None


def canon(name):
    """(canon_group, cls). cls in {real, code, pattern, generic, junk}.
    pattern = a clean nameable cladding family (actionable); generic = pure reader fallback
    with no identity; junk = OCR noise."""
    raw = re.sub(r"\s+", " ", (name or "")).strip()
    u = raw.upper()
    if not u or u.startswith("AI SUGGESTION"):
        return (UNCLASSIFIED, "junk")

    # 1) PATTERN-VARIANT explosion collapse (vector_hatch.py:622/625) -> one family each
    if u.startswith("LAP") and ("COURSE" in u or "HORIZONTAL" in u):
        return ("Lap siding (horizontal)", "pattern")
    if u.startswith("HORIZONTAL PANEL"):
        return ("Horizontal panel", "pattern")
    if u.startswith("VERTICAL PANEL"):
        return ("Vertical panel", "pattern")
    if "VERTICAL RIB" in u:
        return ("Vertical rib panel", "pattern")

    # 2) PURE-GENERIC reader fallbacks (no material identity)
    if u.startswith("COLOR FILL"):
        return ("Color fill", "generic")
    if u.startswith("PANEL WALL"):
        return ("Drawn wall fill (confirm)", "generic")
    if u.startswith("WALL AREA") or u.startswith("WALL BAND"):
        return ("Wall flood (confirm)", "generic")
    if u.startswith("HATCHED AREA"):
        return ("Hatched masonry/EIFS (confirm)", "generic")
    if u.startswith("RENDERED") or u.startswith("SOFFIT/CANOPY"):
        return ("Rendered/soffit (confirm)", "generic")

    # 3) OCR-JUNK: stray tokens / non-material vocabulary
    if u.startswith("GROUP ") or re.match(r"^GROUP\d", u):
        return (UNCLASSIFIED, "junk")
    if u in JUNK_WORDS or len(u) <= 3:
        return (UNCLASSIFIED, "junk")
    if re.search(r"\b[BT]\.?O\.?\b", u) or u.startswith("T.O. BRICK"):
        return (UNCLASSIFIED, "junk")
    if "\n" in (name or ""):
        return (UNCLASSIFIED, "junk")

    # 4) real architect material name.
    # A named material FAMILY or a bare code wins over the prose test, so a real name is never
    # thrown away for containing a spec word ("FIBER CEMENT PANEL, PROVIDED BY OTHERS").
    fam = _family_norm(u)
    if fam:
        return (fam, "real")
    if re.match(r"^[A-Z]{1,4}[-\s]?\d{1,3}[A-Z]?$", u):   # bare code (FC-1, MP-2, GSM-01)
        return (u, "code")
    if is_note(u, raw):          # sheet prose lifted by OCR — never a clickable material
        return (UNCLASSIFIED, "junk")
    disp = raw if len(raw) <= 42 else raw[:42].rstrip() + "…"
    return (disp, "real")


def qcolor(fc):
    """Quantize a fill color to ~6 buckets/channel so shade noise does not over-split."""
    if not fc:
        return None
    try:
        return tuple(round(c * 6) / 6 for c in fc[:3])
    except Exception:
        return None


# the readers' FIXED category palette — NOT real drawing colors, so color-based name
# inheritance must NEVER fire on these (they would cross-contaminate).
#
# ⭐ 2026-08-10 — THE SET NAMED ONE FILE AND THE ENGINE HAS MORE THAN ONE PALETTE.
# The comment said "(vector_hatch.py)" and the seven constants below are indeed
# vector_hatch's + snap_fill's.  `texture.py:15 GROUPS` writes THREE more literal
# category colours into `fill_color` — [0.0,0.80,0.90] / [0.35,0.78,0.45] /
# [0.95,0.45,0.55] — and none of them were here, so `is_synth()` called a texture
# reader's category colour a REAL DRAWING COLOUR.  Measured over the 103-job corpus:
# 3,960 polygons (11.3%) carry one, and `build_color_name_map` bound an architect
# name to one on 96 page-instances.  No polygon inherits through those keys TODAY
# (inheritance needs a bare "Color fill" name, and that reader emits a drawn rgb by
# construction), so this closes a LATENT path, not a live one — measured group delta
# 0 and SF delta 0.000000 over 16,967,795.8 SF (`gate_synth_completeness.py`, 10/10).
#
# The floor is typed; the texture entries are DERIVED from that module so a palette
# edit there propagates instead of silently reopening the gap.  A failed import
# leaves the floor, and `SYNTH_SOURCE` records which happened — an empty derivation
# must be visible, not indistinguishable from a complete one.
SYNTH = {(0.35, 0.55, 0.85), (0.25, 0.75, 0.35), (0.0, 0.72, 0.85), (0.55, 0.75, 0.95),
         (0.85, 0.45, 0.25), (0.45, 0.65, 0.9), (0.45, 0.55, 0.9),
         # texture.py:15 GROUPS — floor copy, kept so the set survives an import failure
         (0.0, 0.8, 0.9), (0.35, 0.78, 0.45), (0.95, 0.45, 0.55)}

SYNTH_SOURCE = "floor"
try:                                    # derive from the module that owns the palette
    import texture as _tex              # noqa: E402  (heavy deps; app.py imports it anyway)
    _derived = {tuple(round(float(c), 2) for c in _col[:3])
                for _name, _col in getattr(_tex, "GROUPS", ())}
    if _derived:
        SYNTH |= _derived
        SYNTH_SOURCE = "texture.GROUPS+floor(%d)" % len(_derived)
except Exception as _e:                 # standalone use (selftests, probes) has no cv2
    SYNTH_SOURCE = "floor (texture unavailable: %s)" % type(_e).__name__


def is_synth(fc):
    if not fc:
        return True
    try:
        return tuple(round(c, 2) for c in fc[:3]) in SYNTH
    except Exception:
        return True


def build_color_name_map(polys):
    """Legend binding: map each REAL (non-synthetic) drawing color to the architect name it
    carries elsewhere in the job, so an un-named 'Color fill' of that same color inherits it.
    `polys` = iterable of dicts with 'material'/'category' and 'fill_color'."""
    m = {}
    for p in polys:
        g, cls = canon(p.get("material") or p.get("category") or "")
        fc = p.get("fill_color")
        if cls in ("real", "code") and not is_synth(fc):
            qc = qcolor(fc)
            if qc is not None:
                m.setdefault(qc, (g, cls))
    return m


def material_group(material, fill_color=None, color_name_map=None):
    """Per-polygon entry point -> (group, cls). For a bare 'Color fill', split by the REAL
    drawing color and inherit a legend name when one exists; otherwise fall back to canon()."""
    g, cls = canon(material)
    if cls == "generic" and g == "Color fill" and fill_color and not is_synth(fill_color):
        qc = qcolor(fill_color)
        if color_name_map and qc in color_name_map:
            return color_name_map[qc]          # inherited architect name for this color
        if qc is not None:
            return ("Color: %s" % (qc,), "code")   # its own real-color swatch group
    return (g, cls)
