"""READ THE DRAWING'S OWN WORDS — material callouts + leader arrows -> region names.

Architects label walls with text callouts connected by leader lines ("METL-SPAN CF LIGHT
MESA", "FIBER CEMENT LAP SIDING"). Instead of guessing materials, follow the arrows:
  1) collect short text blocks that read like material callouts;
  2) find their leader lines (a segment starting at the block, ending in a region);
  3) name the region with the architect's own words.
Falls back to proximity when no leader is found. No-op on textless (curve-flattened) sets.
"""
import re
import fitz

# words that make a short text block smell like a cladding/material callout
_MAT_RE = re.compile(r"\b(panel|siding|lap|shake|shingle|brick|masonry|cmu|stucco|eifs|acm|"
                     r"metal|aluminum|alum|composite|fiber|cement|cementitious|hardie|azek|pvc|"
                     r"board\s*&?\s*batten|standing\s*seam|metl[-\s]?span|kingspan|soffit|fascia|"
                     r"trim|veneer|cladding|wall\s*system)\b", re.I)
# blocks that are definitely NOT material names
_NOISE_RE = re.compile(r"\b(scale|elevation\s*$|sheet|drawn|date|revision|note[s]?:|typ\.?$|"
                       r"see\s+(detail|spec|struct)|contractor|install|provide|existing\s+to\s+remain)\b", re.I)
# LINEAR items (priced per LF) — must never name a big AREA region
_LINEARISH_RE = re.compile(r"\b(trim|railing|handrail|guardrail|watertable|decking|downspout|"
                           r"gutter|drip|flashing|coping)\b", re.I)
# NOT cladding at all — openings, hardware, notes that happen to contain a material word
_NOTCLAD_RE = re.compile(r"\b(glass|glazing|window|door|louver|vent|hose\s*bibb|meter|"
                         r"knox|sconce|light\s*pak|spandrel|parking|column\s*wrap|canopy)\b", re.I)


def _is_dark(c):
    return c and len(c) >= 3 and (c[0] + c[1] + c[2]) / 3 < 0.5


def _arrowheads(pg):
    """Dark ~3-vertex small filled triangles = leader arrowheads. Returns list of
    {tip:(x,y), cx, cy}. tip = sharpest-angle vertex (points AT the region)."""
    import math
    heads = []
    try:
        for d in pg.get_drawings():
            if not (_is_dark(d.get("fill")) or _is_dark(d.get("color"))):
                continue
            r = d["rect"]
            area = r.width * r.height
            if area < 8 or area > 150:
                continue
            pts = []
            for it in d["items"]:
                if it[0] == "l":
                    pts += [(it[1].x, it[1].y), (it[2].x, it[2].y)]
            uniq = list({(round(x, 1), round(y, 1)) for x, y in pts})
            if len(uniq) != 3:
                continue

            def ang(a, b, c):
                v1 = (b[0] - a[0], b[1] - a[1]); v2 = (c[0] - a[0], c[1] - a[1])
                d1 = math.hypot(*v1); d2 = math.hypot(*v2)
                if d1 * d2 == 0:
                    return 999
                cs = max(-1, min(1, (v1[0] * v2[0] + v1[1] * v2[1]) / (d1 * d2)))
                return math.acos(cs)

            angs = [ang(uniq[i], uniq[(i + 1) % 3], uniq[(i + 2) % 3]) for i in range(3)]
            tip = uniq[angs.index(min(angs))]
            heads.append({"tip": tip, "cx": sum(p[0] for p in uniq) / 3, "cy": sum(p[1] for p in uniq) / 3})
    except Exception:
        pass
    return heads[:2000]


def _blocks(pg):
    out = []
    try:
        for b in pg.get_text("blocks"):
            x0, y0, x1, y1, t = b[0], b[1], b[2], b[3], (b[4] or "").strip()
            t = re.sub(r"\s+", " ", t)
            if not (3 <= len(t) <= 90):
                continue
            if not _MAT_RE.search(t):
                continue
            if _NOISE_RE.search(t) or _NOTCLAD_RE.search(t):
                continue
            out.append({"rect": (x0, y0, x1, y1), "text": t})
    except Exception:
        pass
    return out[:80]


class _RectGrid:
    """Uniform-grid index over text-block rects — 'is this point at a callout?' in O(1).
    Without it the anchored filter below would be O(segments x blocks) on every page."""

    __slots__ = ("cell", "d")

    def __init__(self, rects, cell=64.0):
        self.cell = float(cell)
        self.d = {}
        for i, r in enumerate(rects):
            x0, y0, x1, y1 = r
            for cx in range(int(x0 // self.cell), int(x1 // self.cell) + 1):
                for cy in range(int(y0 // self.cell), int(y1 // self.cell) + 1):
                    b = self.d.setdefault((cx, cy), [])
                    if len(b) < 400:
                        b.append((r, i))

    def _near(self, x, y, pad):
        cx, cy = int(x // self.cell), int(y // self.cell)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for (r, i) in self.d.get((cx + dx, cy + dy), ()):
                    if r[0] - pad <= x <= r[2] + pad and r[1] - pad <= y <= r[3] + pad:
                        yield i

    def hit(self, x, y, pad):
        for _ in self._near(x, y, pad):
            return True
        return False

    def idx(self, x, y, pad):
        return sorted(set(self._near(x, y, pad)))


def _leaders(pg, min_len=25, max_len=600, anchors=None, cap=4000, anchor_pad=20.0):
    """Plain-ish segments that could be leader shafts (any angle, medium length).

    FIX 6 (2026-08-04) — THE CAP WAS THE BUG. This used to return `segs[:4000]`: the
    FIRST 4000 length-filtered segments in page order. On a hatched elevation those
    4000 are all seam lines, so the real leaders never survived and leader-naming read
    as "unavailable" on exactly the dense sheets it was written for. Measured on 5,891
    gold walls in 119 paired jobs (MATERIAL_SIGNAL_CENSUS): 92.7% of gold walls sit on
    a page where that cap binds, and 95.7% of the walls a leader COULD name are on such
    a page — uncapped leader tracing is available on 25.7% of walls vs the whole
    production namer's 4.9%.

    The fix is not "raise the cap" (that costs O(segments x blocks) work on every page).
    When `anchors` — the material text-block rects, page space — are supplied, only
    segments with an ENDPOINT AT a callout are kept. That is a SUPERSET of the segments
    name_regions could ever use (its own test is _near_rect(..., 16) on the same rects,
    and anchor_pad here is 20), so this can only ADD coverage, never remove it, and the
    surviving set is ~1000x smaller — so `cap` stops binding instead of deciding.
    """
    segs = []
    grid = _RectGrid(anchors) if anchors else None
    lo2, hi2 = min_len * min_len, max_len * max_len
    full = False
    try:
        for d in pg.get_drawings():
            if full:
                break
            for it in d["items"]:
                if it[0] != "l":
                    continue
                p1, p2 = it[1], it[2]
                L2 = (p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2
                if not (lo2 <= L2 <= hi2):
                    continue
                if grid is not None and not (grid.hit(p1.x, p1.y, anchor_pad)
                                             or grid.hit(p2.x, p2.y, anchor_pad)):
                    continue
                segs.append((p1.x, p1.y, p2.x, p2.y))
                if len(segs) >= cap:
                    full = True
                    break
    except Exception:
        pass
    return segs


def _pip(pt, poly):
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i - 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1:
            inside = not inside
    return inside


def _near_rect(pt, rect, pad):
    return rect[0] - pad <= pt[0] <= rect[2] + pad and rect[1] - pad <= pt[1] <= rect[3] + pad


# note/instruction words — if a "callout" reads like a spec sentence, it's NOT a material name
_NOTE_RE = re.compile(r"\b(refer|see|per|secure|fasten|provide|install|coordinate|seal|"
                      r"match|dams?|ties|sheathing|continuous|prefinished|typ|note|detail|"
                      r"schedule|remove|clean|paint|reattach|exposure|fastener|conceal|"
                      r"air\s*barrier|barrier|balcony|manufacturer|knife|thru|membrane|"
                      r"securement|coursing|opening)\b", re.I)
# a real material name doesn't START with a connective/fragment word
_FRAG_RE = re.compile(r"^(of|and|the|to|at|w/|@|with|or|in|on)\b", re.I)
# cut a material name off where a note-clause begins ("LP LAP SIDING, REFER TO..." -> "LP LAP SIDING")
_TAIL_RE = re.compile(r"\s*(?:[,;:]|\bREFER\b|\bSEE\b|\bPER\b|\bTYP\b|\bTO MATCH\b|\bSECURED\b|"
                      r"\bFASTENED\b|\bW/\b|@).*$", re.I)


def _clean(t):
    t = re.sub(r"[•‣▪]\s*", "", t or "").strip(" .;:,-–—")
    t = re.sub(r"\s+", " ", t)
    if t.count('"') % 2:                     # drop a dangling open-quote from truncation
        t = re.sub(r'\s+"[^"]*$', "", t)
    return t[:80].strip(' "-')


def _material_name(t):
    """Return a clean MATERIAL name, or None if the text is a construction note, not a material.
    Precision-first: a wrong name corrupts the takeoff, so reject anything note-shaped."""
    t = _clean(t)
    if not t:
        return None
    core = _TAIL_RE.sub("", t).strip(' "-@/')  # keep only the material noun-phrase before any note-clause
    if len(core) < 4:
        return None
    if re.match(r"^\d", core):                # "040 CONT", "2 SEAL" — a note, not a material
        return None
    if _FRAG_RE.search(core):                 # "OF SIDING", "@ STONE" — a fragment, not a name
        return None
    if _NOTE_RE.search(core):                 # still reads like an instruction
        return None
    if not _MAT_RE.search(core):              # must actually name a material
        return None
    return core[:56]


# material KEY tags: M1, PNL-1, EF-3, WD2 ... (legend key on the wall)
_KEY_RE = re.compile(r"^[A-Z]{1,3}[-.]?\d{1,2}$")

# ---------------------------------------------------------------- FIX 6 CODE-TAG TIER
# The archive labels two-thirds of its gold SF with a BARE CODE, not an English material
# word (HUNT_BASELINES G10: _MAT_RE recognises 33.2% of gold material SF). A bare code
# printed ON the wall is, measured, the single most accurate identity signal in the
# drawing — on 5,891 gold walls across 119 paired jobs (MATERIAL_SIGNAL_CENSUS, S3b):
#   available on 9.8% of walls · 96.5% correct · +13.3 pts over "always say fiber cement"
#   · +25.1 pts over the GC-majority guess · +6.9 pts over an ORACLE that already knows
#   the job's majority family · +21.1 pts at VARIANT (colour/line-item) level.
# _KEY_RE cannot be reused for this: it is the LEGEND key shape and matches door/window/
# detail tags (W-1, D-2, A-1) just as happily. The tier is therefore gated on a MATERIAL
# PREFIX TABLE — the same table that produced the 96.5% — and names nothing outside it.
# Widening the shape WITHOUT that gate is what the census measured as S3/S3x: legend and
# UniFormat-keynote resolution scored 19.5% / 23.1% correct, i.e. 70-76 points BELOW a
# constant string. That path is deliberately NOT taken here.
_CODE_TAG_RE = re.compile(r"^([A-Z]{1,4})[-.]?(\d{1,2}[A-Z]?)$")
_CODE_FAMILY = {
    # fiber cement
    "FC": "FIBER CEMENT", "FCP": "FIBER CEMENT", "FCS": "FIBER CEMENT",
    "FCL": "FIBER CEMENT", "FCB": "FIBER CEMENT", "FBC": "FIBER CEMENT",
    "HDFC": "FIBER CEMENT", "SFC": "FIBER CEMENT", "FSH": "FIBER CEMENT",
    "FSV": "FIBER CEMENT", "LS": "FIBER CEMENT",
    # metal panel
    "MP": "METAL PANEL", "CMP": "METAL PANEL", "LPSS": "METAL PANEL",
    "SS": "METAL PANEL", "IMP": "METAL PANEL",
    # aluminium/metal composite
    "ACM": "ACM", "MCM": "ACM",
}


def _code_tag(tok):
    """A word token -> (CODE, FAMILY) when it is a MATERIAL code tag, else None."""
    m = _CODE_TAG_RE.match(tok or "")
    if not m:
        return None
    fam = _CODE_FAMILY.get(m.group(1))
    if not fam:
        return None
    return (m.group(1) + "-" + m.group(2), fam)


def _legend_pairs(pg, blocks):
    """Legend convention: a short KEY (M5 / PNL-1) sitting just left of a material text block
    at the same height -> {key: material name}. This is how estimators read the sheet."""
    pairs = {}
    try:
        for w in pg.get_text("words"):
            key = (w[4] or "").strip()
            if not _KEY_RE.match(key):
                continue
            wy = (w[1] + w[3]) / 2
            for b in blocks:
                r = b["rect"]
                if r[0] >= w[2] - 4 and r[0] - w[2] < 180 and r[1] - 8 <= wy <= r[3] + 8:
                    if key not in pairs:
                        pairs[key] = _clean(b["text"])
                    break
    except Exception:
        pass
    return pairs


def _key_words(pg):
    out = []
    try:
        for w in pg.get_text("words"):
            key = (w[4] or "").strip()
            if _KEY_RE.match(key):
                out.append(((w[0] + w[2]) / 2, (w[1] + w[3]) / 2, key))
    except Exception:
        pass
    return out


def _code_words(pg):
    """FIX 6 — MATERIAL code tags printed on the drawing: [(cx, cy, CODE, FAMILY)]."""
    out = []
    try:
        for w in pg.get_text("words"):
            ct = _code_tag((w[4] or "").strip().strip(".:;,()"))
            if ct:
                out.append(((w[0] + w[2]) / 2, (w[1] + w[3]) / 2, ct[0], ct[1]))
    except Exception:
        pass
    return out[:4000]


# FIX 6 tunables. Chosen on the TRAIN half of the held-out split; the swept values and
# the chosen ones are reported in FIX_NAMING_COVERAGE.md.
_CODE_MARGIN_PT = 0.0      # how far outside the region a code tag may sit (0 = strictly on it)
_NEAR_BIND_PT = 0.0        # ARM D — near-bind OFF (35.7% marginal held-out, and it destroys correct names)
_TIERS = ("code", "arrow", "leader")   # ARM D — legend OFF (held-out 6.5% marginal), near-bind OFF


def _poly_dist(pt, poly):
    """Point -> polygon distance in points (0 when inside). Pure-python, no shapely:
    this runs inside the request path."""
    if _pip(pt, poly):
        return 0.0
    x, y = pt
    best = 1e18
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        L2 = dx * dx + dy * dy
        if L2 <= 1e-12:
            d2 = (x - x1) ** 2 + (y - y1) ** 2
        else:
            t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / L2))
            d2 = (x - x1 - t * dx) ** 2 + (y - y1 - t * dy) ** 2
        if d2 < best:
            best = d2
    return best ** 0.5


_SCHED_NUM = re.compile(r"^[\d,]+(?:\.\d+)?$")
_SCHED_UNIT = re.compile(r"^(sf|sq|s\.?f\.?)$", re.I)


def read_schedule_pg(pg):
    """Schedule reader on an already-open page (cheap — no per-page PDF re-parse)."""
    out = []
    try:
        legend = _legend_pairs(pg, _blocks(pg))
        from collections import defaultdict
        rows = defaultdict(list)
        for w in pg.get_text("words"):
            rows[round(w[1] / 4)].append(w)     # cluster by ~4pt y-band = one table row
        best = {}
        for _, ws in rows.items():
            ws = sorted(ws, key=lambda w: w[0])
            toks = [w[4] for w in ws]
            for i in range(len(toks) - 1):
                if not _KEY_RE.match(toks[i]):
                    continue
                for j in range(i + 1, min(i + 3, len(toks))):
                    if _SCHED_NUM.match(toks[j]):
                        val = float(toks[j].replace(",", ""))
                        unit_ok = (j + 1 < len(toks) and _SCHED_UNIT.match(toks[j + 1]))
                        if val >= 50 and unit_ok:
                            best[toks[i]] = max(best.get(toks[i], 0), val)  # schedule total > stray region labels
                        break
        for key, val in best.items():
            mat = legend.get(key, "")
            if not mat or _LINEARISH_RE.search(mat):   # only wall materials (skip railings/trim fixtures)
                continue
            out.append({"key": key, "sf": round(val), "material": mat})
    except Exception:
        pass
    out.sort(key=lambda r: -r["sf"])
    # only trust it as a schedule if there are at least 2 keyed material rows
    return out if len(out) >= 2 else []


# elevation LEVEL markers: "T.O. STEEL / FINISHED FLOOR / CEILING ... ELEV.: 139'-0""
_LEVEL_LABEL = re.compile(r"\b(T\.?O\.?|B\.?O\.?|FIN(?:ISHED)?\s*FL(?:OO)?R|CEILING|ROOF|"
                          r"SECOND\s*FL|FIRST\s*FL|THIRD\s*FL|EAVE|RIDGE|PARAPET|GRADE|STEEL|WALL|FTG)\b", re.I)
_FEETIN = re.compile(r"(\d{1,3})'\s*-?\s*(\d{1,2})?(?:\s*(\d+)/(\d+))?\"?")


def read_levels_pg(pg):
    """Read elevation MARKERS off a text-bearing page: [(y_display, feet, label)].
    A marker's y-position IS the height it names — two or more give the exact vertical
    scale of the sheet by linear fit (how a human knows a wall is 39 ft without measuring)."""
    out = []
    try:
        rot = pg.rotation_matrix
        for b in pg.get_text("blocks"):
            t = re.sub(r"\s+", " ", (b[4] or "")).strip()
            if len(t) > 90 or not _LEVEL_LABEL.search(t):
                continue
            m = _FEETIN.search(t)
            if not m:
                continue
            ft = int(m.group(1)) + (int(m.group(2)) if m.group(2) else 0) / 12.0
            if not (0 <= ft <= 400):
                continue
            cy = (b[1] + b[3]) / 2
            cx = (b[0] + b[2]) / 2
            p = fitz.Point(cx, cy) * rot
            out.append({"y": round(p.y, 1), "ft": round(ft, 2), "label": t[:40]})
    except Exception:
        pass
    return out[:40]


def vertical_scale_from_levels(levels):
    """Markers → vertical scale, done the way sheets actually work: MULTIPLE elevations are
    stacked on one sheet, each with its own marker column, so we CLUSTER by y-gap first and
    fit per cluster. Within a cluster, pairwise-slope MEDIAN (robust: one OCR-mangled value
    can't poison the fit), then consensus filter. Returns (ft_per_point, quality, n_markers)
    or None. Elevation decreases as drawing-y increases; slope must be negative."""
    import statistics
    pts = sorted(set((l["y"], l["ft"]) for l in levels))
    if len(pts) < 2:
        return None
    # cluster by vertical gaps (stacked elevations are far apart)
    clusters = [[pts[0]]]
    for p in pts[1:]:
        if p[0] - clusters[-1][-1][0] > 250:
            clusters.append([p])
        else:
            clusters[-1].append(p)
    slopes = []
    used = 0
    for cl in clusters:
        cl = [p for p in cl]
        if len({p[1] for p in cl}) < 2:
            continue
        # pairwise slopes (ft per drawing-point), keep negatives only
        pw = []
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                dy = cl[j][0] - cl[i][0]
                df = cl[j][1] - cl[i][1]
                if abs(dy) < 8:
                    continue
                s = df / dy
                if s < 0:
                    pw.append(s)
        if len(pw) < 2:
            continue
        # LARGEST AGREEING GROUP of pairwise slopes (within 6%) — a single agreeing pair is
        # not evidence; OCR-mangled values can't form a group, real markers always do
        pw.sort()
        best_grp = []
        for i in range(len(pw)):
            grp = [s for s in pw if abs(s - pw[i]) <= 0.06 * abs(pw[i])]
            if len(grp) > len(best_grp):
                best_grp = grp
        if len(best_grp) < 2:
            continue
        slopes.append(statistics.median(best_grp))
        used += len(cl)
    if not slopes:
        return None
    a = statistics.median(slopes)
    ftpi = abs(a) * 72.0
    if not (0.5 <= ftpi <= 40):          # real elevation scales live here; anything else = bad read
        return None
    spread = max(slopes) - min(slopes) if len(slopes) > 1 else 0.0
    quality = 1.0 - min(1.0, spread / abs(a)) if a else 0.0
    if len(slopes) > 1 and quality < 0.94:
        return None                      # clusters disagree — don't trust the read
    return (abs(a), round(quality, 4), used)


def read_schedule(pdf_bytes, page_index):
    """Convenience wrapper: read the schedule from one page of a PDF byte-stream."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        r = read_schedule_pg(doc[page_index])
        doc.close()
        return r
    except Exception:
        return []


def name_regions(pdf_bytes, page_index, polys, pw, ph):
    """polys carry normalized DISPLAY points. Returns count of regions named.
    Mutates polys in place: sets material/category/group to the callout text when found."""
    doc = None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        return name_regions_pg(doc[page_index], polys, pw, ph)
    except Exception:
        return 0
    finally:
        try:
            if doc is not None:
                doc.close()
        except Exception:
            pass


def name_regions_pg(pg, polys, pw, ph):
    """Same reader on an ALREADY-OPEN page — the caller owns the document. app.py calls
    the naming layer twice per page (first pass + legend-first second pass); each call
    used to re-open the whole PDF from bytes. Nothing else about the read changes."""
    try:
        blocks = _blocks(pg)
        codes = _code_words(pg)
        # FIX 6: the old early-exit was `if not blocks: return 0` — a page whose walls
        # are labelled ONLY with bare codes ("FC-1") has no _MAT_RE text block at all,
        # so the reader gave up before it could read the codes. Two-thirds of the
        # archive's gold SF is labelled exactly that way (HUNT_BASELINES G10).
        if not blocks and not codes:
            return 0
        rot = pg.rotation_matrix
        # FIX 6: anchor the leader scan to the callout blocks instead of truncating the
        # page's first 4000 segments (see _leaders). Same segments the loop below can
        # use, without the seam-line flood deciding which ones survive.
        _want_lead = ("leader" in _TIERS or "arrow" in _TIERS)
        segs = _leaders(pg, anchors=[b["rect"] for b in blocks]) if (blocks and _want_lead) else []
        legend = _legend_pairs(pg, blocks) if (blocks and "legend" in _TIERS) else {}
        keys = _key_words(pg) if (blocks and "legend" in _TIERS) else []
        heads = _arrowheads(pg) if (blocks and "arrow" in _TIERS) else []
        # region polygons in PAGE-display coords
        rpolys = [[(x * pw, y * ph) for x, y in p["points"]] for p in polys]
        # bbox per region — a cheap reject before any point-in-polygon / distance work
        rbb = []
        for rr in rpolys:
            if len(rr) < 3:
                rbb.append(None)
            else:
                xs = [q[0] for q in rr]; ys = [q[1] for q in rr]
                rbb.append((min(xs), min(ys), max(xs), max(ys)))
        # blocks + segs to display coords
        def rp(x, y):
            q = fitz.Point(x, y) * rot
            return (q.x, q.y)
        dblocks = []
        for b in blocks:
            c = [rp(b["rect"][0], b["rect"][1]), rp(b["rect"][2], b["rect"][3])]
            r = (min(c[0][0], c[1][0]), min(c[0][1], c[1][1]), max(c[0][0], c[1][0]), max(c[0][1], c[1][1]))
            dblocks.append({"rect": r, "text": b["text"]})
        named = 0
        used = set()
        # 0) LEGEND-KEY FIRST — the most reliable convention (M5/PNL-1 tag on the wall + a
        # legend mapping key->material). Runs before leaders/proximity so junk text can't
        # claim a region that has a proper key.
        if legend and keys and "legend" in _TIERS:
            for (kx, ky, key) in keys:
                if key not in legend:
                    continue
                q = fitz.Point(kx, ky) * rot
                pt = (q.x, q.y)
                for ri, rr in enumerate(rpolys):
                    if ri in used or len(rr) < 3:
                        continue
                    if _pip(pt, rr):
                        polys[ri]["material"] = legend[key]
                        polys[ri]["category"] = legend[key]
                        polys[ri]["group"] = legend[key]
                        polys[ri]["named_by"] = "legend:" + key
                        used.add(ri)
                        named += 1
                        break
        # 0b) CODE-TAG TIER — a MATERIAL code printed on the wall ("FC-1", "MP-2").
        # Runs after the job's own legend (which is more specific) and before leaders,
        # because a tag sitting ON the region is a stronger claim than a line pointing
        # at it. Gated on _CODE_FAMILY: unknown prefixes are never named (see the table).
        for (cx, cy, code, fam) in (codes if "code" in _TIERS else ()):
            q = fitz.Point(cx, cy) * rot
            pt = (q.x, q.y)
            best, bestd = None, None
            for ri, rr in enumerate(rpolys):
                bb = rbb[ri]
                if ri in used or bb is None:
                    continue
                m = _CODE_MARGIN_PT
                if not (bb[0] - m <= pt[0] <= bb[2] + m and bb[1] - m <= pt[1] <= bb[3] + m):
                    continue
                d = _poly_dist(pt, rr)
                if d <= m and (bestd is None or d < bestd):
                    best, bestd = ri, d
            if best is None:
                continue
            nm = f"{code} ({fam})"
            polys[best]["material"] = nm
            polys[best]["category"] = nm
            polys[best]["group"] = nm
            polys[best]["named_by"] = "code:" + code
            polys[best]["name_dist_pt"] = round(bestd, 1)
            used.add(best)
            named += 1

        # arrowheads in display coords, with a tip that points AT a region
        dheads = [{"tip": rp(*h["tip"]), "c": rp(h["cx"], h["cy"])} for h in heads]
        # FIX 6: grid-index the arrowhead centres. The old code ran min() over every
        # arrowhead on the page for EVERY leader endpoint (up to 2,000 x segments).
        hgrid = _RectGrid([(h["c"][0], h["c"][1], h["c"][0], h["c"][1]) for h in dheads],
                          cell=48.0) if dheads else None

        def head_near(pt, rad=22.0):
            if hgrid is None:
                return None
            best, bd = None, rad * rad
            for hi in hgrid.idx(pt[0], pt[1], rad):
                h = dheads[hi]
                d = (h["c"][0] - pt[0]) ** 2 + (h["c"][1] - pt[1]) ** 2
                if d <= bd:
                    bd, best = d, h
            return best

        def region_at(pt):
            for ri, rr in enumerate(rpolys):
                bb = rbb[ri]
                if ri in used or bb is None:
                    continue
                if not (bb[0] <= pt[0] <= bb[2] and bb[1] <= pt[1] <= bb[3]):
                    continue        # bbox reject before the O(vertices) crossing test
                if _pip(pt, rr):
                    return ri
            return None

        def region_near(pt, radius):
            """FIX 6 — a leader endpoint that lands in EMPTY SPACE still points at
            something. On 26-170 p1 the leader ends 424 pt from the nearest detected
            polygon: the callout is readable, the region under it was simply never
            detected, and the name is thrown away. Bind to the nearest region within
            `radius` and RECORD THE DISTANCE, so the confidence of the bind travels
            with it instead of being asserted. radius <= 0 disables the tier."""
            if radius <= 0:
                return None, None
            best, bestd = None, None
            for ri, rr in enumerate(rpolys):
                bb = rbb[ri]
                if ri in used or bb is None:
                    continue
                if not (bb[0] - radius <= pt[0] <= bb[2] + radius
                        and bb[1] - radius <= pt[1] <= bb[3] + radius):
                    continue
                d = _poly_dist(pt, rr)
                if d <= radius and (bestd is None or d < bestd):
                    best, bestd = ri, d
            return best, bestd

        def assign(ri, nm, how, dist=0.0):
            nm = _material_name(nm)
            if not nm:
                return False
            # a trim/railing/opening callout crossing a big wall points at a DETAIL, not the
            # wall — naming 4,800 SF of panel "PVC TRIM" would corrupt the takeoff
            if _LINEARISH_RE.search(nm) and (polys[ri].get("area_sf") or 0) > 250:
                return False
            polys[ri]["material"] = nm
            polys[ri]["category"] = nm
            polys[ri]["group"] = nm
            polys[ri]["named_by"] = how
            polys[ri]["name_dist_pt"] = round(float(dist), 1)
            used.add(ri)
            return True

        # 1) TEXT + LEADER, confidence-tiered per material block:
        #    (a) arrowhead-confirmed — a leader from the block ends at an arrowhead whose TIP
        #        sits in a region. This is the "read the arrow" path: only arrowheads paired
        #        with material text count, so detail-bubble/dimension arrows never mislabel.
        #    (b) plain leader — a leader segment from the block ends inside a region.
        #    (c) proximity — the block center sits inside a region.
        # FIX 6: index each leader by the callout block its TAIL sits at. Production
        # scanned every segment against every block; with the anchored scan every segment
        # is at SOME block, so that product would explode on a dense sheet. The index runs
        # the identical _near_rect(...,16) test once per endpoint and keeps the loop linear.
        seg_by_block = {}
        if segs and dblocks and ("leader" in _TIERS or "arrow" in _TIERS):
            bgrid = _RectGrid([b["rect"] for b in dblocks])
            for (x1, y1, x2, y2) in segs:
                a = rp(x1, y1); c2 = rp(x2, y2)
                for (p_from, p_to) in ((a, c2), (c2, a)):
                    for bi in bgrid.idx(p_from[0], p_from[1], 16):
                        if _near_rect(p_from, dblocks[bi]["rect"], 16):
                            seg_by_block.setdefault(bi, []).append((p_from, p_to))
        for bi, b in (enumerate(dblocks) if ("leader" in _TIERS or "arrow" in _TIERS) else ()):
            nm = b["text"]
            hit = False
            for (p_from, p_to) in seg_by_block.get(bi, ()):
                for _once in (0,):
                    # (a) is there an arrowhead near this leader's far end?
                    ah = head_near(p_to) if "arrow" in _TIERS else None
                    if ah is not None:
                        ri = region_at(ah["tip"])
                        if ri is None:
                            ri = region_at(p_to)
                        if ri is not None and assign(ri, nm, "arrow"):
                            hit = True; break
                        # (a2) FIX 6 — the arrow's tip lands in empty space
                        ri, dd = region_near(ah["tip"], _NEAR_BIND_PT)
                        if ri is not None and assign(ri, nm, "arrow~near", dd):
                            hit = True; break
                    # (b) plain leader end in a region
                    if "leader" not in _TIERS:
                        continue
                    ri = region_at(p_to)
                    if ri is not None and assign(ri, nm, "leader"):
                        hit = True; break
                    # (c) FIX 6 — the leader's far end lands in empty space: the region it
                    # points at was never detected. Bind to the nearest one within
                    # _NEAR_BIND_PT and carry the distance as the confidence.
                    ri, dd = region_near(p_to, _NEAR_BIND_PT)
                    if ri is not None and assign(ri, nm, "leader~near", dd):
                        hit = True; break
                if hit:
                    break
            # NOTE: no pure-proximity tier — a material-word note floating over a wall would
            # mislabel it, and a wrong name corrupts the takeoff. Only leader/arrow/legend name.
        return sum(1 for p in polys if p.get("named_by"))
    except Exception:
        return 0
