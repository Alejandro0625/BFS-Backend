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


def _leaders(pg, min_len=25, max_len=600):
    """Plain-ish segments that could be leader shafts (any angle, medium length)."""
    segs = []
    try:
        for d in pg.get_drawings():
            for it in d["items"]:
                if it[0] != "l":
                    continue
                p1, p2 = it[1], it[2]
                L = ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5
                if min_len <= L <= max_len:
                    segs.append((p1.x, p1.y, p2.x, p2.y))
    except Exception:
        pass
    return segs[:4000]


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
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pg = doc[page_index]
        blocks = _blocks(pg)
        if not blocks:
            doc.close(); return 0
        rot = pg.rotation_matrix
        segs = _leaders(pg)
        legend = _legend_pairs(pg, blocks)
        keys = _key_words(pg)
        heads = _arrowheads(pg)
        doc.close()
        # region polygons in PAGE-display coords
        rpolys = [[(x * pw, y * ph) for x, y in p["points"]] for p in polys]
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
        if legend and keys:
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
        # arrowheads in display coords, with a tip that points AT a region
        dheads = [{"tip": rp(*h["tip"]), "c": rp(h["cx"], h["cy"])} for h in heads]

        def region_at(pt):
            for ri, rr in enumerate(rpolys):
                if ri not in used and len(rr) >= 3 and _pip(pt, rr):
                    return ri
            return None

        def assign(ri, nm, how):
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
            used.add(ri)
            return True

        # 1) TEXT + LEADER, confidence-tiered per material block:
        #    (a) arrowhead-confirmed — a leader from the block ends at an arrowhead whose TIP
        #        sits in a region. This is the "read the arrow" path: only arrowheads paired
        #        with material text count, so detail-bubble/dimension arrows never mislabel.
        #    (b) plain leader — a leader segment from the block ends inside a region.
        #    (c) proximity — the block center sits inside a region.
        for b in dblocks:
            nm = b["text"]
            hit = False
            for (x1, y1, x2, y2) in segs:
                a = rp(x1, y1); c2 = rp(x2, y2)
                for (p_from, p_to) in ((a, c2), (c2, a)):
                    if not _near_rect(p_from, b["rect"], 16):
                        continue
                    # (a) is there an arrowhead near this leader's far end?
                    ah = min(dheads, key=lambda h: (h["c"][0] - p_to[0]) ** 2 + (h["c"][1] - p_to[1]) ** 2, default=None) if dheads else None
                    if ah and (ah["c"][0] - p_to[0]) ** 2 + (ah["c"][1] - p_to[1]) ** 2 <= 22 ** 2:
                        ri = region_at(ah["tip"])
                        if ri is None:
                            ri = region_at(p_to)
                        if ri is not None and assign(ri, nm, "arrow"):
                            hit = True; break
                    # (b) plain leader end in a region
                    ri = region_at(p_to)
                    if ri is not None and assign(ri, nm, "leader"):
                        hit = True; break
                if hit:
                    break
            # NOTE: no pure-proximity tier — a material-word note floating over a wall would
            # mislabel it, and a wrong name corrupts the takeoff. Only leader/arrow/legend name.
        return sum(1 for p in polys if p.get("named_by"))
    except Exception:
        return 0
