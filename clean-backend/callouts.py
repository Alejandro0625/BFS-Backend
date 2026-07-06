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
            if _NOISE_RE.search(t):
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


def _clean(t):
    t = re.sub(r"[•‣▪\-–—]\s*", "", t).strip(" .;:,")
    return t[:48]


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
        for bi, b in enumerate(dblocks):
            target = None
            # 1) leader: one end near the block, other end inside a region
            for (x1, y1, x2, y2) in segs:
                a = rp(x1, y1); c2 = rp(x2, y2)
                for (p_from, p_to) in ((a, c2), (c2, a)):
                    if _near_rect(p_from, b["rect"], 14):
                        for ri, rr in enumerate(rpolys):
                            if len(rr) >= 3 and _pip(p_to, rr):
                                target = ri; break
                    if target is not None:
                        break
                if target is not None:
                    break
            # 2) fallback: block center inside (or very near) a region
            if target is None:
                cx = (b["rect"][0] + b["rect"][2]) / 2
                cy = (b["rect"][1] + b["rect"][3]) / 2
                for ri, rr in enumerate(rpolys):
                    if len(rr) >= 3 and _pip((cx, cy), rr):
                        target = ri; break
            if target is None or target in used:
                continue
            nm = _clean(b["text"])
            if not nm:
                continue
            # a trim/railing callout crossing a big wall region is pointing at the DETAIL,
            # not the wall — naming 4,800 SF of panel "PVC TRIM" would corrupt the takeoff
            if _LINEARISH_RE.search(nm) and (polys[target].get("area_sf") or 0) > 250:
                continue
            polys[target]["material"] = nm
            polys[target]["category"] = nm
            polys[target]["group"] = nm
            polys[target]["named_by"] = "callout"
            used.add(target)
            named += 1
        # 3) LEGEND-KEY convention: an M5/PNL-1 tag on (or leadered to) the wall + a legend
        # mapping the key to its material name — the dominant convention on CD sets.
        doc2 = fitz.open(stream=pdf_bytes, filetype="pdf")
        pg2 = doc2[page_index]
        legend = _legend_pairs(pg2, blocks)
        keys = _key_words(pg2)
        rot2 = pg2.rotation_matrix
        doc2.close()
        if legend and keys:
            for (kx, ky, key) in keys:
                if key not in legend:
                    continue
                q = fitz.Point(kx, ky) * rot2
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
        return named
    except Exception:
        return 0
