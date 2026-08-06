"""MULTI-JOB GOLD BENCHMARK — 'comparing them all'.
Every past job with his real Bluebeam takeoff = a gold answer key. Strip the annots ->
synthetic clean drawing -> run the engine -> score every wall vs his SF + shape.
Outputs a per-wall and per-job report; the failures across MANY jobs tell us the next fix.
"""
import sys, os, re, json
sys.path.insert(0, r"C:\Users\User\Downloads\bfs-clean-backend")
import fitz
from shapely.geometry import Polygon

import vector_hatch, snap_fill

ROOT = r"V:\Bids 2026\Siding Bids 2026\00 - Submitted"
# 2026-07-29: office staff MOVE job folders between status dirs (26-025 Avita went
# Submitted -> "00 - Scope review" for a rebid; the frozen exam silently lost its
# 113 gold walls and EVERY engine scored 66/372 vs bar 140). Manifest-pinned runs
# now search these roots READ-ONLY in order (Submitted first); heuristic
# (no-manifest) runs still use ROOT only, so corpus discovery semantics are unchanged.
ROOTS = [ROOT,
         r"V:\Bids 2026\Siding Bids 2026\00 - Scope review",
         r"V:\Bids 2026\Siding Bids 2026\00 - WON",
         r"V:\Bids 2026\Siding Bids 2026\00 - Not Submitted"]
# 2026-07-30: under Task Scheduler's "run whether user is logged on" session the
# V: drive mapping DOES NOT EXIST (autopilot exam scored 0/372 all night after the
# logon-type change). Fall back to the UNC share transparently so every scheduler
# session grades the same archive.
_UNC_BASE = "\\\\192.168.168.2\\Boston"
if not os.path.isdir(ROOT):
    ROOT = _UNC_BASE + ROOT[2:]
    ROOTS = [_UNC_BASE + r[2:] for r in ROOTS]
OUT = os.environ.get("BENCH_OUT",
    r"C:\Users\User\AppData\Local\Temp\claude\C--Users-User--claude-projects-C--Users-User-Downloads\7923c776-90d9-4429-bc6e-2042f5ab0117\scratchpad\gold_bench_results.json")
MAX_JOBS = int(os.environ.get("BENCH_MAX_JOBS", "30"))     # defaults = the frozen v2 gate
_MANIFEST_FILE = os.environ.get("BENCH_MANIFEST", "gold_manifest.json")  # gold_manifest_full.json = every submitted bid
MAX_PAGES = 6          # per doc — elevations live early in elevation-only files
MAX_MB = 60

def gold_walls(pg):
    """EXAM v2 (2026-07-10, per the 11-job gold audit): two contamination filters —
    (1) 'AREA'/'Area Measurement' subjects are whole-floor GFA takeoffs, not walls
    (26-205A: 4 of 6 golds were GFA polygons, one duplicated verbatim);
    (2) identical polygons marked multiple times count ONCE, keeping the material-
    named copy (26-031: same polygon marked as Area + FC + sheathing).
    v1 baseline for comparison: 379 walls / found 177 / covered 167 / money 127."""
    W, H = pg.rect.width, pg.rect.height
    rot = pg.rotation_matrix
    out = []
    for a in (pg.annots() or []):
        if a.type[0] != 6 or not a.vertices or len(a.vertices) < 3:
            continue
        c = a.info.get("content", "") or ""
        m = re.search(r"([\d,]+(?:\.\d+)?)\s*sf", c, re.I)
        if not m:
            continue
        sf = float(m.group(1).replace(",", ""))
        if sf < 40 or sf > 60000:
            continue
        sub = (a.info.get("subject") or "").strip()
        su = sub.upper()
        if su == "AREA" or "AREA MEASUREMENT" in su:
            continue                      # GFA / generic area — not a wall
        pts = []
        for v in a.vertices:
            vx, vy = (v[0], v[1]) if isinstance(v, (list, tuple)) else (v.x, v.y)
            p = fitz.Point(vx, vy) * rot
            pts.append((p.x / W, p.y / H))
        out.append({"pts": pts, "sf": sf, "mat": sub[:30]})
    # dedupe identical polygons (same rounded vertex set) — keep the material-named copy
    seen = {}
    dedup = []
    for g in out:
        key = tuple(sorted((round(x, 4), round(y, 4)) for x, y in g["pts"]))
        prev = seen.get(key)
        if prev is None:
            seen[key] = len(dedup)
            dedup.append(g)
        elif not _MATWORD.search(dedup[prev]["mat"] or "") and _MATWORD.search(g["mat"] or ""):
            dedup[prev] = g               # replace the generic copy with the named one
    return dedup

_MATWORD = re.compile(r"panel|siding|lap\b|brick|eifs|metal|mtl|fiber|cement|stone|masonr|acm|pnl|side-|shake|board|batten|veneer|stucco|hardi|cedar|alum|corrug|soffit|azek|nichiha", re.I)


def find_marked_pdf(jobdir):
    """Pick the file whose gold is a real CLADDING takeoff: prefer elevation-named files
    and MATERIAL-named gold subjects; penalize site/logistics/civil sheets (26-040's
    'Site Logistics & Phasing' gold = roof-footprint areas, not walls — poisoned the exam)."""
    cands = []
    try:
        for f in os.listdir(jobdir):
            if not f.lower().endswith(".pdf"):
                continue
            fp = os.path.join(jobdir, f)
            mb = os.path.getsize(fp) / 1e6
            if mb > MAX_MB:
                continue
            score = 0
            fl = f.lower()
            if "elev" in fl: score += 10
            if any(k in fl for k in ("site", "logistic", "phas", "civil", "demo ", "landsc")):
                score -= 40
            if "no mark" in fl or "nomark" in fl or "clean" in fl: score -= 100
            cands.append((score, mb, fp))
    except Exception:
        return None
    scored = []
    for score, mb, fp in sorted(cands, key=lambda t: (-t[0], t[1]))[:8]:
        try:
            doc = fitz.open(fp)
            walls = [w for i in range(min(len(doc), MAX_PAGES)) for w in gold_walls(doc[i])]
            doc.close()
            if len(walls) < 2:
                continue
            n_mat = sum(1 for w in walls if _MATWORD.search(w["mat"] or ""))
            scored.append((score + 3 * min(n_mat, 8), fp, n_mat, len(walls)))
        except Exception:
            continue
    if not scored:
        return None
    scored.sort(key=lambda t: -t[0])
    # if the best file's gold has NO material-named walls at all it's probably not a
    # siding takeoff (footprints, phasing areas) — only accept it if nothing better exists
    return scored[0][1]

def strip_annots(doc):
    """Delete every annotation (popup children die with parents — tolerate)."""
    for pi in range(len(doc)):
        pg = doc[pi]
        for _ in range(200):
            a = pg.first_annot
            if not a:
                break
            try:
                pg.delete_annot(a)
            except Exception:
                break
    return doc


if __name__ != "__main__":
    import sys as _s
    _s.modules[__name__].__dict__.setdefault("results", [])
results = []
jobs_done = 0
if __name__ != "__main__":
    _RUN = False
else:
    _RUN = True
_MANIFEST = {}
try:
    _MANIFEST = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            _MANIFEST_FILE)))
except Exception:
    pass
_jobdirs = {}
if _RUN:
    for _r in (ROOTS if _MANIFEST else [ROOT]):
        try:
            for _j in os.listdir(_r):
                _jobdirs.setdefault(_j, _r)      # first root wins (Submitted preferred)
        except Exception:
            pass
_joblist = sorted(_jobdirs, reverse=bool(os.environ.get("BENCH_DESC")))
_graded_keys = set()      # a moved job can exist under two names — grade each key ONCE
for job in _joblist:
    if jobs_done >= MAX_JOBS:
        break
    jd = os.path.join(_jobdirs[job], job)
    if not os.path.isdir(jd):
        continue
    # FROZEN EXAM: the manifest pins job->file so every run grades the same test —
    # heuristic re-picking changed the exam twice and made runs incomparable.
    # Entries may be a filename string (gold = first MAX_PAGES pages, the v2 gate) or
    # a dict {"file":..., "pages":[1-based,...]} pinning WHERE the gold lives (the
    # full-corpus exam: takeoffs sit on pages 6-70 of the arch crops).
    pin_pages = None
    _mkey = None
    if _MANIFEST:
        _mkey = next((k for k in _MANIFEST if job[:44] == k), None)
        if _mkey is None or _mkey in _graded_keys:
            continue
        mf = _MANIFEST[_mkey]
        if isinstance(mf, dict):
            pin_pages = [int(p) - 1 for p in (mf.get("pages") or [])]
            mf = mf.get("file") or ""
        cand = [os.path.join(jd, f) for f in os.listdir(jd) if f.startswith(mf[:36])]
        fp = cand[0] if cand else None
    else:
        fp = find_marked_pdf(jd)
    if not fp:
        continue
    try:
        doc = fitz.open(fp)
        # gold per page
        gold_by_pg = {}
        page_iter = pin_pages if pin_pages else range(min(len(doc), MAX_PAGES))
        for pi in page_iter:
            if pi < 0 or pi >= len(doc):
                continue
            g = gold_walls(doc[pi])
            if g:
                gold_by_pg[pi] = g
        if not gold_by_pg:
            doc.close(); continue
        # SYNTHETIC CLEAN: strip every annotation
        strip_annots(doc)
        clean = doc.tobytes()
        doc.close()
    except Exception as e:
        continue
    jrec = {"job": job[:44], "file": os.path.basename(fp)[:40], "walls": [], "scale_conf": None}
    for pi, walls in gold_by_pg.items():
        try:
            pieces, VW, VH, sinfo = vector_hatch.detect(clean, pi)
        except Exception:
            pieces = []; sinfo = {}
        jrec["scale_conf"] = bool(sinfo.get("scale_confirmed"))
        ft_pt = (float(sinfo.get("ft_per_in") or 0) / 72.0) if sinfo.get("scale_confirmed") else None
        piece_polys = []
        for p in pieces:
            try:
                pp = Polygon([(x, y) for x, y in p["points"]]).buffer(0)
                if not pp.is_empty:
                    piece_polys.append((pp, p))
            except Exception:
                pass
        for g in walls:
            try:
                gp = Polygon(g["pts"]).buffer(0)
                if gp.is_empty:
                    continue
            except Exception:
                continue
            # ASSEMBLED scoring — what the takeoff actually totals for this wall: every
            # piece ≥50%-inside it, summed SF + union coverage. (Single-piece IoU can
            # never score 205A's one-sweep-per-elevation walls against our N pieces,
            # nor his one wall assembled from our splits — Fleet's lab metric, adopted.)
            mine = []
            for pp, p in piece_polys:
                try:
                    if pp.intersection(gp).area >= 0.5 * pp.area:
                        mine.append((pp, p))
                except Exception:
                    pass
            asf = sum(p.get("area_sf", 0) for _, p in mine)
            cov = 0.0
            try:
                from shapely.ops import unary_union
                if mine:
                    u = unary_union([pp for pp, _ in mine])
                    cov = u.intersection(gp).area / max(1e-9, gp.area)
            except Exception:
                pass
            # confidence features for the precision/abstention curve (pillar-1 of the
            # road to 100: the system must KNOW when it's right — iou alone measured
            # only 53% precision at the money gate; reader class + scale are the
            # candidate signals). Additive keys; bench_diff reads got/iou only.
            def _rdr(p):
                m = str(p.get("material") or "")
                for pre, nm in (("Hatched area", "hatch"), ("Color fill", "color"),
                                ("Rendered", "rc"), ("Wall area (AI boundary", "v13"),
                                ("Wall band", "band"), ("Wall area", "flood"),
                                ("Panel wall", "fill")):
                    if m.startswith(pre):
                        return nm
                return "train"
            jrec["walls"].append({"pg": pi + 1, "mat": g["mat"], "gold": g["sf"],
                                  "got": round(asf, 1), "iou": round(cov, 2),
                                  "n_pieces": len(mine),
                                  "readers": sorted(set(_rdr(p) for _, p in mine)),
                                  "named": any(p.get("named_by_tag") or p.get("stmt")
                                               for _, p in mine)})
        del piece_polys
    if jrec["walls"]:
        results.append(jrec)
        jobs_done += 1
        if _mkey is not None:
            _graded_keys.add(_mkey)
        ok = sum(1 for w in jrec["walls"] if w["iou"] >= 0.7 and abs(w["got"] - w["gold"]) <= 0.15 * w["gold"])
        print(f"[{jobs_done}] {jrec['job'][:40]:<42} walls {len(jrec['walls']):>3}  OK {ok:>3}  "
              f"scale_conf={jrec['scale_conf']}", flush=True)

if _RUN:
    json.dump(results, open(OUT, "w"), indent=1)
    # DENOMINATOR GUARD (2026-07-29 lesson: office moved 26-025's folder and the
    # exam silently graded 66/372 — a missing manifest job must be LOUD, never silent):
    if _MANIFEST:
        _graded_jobs = {r["job"] for r in results}
        _missing = [k for k in _MANIFEST
                    if not any(str(j).startswith(str(k)[:6]) for j in _graded_jobs)]
        if _missing:
            print(f"\n!!!! DENOMINATOR SHORTFALL: {len(results)}/{len(_MANIFEST)} manifest "
                  f"jobs graded — MISSING (folder moved/renamed?): {_missing}", flush=True)
    tw = sum(len(r["walls"]) for r in results)
    ok = sum(1 for r in results for w in r["walls"] if w["iou"] >= 0.7 and abs(w["got"] - w["gold"]) <= 0.15 * w["gold"])
    shape_ok = sum(1 for r in results for w in r["walls"] if w["iou"] >= 0.7)
    found = sum(1 for r in results for w in r["walls"] if w["iou"] >= 0.3)
    print(f"\n===== BENCHMARK: {len(results)} jobs, {tw} gold walls")
    print(f"  wall FOUND (cover>=0.3):        {found:>4}  ({100*found/max(tw,1):.0f}%)")
    print(f"  covered (cover>=0.7):       {shape_ok:>4}  ({100*shape_ok/max(tw,1):.0f}%)")
    print(f"  MONEY-RIGHT (cover+SF<=15%):  {ok:>4}  ({100*ok/max(tw,1):.0f}%)")
