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
OUT = r"C:\Users\User\AppData\Local\Temp\claude\C--Users-User--claude-projects-C--Users-User-Downloads\7923c776-90d9-4429-bc6e-2042f5ab0117\scratchpad\gold_bench_results.json"
MAX_JOBS = 30
MAX_PAGES = 6          # per doc — elevations live early in elevation-only files
MAX_MB = 60

def gold_walls(pg):
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
        pts = []
        for v in a.vertices:
            vx, vy = (v[0], v[1]) if isinstance(v, (list, tuple)) else (v.x, v.y)
            p = fitz.Point(vx, vy) * rot
            pts.append((p.x / W, p.y / H))
        out.append({"pts": pts, "sf": sf, "mat": (a.info.get("subject") or "").strip()[:30]})
    return out

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
                                            "gold_manifest.json")))
except Exception:
    pass
for job in (sorted(os.listdir(ROOT)) if _RUN else []):
    if jobs_done >= MAX_JOBS:
        break
    jd = os.path.join(ROOT, job)
    if not os.path.isdir(jd):
        continue
    # FROZEN EXAM: the manifest pins job->file so every run grades the same test —
    # heuristic re-picking changed the exam twice and made runs incomparable
    if _MANIFEST:
        mf = next((v for k, v in _MANIFEST.items() if job[:44] == k), None)
        if mf is None:
            continue
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
        for pi in range(min(len(doc), MAX_PAGES)):
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
            jrec["walls"].append({"pg": pi + 1, "mat": g["mat"], "gold": g["sf"],
                                  "got": round(asf, 1), "iou": round(cov, 2),
                                  "n_pieces": len(mine)})
        del piece_polys
    if jrec["walls"]:
        results.append(jrec)
        jobs_done += 1
        ok = sum(1 for w in jrec["walls"] if w["iou"] >= 0.7 and abs(w["got"] - w["gold"]) <= 0.15 * w["gold"])
        print(f"[{jobs_done}] {jrec['job'][:40]:<42} walls {len(jrec['walls']):>3}  OK {ok:>3}  "
              f"scale_conf={jrec['scale_conf']}", flush=True)

if _RUN:
    json.dump(results, open(OUT, "w"), indent=1)
    tw = sum(len(r["walls"]) for r in results)
    ok = sum(1 for r in results for w in r["walls"] if w["iou"] >= 0.7 and abs(w["got"] - w["gold"]) <= 0.15 * w["gold"])
    shape_ok = sum(1 for r in results for w in r["walls"] if w["iou"] >= 0.7)
    found = sum(1 for r in results for w in r["walls"] if w["iou"] >= 0.3)
    print(f"\n===== BENCHMARK: {len(results)} jobs, {tw} gold walls")
    print(f"  wall FOUND (cover>=0.3):        {found:>4}  ({100*found/max(tw,1):.0f}%)")
    print(f"  covered (cover>=0.7):       {shape_ok:>4}  ({100*shape_ok/max(tw,1):.0f}%)")
    print(f"  MONEY-RIGHT (cover+SF<=15%):  {ok:>4}  ({100*ok/max(tw,1):.0f}%)")
