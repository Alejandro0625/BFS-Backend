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
MAX_JOBS = 12
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

def find_marked_pdf(jobdir):
    """Prefer elevation-named files; else smallest marked pdf with sf-polygons."""
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
            if "no mark" in fl or "nomark" in fl or "clean" in fl: score -= 100
            cands.append((score, mb, fp))
    except Exception:
        return None
    for score, mb, fp in sorted(cands, key=lambda t: (-t[0], t[1])):
        try:
            doc = fitz.open(fp)
            n_gold = sum(len(gold_walls(doc[i])) for i in range(min(len(doc), MAX_PAGES)))
            doc.close()
            if n_gold >= 2:
                return fp
        except Exception:
            continue
    return None

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
for job in (sorted(os.listdir(ROOT)) if _RUN else []):
    if jobs_done >= MAX_JOBS:
        break
    jd = os.path.join(ROOT, job)
    if not os.path.isdir(jd):
        continue
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
            best_iou, best_p = 0.0, None
            for pp, p in piece_polys:
                try:
                    iou = gp.intersection(pp).area / max(1e-9, gp.union(pp).area)
                except Exception:
                    iou = 0
                if iou > best_iou:
                    best_iou, best_p = iou, p
            bsf = best_p["area_sf"] if best_p else 0
            # geometric recompute of the matched piece (what a bucket click returns)
            if best_p is not None and ft_pt:
                try:
                    pp = Polygon([(x * VW, y * VH) for x, y in best_p["points"]]).buffer(0)
                    hp = sum(abs(Polygon([(q[0] * VW, q[1] * VH) for q in h]).area)
                             for h in (best_p.get("holes") or []) if len(h) >= 3)
                    bsf = round(max(0.0, pp.area - hp) * ft_pt * ft_pt, 1)
                except Exception:
                    pass
            jrec["walls"].append({"pg": pi + 1, "mat": g["mat"], "gold": g["sf"],
                                  "got": bsf, "iou": round(best_iou, 2)})
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
    print(f"  wall FOUND (IoU>=0.3):        {found:>4}  ({100*found/max(tw,1):.0f}%)")
    print(f"  shape RIGHT (IoU>=0.7):       {shape_ok:>4}  ({100*shape_ok/max(tw,1):.0f}%)")
    print(f"  MONEY-RIGHT (shape+SF<=15%):  {ok:>4}  ({100*ok/max(tw,1):.0f}%)")
