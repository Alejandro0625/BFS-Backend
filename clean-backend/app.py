"""
BFS Estimator — clean lightweight backend.  (deploy build: clean-backend v1)
Reads the estimator's Bluebeam markup polygons (digitize-markup) → exact SF per material.
Light deps only (PyMuPDF + OpenCV + numpy) → boots instantly, never OOMs.
Matches the existing React frontend contract:
  POST /analyze (multipart 'pdf') -> {jobId}
  GET  /status/{jobId}            -> {status, phase, log, progress, legend, takeoffData, scheduleData, error}
  GET  /polygons/{jobId}/{page}   -> {polygons, width, height}
  GET  /page-image/{jobId}/{page} -> PNG
  GET  /health
"""
import os, io, re, uuid, threading, json, time
from collections import defaultdict
import fitz  # PyMuPDF
import texture  # full-res texture auto-detection for unmarked drawings
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Body
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

# Where corrections accumulate as training data (mount a Railway volume here to persist).
CORR_DIR = os.environ.get("CORRECTIONS_DIR", "/data/corrections")

app = FastAPI(title="BFS Clean Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
jobs = {}  # jobId -> dict

# ── helpers ────────────────────────────────────────────────────────────────
def shoelace(pts):
    n = len(pts); a = 0.0
    for i in range(n):
        x1, y1 = pts[i]; x2, y2 = pts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0

def scale_ft_per_in(s):
    if not s: return 8.0
    m = re.search(r'(\d+)\s*/\s*(\d+)', s)
    if m and int(m.group(1)) > 0: return int(m.group(2)) / int(m.group(1))
    m = re.search(r'([\d.]+)', s)
    if m:
        v = float(m.group(1))
        if 0 < v < 1: return 1.0 / v
    return 8.0

def categorize(subject):
    s = (subject or "").lower()
    if any(k in s for k in ["acm", "mcm", "composite"]): return "ACM/Composite Panel"
    if any(k in s for k in ["lap", "cementitious", "fiber cement", "fibercement", "nichiha"]): return "Fiber Cement / Lap"
    if any(k in s for k in ["soffit", "fascia", "trim", "return"]): return "Soffit/Trim"
    if any(k in s for k in ["brick", "masonry", "cmu"]): return "Masonry"
    if any(k in s for k in ["standing seam", "metal", "pnl", "panel", "alum"]): return "Metal Wall Panel"
    if any(k in s for k in ["shingle", "shake"]): return "Shingle/Shake"
    if "pvc" in s or "azek" in s: return "PVC/Trim"
    return subject or "Other"

def jlog(job, msg, level="info"):
    job["log"].append({"msg": msg, "level": level})

def extract_page_polygons(pg, pw, ph, ft_per_in):
    polys = []
    for a in (pg.annots() or []):
        if a.type[0] != 6:  # 6 = Polygon
            continue
        verts = a.vertices or []
        if len(verts) < 3:
            continue
        info = a.info or {}
        content = info.get("content", "") or ""
        subject = info.get("subject", "") or ""
        fill = a.colors.get("fill") if a.colors else None
        pts_pdf = [(v[0], v[1]) if isinstance(v, (list, tuple)) else (v.x, v.y) for v in verts]
        m = re.search(r"([\d,]+(?:\.\d+)?)\s*sf", content, re.I)
        if m:
            sf = float(m.group(1).replace(",", ""))
        else:
            sf = round(shoelace(pts_pdf) * (ft_per_in / 72.0) ** 2, 1)
        norm = [[round(x / pw, 5), round(y / ph, 5)] for (x, y) in pts_pdf]
        cx = round(sum(p[0] for p in norm) / len(norm), 5)
        cy = round(sum(p[1] for p in norm) / len(norm), 5)
        polys.append({
            "id": len(polys), "points": norm, "area_sf": sf, "cx": cx, "cy": cy,
            "fill_color": [round(c, 3) for c in fill] if fill else None,
            "source": "bluebeam", "material": subject or None,
            "category": categorize(subject), "label": content,
        })
    return polys

def process(jid, pdf_bytes):
    job = jobs[jid]
    try:
        job["status"] = "running"; job["phase"] = "loading"
        job["progress"] = {"label": "Loading PDF", "pct": 3}
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        n = doc.page_count
        jlog(job, f"PDF loaded — {n} page(s)", "ok")
        legend = {}
        for pi in range(n):
            job["progress"] = {"label": f"Reading page {pi+1} of {n}", "pct": 5 + int(90 * pi / max(n, 1))}
            job["phase"] = "analyzing"
            pg = doc[pi]; pw, ph = pg.rect.width, pg.rect.height
            ft = 8.0  # scale fallback; digitize SF comes from the markup's own labels
            polys = extract_page_polygons(pg, pw, ph, ft)
            auto = False
            if not polys:
                # no estimator markup on this page -> texture auto-detect (suggestions)
                try:
                    tpolys, _, _ = texture.detect(pdf_bytes, pi, ft_per_in=ft, zoom=2.0)
                    if tpolys:
                        polys = tpolys; auto = True
                except Exception as te:
                    jlog(job, f"Page {pi+1}: auto-detect skipped ({te})", "warn")
            job["polygons_by_page"][pi + 1] = polys
            job["dims_by_page"][pi + 1] = {"width": pw, "height": ph}
            if not polys:
                continue
            bymat = defaultdict(lambda: {"sf": 0.0, "n": 0, "category": None})
            for p in polys:
                key = p.get("material") or p.get("category") or "Unlabeled"
                bymat[key]["sf"] += p["area_sf"]; bymat[key]["n"] += 1
                bymat[key]["category"] = p.get("category")
            zones = []
            src_txt = "AI texture (confirm)" if auto else "markup"
            for mat, d in bymat.items():
                cat = d["category"] or "Other"
                zones.append({
                    "materialName": mat, "material_type": mat, "category": cat,
                    "netArea": round(d["sf"], 1), "grossArea": round(d["sf"], 1),
                    "totalOpeningArea": 0, "description": f"{d['n']} region(s) from {src_txt}",
                })
                legend[mat] = {"id": mat, "name": mat, "category": cat}
            job["takeoffData"].append({
                "pageNumber": pi + 1, "title": f"Sheet page {pi+1}", "sheetRef": f"p{pi+1}",
                "scale": "auto (calibrate)" if auto else "from markup",
                "scaleSource": "AI auto — verify" if auto else "estimator markup", "building": "Building",
                "zones": zones, "flags": (["AI suggestion — verify SF before bidding"] if auto else []),
                "source": "texture-auto" if auto else "digitize",
            })
            jlog(job, f"Page {pi+1}: " + (f"{len(polys)} AI-suggested zone(s)" if auto else f"{len(polys)} marked region(s)") + f", {len(zones)} material(s)", "warn" if auto else "ok")
        doc.close()
        job["legend"] = list(legend.values())
        total = sum(z["netArea"] for e in job["takeoffData"] for z in e["zones"])
        if not job["takeoffData"]:
            jlog(job, "No Bluebeam markup found on any page — load a marked-up drawing for digitize-markup.", "warn")
        else:
            jlog(job, f"Done — {len(job['takeoffData'])} page(s), {round(total):,} SF total", "success")
        job["status"] = "done"; job["phase"] = "done"
        job["progress"] = {"label": "Complete", "pct": 100}
    except Exception as e:
        import traceback; traceback.print_exc()
        job["status"] = "error"; job["phase"] = "error"; job["error"] = str(e)
        jlog(job, f"Error: {e}", "error")

# ── endpoints ──────────────────────────────────────────────────────────────
@app.post("/analyze")
async def analyze(background_tasks: BackgroundTasks, pdf: UploadFile = File(...)):
    data = await pdf.read()
    jid = str(int(__import__("time").time() * 1000))
    jobs[jid] = {"status": "queued", "phase": "idle", "log": [], "progress": {"label": "Queued", "pct": 0},
                 "legend": [], "takeoffData": [], "scheduleData": None, "error": None,
                 "polygons_by_page": {}, "dims_by_page": {}, "pdf": data}
    background_tasks.add_task(process, jid, data)
    return {"jobId": jid}

@app.get("/status/{jid}")
def status(jid: str):
    j = jobs.get(jid)
    if not j: raise HTTPException(404, "job not found")
    return {"status": j["status"], "phase": j.get("phase", ""), "log": j["log"], "progress": j["progress"],
            "legend": j.get("legend", []), "takeoffData": j.get("takeoffData", []),
            "scheduleData": j.get("scheduleData"), "error": j.get("error")}

@app.get("/polygons/{jid}/{page}")
def polygons(jid: str, page: int):
    j = jobs.get(jid)
    if not j: raise HTTPException(404, "job not found")
    dims = j.get("dims_by_page", {}).get(page) or {"width": 612, "height": 792}
    return {"polygons": j.get("polygons_by_page", {}).get(page, []),
            "width": dims["width"], "height": dims["height"]}

@app.get("/page-image/{jid}/{page}")
def page_image(jid: str, page: int):
    j = jobs.get(jid)
    if not j: raise HTTPException(404, "job not found")
    doc = fitz.open(stream=j["pdf"], filetype="pdf")
    if page < 1 or page > doc.page_count:
        doc.close(); raise HTTPException(404, "page out of range")
    pix = doc[page - 1].get_pixmap(matrix=fitz.Matrix(2, 2))  # ~144 dpi
    png = pix.tobytes("png"); doc.close()
    return Response(content=png, media_type="image/png")

@app.post("/learn")
def learn(payload: dict = Body(...)):
    """Capture a manual/corrected takeoff as labeled training data (the flywheel).
    payload: {jobId, page, shapes:[{points,name,color,type}], source}"""
    jid = payload.get("jobId")
    job = jobs.get(jid)
    try:
        os.makedirs(CORR_DIR, exist_ok=True)
        ts = str(int(time.time() * 1000))
        d = os.path.join(CORR_DIR, ts)
        os.makedirs(d, exist_ok=True)
        if job and job.get("pdf"):
            with open(os.path.join(d, "drawing.pdf"), "wb") as fh:
                fh.write(job["pdf"])
        with open(os.path.join(d, "labels.json"), "w", encoding="utf-8") as fh:
            json.dump({"jobId": jid, "page": payload.get("page"), "source": payload.get("source", "manual"),
                       "shapes": payload.get("shapes", []), "at": ts}, fh)
        n = len([x for x in os.listdir(CORR_DIR) if os.path.isdir(os.path.join(CORR_DIR, x))])
        return {"ok": True, "saved": ts, "total": n}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/learn-status")
def learn_status():
    try:
        n = len([x for x in os.listdir(CORR_DIR) if os.path.isdir(os.path.join(CORR_DIR, x))]) if os.path.isdir(CORR_DIR) else 0
    except Exception:
        n = 0
    return {"corrections": n}

@app.get("/health")
def health():
    return {"status": "ok", "engine": "digitize-markup", "deps": "pymupdf-light"}
