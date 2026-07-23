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
import os, io, re, uuid, threading, json, time, shutil
from collections import defaultdict
import fitz  # PyMuPDF
import texture  # classical-CV texture fallback for unmarked drawings
import model_infer  # trained cladding-extent model (ONNX) — the real auto-markup for RAW drawings
import vector_hatch  # reads the DRAWN pattern vectors (seam trains + gray fills) — exact, preferred on clean pages
import callouts  # reads the drawing's own text callouts + leader arrows -> names the regions
import density_reader  # dense-microtexture suggestions (Callahan class)
import ocr_text  # OCR fallback (onnxruntime RapidOCR) for FLATTENED sets — lazy, memory-safe
import snap_fill  # coloring-book BUCKET fill + corner-snap → exact SF from vector geometry (assist layer)
import material_groups  # within-job texture grouping → a selectable PREVIEW of material groups (assist layer)
import auto_trim as auto_trim_mod  # derive corner/base/opening LF from face geometry (blueprint 1c) — suggestions only
import dim_scale  # self-calibrate scale from the drawing's own dimension strings (blueprint 1b) — cross-check only
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Body
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware

# Where corrections accumulate as training data (mount a Railway volume here to persist).
CORR_DIR = os.environ.get("CORRECTIONS_DIR", "/data/corrections")
# Durable job store — jobs are also written here so takeoffs survive a restart/redeploy.
JOBS_DIR = os.environ.get("JOBS_DIR", "/data/jobs")
MAX_MEM_JOBS = int(os.environ.get("MAX_MEM_JOBS", "20"))   # cap RAM: never OOM (evicted jobs live on disk)
MAX_DISK_JOBS = int(os.environ.get("MAX_DISK_JOBS", "200"))  # cap volume: prune oldest persisted jobs
MAX_AUTO_PAGES = int(os.environ.get("MAX_AUTO_PAGES", "30"))  # cap texture auto-detect work per PDF

app = FastAPI(title="BFS Clean Backend")
# CORS: browsers may only call this API from the app's own origins (the Vercel prod
# domain + its preview builds + local dev). Non-browser clients (battery scripts,
# curl) send no Origin header and are unaffected — CORS is a browser-side gate.
app.add_middleware(CORSMiddleware,
                   allow_origins=["https://bfs-estimator.vercel.app",
                                  "http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_origin_regex=r"https://bfs-estimator[a-z0-9\-]*\.vercel\.app",
                   allow_methods=["*"], allow_headers=["*"])

# ---- APP LOCK (#32, ships DARK): set env BFS_APP_KEY on Railway to require
# X-BFS-Key on every request (health + CORS preflight exempt). Unset = open,
# exactly today's behavior. Flip = one env var, no deploy.
_APP_KEY = os.environ.get("BFS_APP_KEY", "")

@app.middleware("http")
async def _app_lock(request, call_next):
    if _APP_KEY and request.method != "OPTIONS" and request.url.path != "/health":
        supplied = request.headers.get("x-bfs-key") or request.query_params.get("key")
        if supplied != _APP_KEY:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "locked"}, status_code=401)
    return await call_next(request)
jobs = {}  # jobId -> dict  (in-memory hot cache, bounded to MAX_MEM_JOBS)

# ── durable, bounded job store ───────────────────────────────────────────────
# The old code kept every uploaded PDF in `jobs` forever → unbounded RAM → OOM.
# Now: keep a small hot set in RAM, persist each job to the volume, rehydrate on demand.
def _job_dir(jid): return os.path.join(JOBS_DIR, str(jid))

BIG_PDF_BYTES = int(os.environ.get("BIG_PDF_BYTES", str(200 * 1024 * 1024)))

def _load_pdf_bytes(path):
    """HEAVY-DOC SAFETY, phase 1: uploads now STREAM to disk (no doubled RAM while a
    giant uploads+parses). Processing still loads bytes — PyMuPDF rejects mmap objects
    ('bad stream: mmap.mmap', tested 2026-07-15), so true zero-copy needs the
    path-based refactor (every fitz.open(stream=...) site accepts a file path via a
    small _open() helper). That is phase 2 — its own gated cycle."""
    with open(path, "rb") as fh:
        return fh.read()

def _persist_job(jid):
    """Snapshot a job to the volume so it survives eviction + restarts. Best-effort (no-op if no volume)."""
    job = jobs.get(jid)
    if not job: return
    try:
        d = _job_dir(jid); os.makedirs(d, exist_ok=True)
        if job.get("pdf") and not os.path.exists(os.path.join(d, "input.pdf")):
            with open(os.path.join(d, "input.pdf"), "wb") as fh: fh.write(job["pdf"])
        meta = {k: job.get(k) for k in ("status", "phase", "progress", "legend", "takeoffData",
                "scheduleData", "error", "polygons_by_page", "dims_by_page", "log", "projName", "pageCount")}
        with open(os.path.join(d, "job.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh)
    except Exception:
        pass  # persistence is a bonus; the app still works from RAM without a volume

def _load_job(jid):
    d = _job_dir(jid)
    if not os.path.isdir(d): return None
    try:
        with open(os.path.join(d, "job.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        for k in ("polygons_by_page", "dims_by_page"):  # JSON stringifies int page keys → restore them
            meta[k] = {int(kk): vv for kk, vv in (meta.get(k) or {}).items()}
        p = os.path.join(d, "input.pdf")
        meta["pdf"] = _load_pdf_bytes(p) if os.path.exists(p) else None
        return meta
    except Exception:
        return None

def get_job(jid):
    """Hot cache first, else rehydrate the job from the durable store (survives restart/eviction)."""
    j = jobs.get(jid)
    if j is None:
        j = _load_job(jid)
        if j is not None: jobs[jid] = j
    return j

def _evict_mem():
    """Keep RAM bounded — drop oldest *finished* jobs (they persist on disk, rehydrate on access)."""
    if len(jobs) <= MAX_MEM_JOBS: return
    for jid in list(jobs.keys()):
        if len(jobs) <= MAX_MEM_JOBS: break
        if jobs[jid].get("status") in ("done", "error"):
            _persist_job(jid); jobs.pop(jid, None)

MAX_DISK_GB = float(os.environ.get("MAX_DISK_GB", "20"))   # heavy-doc safety: cap SIZE too

def _gc_disk():
    """Keep the volume bounded — prune oldest job folders by COUNT and by TOTAL SIZE
    (200 giant sets could fill the volume long before the count cap fires)."""
    try:
        if not os.path.isdir(JOBS_DIR): return
        dirs = [os.path.join(JOBS_DIR, d) for d in os.listdir(JOBS_DIR)]
        dirs = [d for d in dirs if os.path.isdir(d)]
        def _dsize(d):
            t = 0
            try:
                for f in os.listdir(d):
                    t += os.path.getsize(os.path.join(d, f))
            except Exception:
                pass
            return t
        dirs.sort(key=lambda d: os.path.getmtime(d))
        total = sum(_dsize(d) for d in dirs)
        budget = MAX_DISK_GB * 1024 ** 3
        while dirs and (len(dirs) > MAX_DISK_JOBS or total > budget):
            d = dirs.pop(0)
            total -= _dsize(d)
            shutil.rmtree(d, ignore_errors=True)
    except Exception:
        pass

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
    rot = pg.rotation_matrix                       # ROTATED pages (e.g. 270°) store annot vertices in unrotated coords;
    for a in (pg.annots() or []):                  # apply the page rotation so shapes align to the drawing (identity if rotation=0)
        if a.type[0] != 6:  # 6 = Polygon
            continue
        verts = a.vertices or []
        if len(verts) < 3:
            continue
        info = a.info or {}
        content = info.get("content", "") or ""
        subject = info.get("subject", "") or ""
        fill = a.colors.get("fill") if a.colors else None
        pts_pdf = []
        for v in verts:
            vx, vy = (v[0], v[1]) if isinstance(v, (list, tuple)) else (v.x, v.y)
            p = fitz.Point(vx, vy) * rot
            pts_pdf.append((p.x, p.y))
        m = re.search(r"([\d,]+(?:\.\d+)?)\s*sf", content, re.I)
        if m:
            sf = float(m.group(1).replace(",", "")); sf_exact = True   # estimator's own measured SF — ground truth
        else:
            sf = round(shoelace(pts_pdf) * (ft_per_in / 72.0) ** 2, 1); sf_exact = False  # geometry estimate (needs scale)
        norm = [[round(x / pw, 5), round(y / ph, 5)] for (x, y) in pts_pdf]
        cx = round(sum(p[0] for p in norm) / len(norm), 5)
        cy = round(sum(p[1] for p in norm) / len(norm), 5)
        polys.append({
            "id": len(polys), "points": norm, "area_sf": sf, "cx": cx, "cy": cy,
            "fill_color": [round(c, 3) for c in fill] if fill else None,
            "source": "bluebeam", "material": subject or None,
            "category": categorize(subject), "label": content, "sf_exact": sf_exact,
        })
    return polys

def _feet(s):
    """Parse feet-inches ('43\\'-3\"', '12\\'-0', '7\\'-2\"') → float feet."""
    m = re.search(r"(\d+)\s*'\s*-?\s*(\d+)?", s or "")
    if not m:
        return None
    return int(m.group(1)) + (int(m.group(2)) if m.group(2) else 0) / 12.0

def _height_of(subj):
    m = re.search(r"H\s*[:=]\s*(\d+)\s*'\s*-?\s*(\d+)?", subj or "", re.I)
    if not m:
        return None
    return int(m.group(1)) + (int(m.group(2)) if m.group(2) else 0) / 12.0

def _explicit_sf(s):
    m = re.search(r"-?\s*([\d,]+(?:\.\d+)?)\s*sf", s or "", re.I)
    return float(m.group(1).replace(",", "")) if m else None

def _clean_material(subj):
    return re.sub(r"\s*\(.*?\)\s*", " ", re.sub(r"-?\d+sf", "", subj or "", flags=re.I)).strip(" -:") or "Unlabeled"

def extract_page_polylines(pg, pw, ph):
    """Capture the estimator's LINEAR measurements (Bluebeam type-7 polylines) that the polygon path misses.
    Length is exact (in the label). Height in the subject '(H:..)' → SF for cladding runs; else LF (trim/soffit).
    Returns (sf_polys, lf_items): sf_polys look like polygon dicts (added to the takeoff SF total); lf_items are {material, lf}."""
    sf_polys, lf_items = [], []
    for a in (pg.annots() or []):
        if a.type[0] != 7:  # 7 = PolyLine (length measurement)
            continue
        info = a.info or {}
        content = info.get("content", "") or ""
        subject = info.get("subject", "") or ""
        length_ft = _feet(content)
        if not length_ft:  # not a length measurement (e.g. a plain leader line) → skip
            continue
        mat = _clean_material(subject)
        esf = _explicit_sf(subject) or _explicit_sf(content)
        h = _height_of(subject)
        stroke = a.colors.get("stroke") if a.colors else None
        if esf or h:  # a CLADDING run → SF (explicit, else length × wall height)
            sf = round(esf if esf else length_ft * h, 1)
            verts = a.vertices or []
            pts_pdf = [( (fitz.Point(v[0], v[1]) if isinstance(v, (list, tuple)) else fitz.Point(v.x, v.y)) * pg.rotation_matrix ) for v in verts]
            pts_pdf = [(p.x, p.y) for p in pts_pdf]
            norm = [[round(x / pw, 5), round(y / ph, 5)] for (x, y) in pts_pdf] or [[0, 0]]
            cx = round(sum(p[0] for p in norm) / len(norm), 5)
            cy = round(sum(p[1] for p in norm) / len(norm), 5)
            sf_polys.append({
                "points": norm, "area_sf": sf, "cx": cx, "cy": cy,
                "fill_color": [round(c, 3) for c in stroke] if stroke else None,
                "source": "bluebeam-linear", "material": mat, "category": categorize(subject),
                "label": f"{content} × {round(h,1) if h else '?'}' = {sf:,.0f} SF", "sf_exact": True,
                "length_ft": round(length_ft, 1),
            })
        else:  # trim / soffit / fascia → LINEAR feet (priced per LF, no SF)
            lf_items.append({"material": mat, "lf": round(length_ft, 1)})
    return sf_polys, lf_items

def flag_label_outliers(polys, pw, ph):
    """Money-safety: catch a typo'd/mislabeled SF before it reaches a bid.
    Every marked polygon on ONE sheet shares ONE scale, so back it out per-poly and flag any
    whose typed SF label disagrees with its geometry (area is rotation-invariant → safe).
    Non-destructive: only sets a warning, never changes an SF. Returns page-level warning strings."""
    scales = []
    for p in polys:
        if p.get("source") != "bluebeam" or not p.get("sf_exact") or p.get("area_sf", 0) <= 0:
            continue  # only true area polygons — linear runs have no meaningful shoelace area
        sh = shoelace(p["points"]) * pw * ph  # normalized-shoelace → PDF points^2
        if sh > 0:
            scales.append(((72.0 * (p["area_sf"] / sh) ** 0.5), p))  # implied ft-per-inch
    if len(scales) < 3:  # need a few labeled regions to establish the sheet's true scale
        return []
    med = sorted(s for s, _ in scales)[len(scales) // 2]
    if med <= 0:
        return []
    warns = []
    for s, p in scales:
        r = s / med
        if r > 1.32 or r < 0.76:  # area off by >~1.75x vs the sheet's consistent scale = almost certainly a mistake
            implied = p["area_sf"] * (med / s) ** 2
            p["sf_warn"] = True
            p["sf_warn_msg"] = f"labeled {p['area_sf']:,.0f} SF but this sheet's scale implies ≈{implied:,.0f} SF — verify"
            warns.append(f"{p.get('material') or p.get('category') or 'A region'}: " + p["sf_warn_msg"])
    return warns

def is_elevation_page(pg):
    """Auto-crop to the pages the estimator actually measures = elevations, returns, soffits
    (her step 4-5). Skip plans/details/schedules/cover so we don't invent cladding there."""
    try:
        t = (pg.get_text() or "").lower()
    except Exception:
        return False
    if ("roof plan" in t) or ("floor plan" in t):
        return False
    return ("elevation" in t) or ("return" in t) or ("soffit" in t)

def _page_struct_lines(pg):
    """Long H/V vector lines of the page (same rule as /snap-points): the building's real geometry."""
    W, H = pg.rect.width, pg.rect.height
    Lmin = 0.06 * max(W, H)
    hlines, vlines = [], []
    try:
        for d in pg.get_drawings():
            for it in d["items"]:
                segs = []
                if it[0] == "l":
                    segs = [(it[1].x, it[1].y, it[2].x, it[2].y)]
                elif it[0] == "re":
                    r = it[1]; segs = [(r.x0, r.y0, r.x1, r.y0), (r.x1, r.y0, r.x1, r.y1),
                                       (r.x1, r.y1, r.x0, r.y1), (r.x0, r.y1, r.x0, r.y0)]
                for (x1, y1, x2, y2) in segs:
                    if (x2 - x1) ** 2 + (y2 - y1) ** 2 < Lmin ** 2:
                        continue
                    if abs(y2 - y1) < abs(x2 - x1) * 0.02:
                        hlines.append((min(x1, x2), max(x1, x2), (y1 + y2) / 2))
                    elif abs(x2 - x1) < abs(y2 - y1) * 0.02:
                        vlines.append((min(y1, y2), max(y1, y2), (x1 + x2) / 2))
    except Exception:
        pass
    return hlines[:600], vlines[:600]

def snap_auto_polys(pg, polys, pw, ph):
    """The model paints a pixel mask — its outline is blobby and ignores the CAD geometry that is
    RIGHT THERE in the vector file. Make the AI's shapes look like an estimator drew them:
    simplify the contour, then snap each vertex to the drawing's real corners/lines (Bluebeam-style).
    SF is rescaled by the area ratio so the mask's net-of-openings measurement is preserved."""
    import numpy as np, cv2
    hlines, vlines = _page_struct_lines(pg)
    if not hlines and not vlines:
        return polys
    corners = []
    for (xa, xb, yh) in hlines:
        for (ya, yb, xv) in vlines:
            if xa - 2 <= xv <= xb + 2 and ya - 2 <= yh <= yb + 2:
                corners.append((xv, yh))
    corners = np.array(corners[:4000], dtype=np.float32) if corners else None
    hy = np.array([y for (_, _, y) in hlines], dtype=np.float32) if hlines else None
    vx = np.array([x for (_, _, x) in vlines], dtype=np.float32) if vlines else None
    tol = 0.012 * max(pw, ph)   # snap radius ~1.2% of the sheet
    out = []
    for p in polys:
        try:
            pts = np.array([(x * pw, y * ph) for x, y in p["points"]], dtype=np.float32)
            if len(pts) < 3:
                out.append(p); continue
            orig_area = abs(cv2.contourArea(pts.reshape(-1, 1, 2)))
            appr = cv2.approxPolyDP(pts.reshape(-1, 1, 2), 0.012 * cv2.arcLength(pts.reshape(-1, 1, 2), True), True).reshape(-1, 2)
            if len(appr) < 3:
                out.append(p); continue
            snapped = []
            for (x, y) in appr:
                nx, ny = float(x), float(y)
                hit = False
                if corners is not None and len(corners):
                    d = np.hypot(corners[:, 0] - nx, corners[:, 1] - ny)
                    i = int(d.argmin())
                    if d[i] < tol:
                        nx, ny = float(corners[i][0]), float(corners[i][1]); hit = True
                if not hit:  # no true corner nearby — align edges to the nearest structural line
                    if vx is not None and len(vx):
                        dv = np.abs(vx - nx); i = int(dv.argmin())
                        if dv[i] < tol: nx = float(vx[i])
                    if hy is not None and len(hy):
                        dh = np.abs(hy - ny); i = int(dh.argmin())
                        if dh[i] < tol: ny = float(hy[i])
                if not snapped or abs(snapped[-1][0] - nx) > 1 or abs(snapped[-1][1] - ny) > 1:
                    snapped.append((nx, ny))
            if len(snapped) < 3:
                out.append(p); continue
            new_area = abs(cv2.contourArea(np.array(snapped, dtype=np.float32).reshape(-1, 1, 2)))
            q = dict(p)
            q["points"] = [[round(x / pw, 5), round(y / ph, 5)] for (x, y) in snapped]
            if orig_area > 0 and new_area > 0:
                q["area_sf"] = round(p.get("area_sf", 0) * new_area / orig_area, 1)
            out.append(q)
        except Exception:
            out.append(p)
    return out

def process(jid, pdf_bytes):
    job = jobs[jid]
    try:
        job["status"] = "running"; job["phase"] = "loading"
        job["progress"] = {"label": "Loading PDF", "pct": 3}
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        n = doc.page_count
        job["pageCount"] = n   # Draw tab needs the full page list — photo/sparse sheets have no takeoffData entry
        jlog(job, f"PDF loaded — {n} page(s)", "ok")
        legend = {}
        auto_used = 0  # cap expensive texture renders so a big unmarked PDF can't blow memory
        # If the estimator marked ANY page, this is a DIGITIZE job — don't texture-auto the other
        # pages (they're plans/details, not cladding) or we inflate the bid with invented SF.
        # Count only annotations the digitize path actually READS as measurements: a polygon
        # with real vertices, or a polyline whose label carries a length. Stray leader lines,
        # clouds, or review stamps in an issued set must NOT disable auto-detect (that made a
        # "no markups" full set return a completely empty takeoff).
        doc_has_markup = False
        for pi in range(n):
            try:
                for a in (doc[pi].annots() or []):
                    t = a.type[0]
                    if t == 6 and len(a.vertices or []) >= 3:
                        doc_has_markup = True; break
                    if t == 7 and _feet((a.info or {}).get("content", "") or ""):
                        doc_has_markup = True; break
            except Exception:
                pass
            if doc_has_markup:
                break
        # auto-crop: which pages are elevations? (only mark those; if none detectable, fall back to all)
        # Text-based classification only works if the PDF actually HAS text — many issued sets
        # flatten CAD text to curves (a 77-page set had text on 1 page). If too few pages carry
        # text, the filter is meaningless: turn it OFF and try every page instead of page 6 only.
        text_pages = sum(1 for pi in range(n) if len(doc[pi].get_text() or "") > 150)
        text_reliable = text_pages >= max(3, int(n * 0.3))
        doc_any_elevation = text_reliable and any(is_elevation_page(doc[pi]) for pi in range(n))
        if not text_reliable and n > 1:
            jlog(job, f"Drawing text not extractable ({text_pages}/{n} pages with text) — scanning every page for cladding", "warn")
        auto_tried = 0; auto_hits = 0
        for pi in range(n):
            job["progress"] = {"label": f"Reading page {pi+1} of {n}", "pct": 5 + int(90 * pi / max(n, 1))}
            job["phase"] = "analyzing"
            pg = doc[pi]; pw, ph = pg.rect.width, pg.rect.height
            ft = 8.0  # scale fallback; digitize SF comes from the markup's own labels
            polys = extract_page_polygons(pg, pw, ph, ft)
            lin_sf_polys, lin_lf_items = extract_page_polylines(pg, pw, ph)  # estimator's linear measurements
            polys = polys + lin_sf_polys
            for i, p in enumerate(polys):
                p["id"] = i  # unique ids across polygons + linear runs
            auto = False; scale_conf = True; scale_val = None; auto_engine = None
            page_is_elev = is_elevation_page(pg) or not doc_any_elevation  # only auto-mark elevations (or all if none detectable)
            if not polys and not doc_has_markup and page_is_elev:
                # RAW/clean page -> auto-markup. Engine order: VECTOR (reads the drawn pattern —
                # exact geometry, openings netted; cheap, so EVERY page gets it) -> trained MODEL
                # -> texture heuristics (heavy, capped so a 77-page set can't stall the worker).
                try:
                    auto_tried += 1
                    tpolys = []; sinfo = {}
                    try:
                        tpolys, _, _, sinfo = vector_hatch.detect(pdf_bytes, pi)
                        auto_engine = "vector"
                    except Exception:
                        tpolys = []
                    if not tpolys and auto_used < MAX_AUTO_PAGES:
                        if model_infer.available():
                            tpolys, _, _, sinfo = model_infer.detect(pdf_bytes, pi, zoom=2.0)
                            auto_engine = "model"
                        else:
                            tpolys, _, _, sinfo = texture.detect(pdf_bytes, pi, ft_per_in=ft, zoom=2.0)
                            auto_engine = "texture"
                        if tpolys:
                            try:
                                tpolys = snap_auto_polys(pg, tpolys, pw, ph)  # lock AI shapes to the drawing's real corners
                            except Exception:
                                pass
                    if tpolys:
                        try:  # read the architect's own labels: callout text + leader arrows -> region names
                            n_named = callouts.name_regions(pdf_bytes, pi, tpolys, pw, ph)
                            if n_named:
                                jlog(job, f"Page {pi+1}: {n_named} region(s) named from the drawing's callouts", "ok")
                        except Exception:
                            pass
                        polys = tpolys; auto = True; auto_used += 1; auto_hits += 1
                        scale_conf = bool(sinfo.get("scale_confirmed")); scale_val = sinfo.get("ft_per_in")
                except Exception as te:
                    jlog(job, f"Page {pi+1}: auto-detect skipped ({te})", "warn")
            sf_warns = flag_label_outliers(polys, pw, ph) if not auto else []  # catch typo'd markup labels
            # RASTER-PAGE v13 SUGGESTIONS (2026-07-21; the 5-iteration reclaim verdict:
            # no runtime signal separates junk floods from real walls, so the HUMAN is
            # the discriminator). On raster-underlay pages, run the boundary model with
            # EMPTY ownership and surface its pieces as suggest_only: never counted in
            # zones/totals/Excel/evidence until the estimator accepts each one.
            if not doc_has_markup and page_is_elev:
                # (runs even when auto-detect found NOTHING — the emptiest raster
                # pages need suggestions most; acid test: p19 had 0 pieces, 0 sugs)
                try:
                    _imga = 0.0
                    for _im9 in pg.get_images(full=True):
                        for _r9 in pg.get_image_rects(_im9[0]):
                            _imga += max(0, _r9.width) * max(0, _r9.height)
                    _sugs = []
                    # DENSITY FIRST (Callahan acid 2026-07-21: dense hairline pages often
                    # ALSO carry a backdrop image, so the raster test stole them and — via
                    # an indent bug — dropped the pieces; 40k hairlines is the more
                    # specific signature, so it wins the tie).
                    _dense9 = density_reader.is_dense_page(pg)
                    if _dense9:
                        # Callahan-class micro-texture: the density reader's fields
                        # become confirmable suggestions (probe-proven 22/114 ensemble)
                        _sugs = density_reader.suggest_pieces(pdf_bytes, pi, pw, ph, max_new=40)
                    elif _imga >= 0.25 * pw * ph:
                        _ftpt9 = float(scale_val or 8.0) / 72.0
                        _sugs = vector_hatch._v13_regions(pdf_bytes, pi, [], pw, ph, _ftpt9, max_new=40)
                    jlog(job, f"Page {pi+1}: suggestion probe — dense={_dense9}, "
                              f"img={_imga / max(pw * ph, 1):.0%}, suggestions={len(_sugs)}", "info")
                    # tag + append OUTSIDE the branches — BOTH readers' suggestions ship
                    # (the indent bug had raster suggestions computed then discarded)
                    for _s9 in _sugs:
                        _s9["suggest_only"] = True
                        _s9["material"] = "AI suggestion (confirm to add)"
                        _s9["category"] = "AI suggestion (confirm to add)"
                        _s9["group"] = "AI suggestion (confirm to add)"
                        _s9["sf_exact"] = False
                    if _sugs:
                        polys = polys + _sugs
                        auto_flags_pre = f"🤖 {len(_sugs)} AI wall suggestion(s) on this page — dashed outlines; click Accept on the ones that are real walls. NOT counted until accepted."
                    else:
                        auto_flags_pre = None
                except Exception as _se9:
                    jlog(job, f"Page {pi+1}: suggestion probe FAILED — {type(_se9).__name__}: {_se9}", "warn")
                    auto_flags_pre = None
            else:
                auto_flags_pre = None
            job["polygons_by_page"][pi + 1] = polys
            job["dims_by_page"][pi + 1] = {"width": pw, "height": ph}
            if not polys and not lin_lf_items:  # keep pages that have linear (trim/LF) measurements even with no area polygons
                continue
            bymat = defaultdict(lambda: {"sf": 0.0, "n": 0, "category": None})
            for p in polys:
                if p.get("suggest_only"):
                    continue          # suggestions NEVER enter zones/totals until accepted
                key = p.get("material") or p.get("category") or "Unlabeled"
                bymat[key]["sf"] += p["area_sf"]; bymat[key]["n"] += 1
                bymat[key]["category"] = p.get("category")
            zones = []
            src_txt = ({"vector": "drawing vectors (exact geometry)", "model": "AI model (confirm)"}.get(auto_engine, "AI texture (confirm)")) if auto else "markup"
            for mat, d in bymat.items():
                cat = d["category"] or "Other"
                zones.append({
                    "materialName": mat, "material_type": mat, "category": cat,
                    "netArea": round(d["sf"], 1), "grossArea": round(d["sf"], 1),
                    "totalOpeningArea": 0, "description": f"{d['n']} region(s) from {src_txt}",
                })
                legend[mat] = {"id": mat, "name": mat, "category": cat}
            auto_flags = []
            if auto_flags_pre:
                auto_flags.append(auto_flags_pre)
            page_levels = []
            if auto:
                # SOFFIT/RETURN SENTINEL — the historical money-loser was FORGETTING these
                # (they hide in section views where area readers don't reach). If the sheet
                # mentions them, the takeoff refuses to let them be forgotten.
                try:
                    _t = (pg.get_text() or "").upper()
                    _hits = [k for k in ("SOFFIT", "CANOPY", "RETURN") if k in _t]
                    if _hits:
                        auto_flags.append("⚠ " + "/".join(h.title() + "s" for h in _hits) +
                                          " on this job — often drawn in SECTION views the auto-read"
                                          " doesn't measure yet. Verify their SF before bidding"
                                          " (missed soffits/returns have cost real money).")
                    # REWORK-CASCADE sentinel (93 Bennington loss: 'REMOVE AND REFABRICATE
                    # panels that INTERFACE with the shifted wall' — scope written in notes,
                    # never measured). Rework verbs = scope beyond the highlighted limits.
                    if any(k in _t for k in ("REFABRICATE", "REMOVE AND REINSTALL", "REWORK",
                                             "LIMITS OF WORK")):
                        auto_flags.append("⚠ REWORK job — notes extend scope beyond the marked"
                                          " limits (interfacing panels, returns, make-good)."
                                          " Read every note on this sheet before pricing.")
                except Exception:
                    pass
                auto_flags.append("Read from the drawing's pattern vectors — confirm which walls are in your scope"
                                  if auto_engine == "vector" else "AI suggestion — verify SF before bidding")
                # JOB-LEVEL SANITY (audit item 5 — flags only, SF never touched):
                # a merged/leaked region announces itself as a size outlier.
                try:
                    _sfs = sorted(p.get("area_sf", 0) for p in polys
                                  if not p.get("suggest_only") and p.get("area_sf", 0) > 0)
                    if len(_sfs) >= 4:
                        _med = _sfs[len(_sfs) // 2]
                        _big = _sfs[-1]
                        if _med > 0 and _big > 3 * _med and _big > 1500:
                            auto_flags.append(f"⚠ SANITY: largest piece ({_big:,.0f} SF) is "
                                              f"{_big/_med:.0f}× the page median — check for a merged or "
                                              f"leaked region before trusting its SF.")
                    _ptot = sum(_sfs)
                    if _ptot > 40000:
                        auto_flags.append(f"⚠ SANITY: page total {_ptot:,.0f} SF is unusually large for "
                                          f"one sheet — verify the scale and check for double-counted faces.")
                except Exception:
                    pass
                if not scale_conf:
                    # scale could NOT be read → SF used a default 8.0 and is unreliable. Force a calibrate.
                    auto_flags.append("Scale could not be read on this sheet — calibrate before trusting SF (SF may be off several ×)")
                else:
                    auto_flags.append(f"Scale read as 1\"={scale_val}' — verify")
                # PER-VIEW SCALES (owner rule: never measure at a scale the drawing didn't
                # state): pieces re-measured at their view's own printed scale say so; a
                # piece in an AMBIGUOUS multi-scale zone is flagged, never silently priced.
                try:
                    n_vs = sum(1 for p in polys if p.get("view_scale"))
                    if n_vs:
                        vss = sorted({p["view_scale"] for p in polys if p.get("view_scale")})
                        auto_flags.append(f"✓ {n_vs} region(s) measured at their view's own printed scale "
                                          + ", ".join(f"1\"={v}'" for v in vss))
                    if any(p.get("scale_risk") for p in polys):
                        auto_flags.append("⚠ Multiple view scales printed near some regions — verify those "
                                          "SF against the view's own scale before pricing")
                except Exception:
                    pass
                # ELEVATION MARKERS (blueprint 1a): "T.O. STEEL 139'-0\"" texts give exact heights.
                # When ≥2 markers agree they yield the sheet's TRUE vertical scale — the strongest
                # possible check. Conservative: only speaks when the fit has real consensus.
                try:
                    lv = callouts.read_levels_pg(pg)
                    if len(lv) < 2 and len((pg.get_text() or "")) < 250 and ocr_text.available():
                        lv = ocr_text.read_levels(pdf_bytes, pi)
                    if lv:
                        page_levels = [{"ft": l["ft"], "label": l["label"]} for l in lv[:8]]
                        fit = callouts.vertical_scale_from_levels(lv)
                        if fit:
                            m_ftpi = fit[0] * 72.0
                            used_ftpi = float(scale_val) if (scale_conf and scale_val) else 8.0
                            agree = min(m_ftpi, used_ftpi) / max(m_ftpi, used_ftpi)
                            if agree >= 0.93:
                                auto_flags.append(f"✓ Elevation markers confirm the scale ({m_ftpi:.1f} ft/in from {fit[2]} markers)")
                                jlog(job, f"Page {pi+1}: elevation markers CONFIRM scale ({m_ftpi:.1f} ft/in)", "ok")
                            else:
                                scale_conf = False   # trips the existing calibrate-before-trusting safety
                                auto_flags.append(f"⚠ Elevation markers imply {m_ftpi:.1f} ft/in but {used_ftpi:.1f} was used — calibrate before trusting SF")
                                jlog(job, f"Page {pi+1}: markers imply {m_ftpi:.1f} ft/in vs {used_ftpi:.1f} used — flagged", "warn")
                except Exception:
                    pass
                # DIMENSION-STRING SELF-CALIBRATION (blueprint 1b): the drawing's own "24'-0\""
                # strings paired with their measured dim lines = the sheet's true scale.
                # Cross-check only (same safety pattern as markers): confirm, or force calibrate.
                try:
                    ds = dim_scale.sheet_scale(pg)
                    if ds:
                        d_ftpi = ds[0] * 72.0
                        used_ftpi = float(scale_val) if (scale_conf and scale_val) else 8.0
                        agree = min(d_ftpi, used_ftpi) / max(d_ftpi, used_ftpi)
                        if agree >= 0.95:
                            auto_flags.append(f"✓ {ds[1]} dimension strings confirm the scale ({d_ftpi:.2f} ft/in)")
                            jlog(job, f"Page {pi+1}: {ds[1]} dimension strings CONFIRM scale ({d_ftpi:.2f} ft/in)", "ok")
                        else:
                            scale_conf = False   # trips the existing calibrate-before-trusting safety
                            auto_flags.append(f"⚠ {ds[1]} dimension strings imply {d_ftpi:.2f} ft/in but {used_ftpi:.1f} was used — calibrate before trusting SF")
                            jlog(job, f"Page {pi+1}: dims imply {d_ftpi:.2f} ft/in vs {used_ftpi:.1f} used — flagged", "warn")
                except Exception:
                    pass
            # aggregate the estimator's LINEAR (LF) measurements — trim/soffit/fascia (priced per LF)
            lin_by_mat = defaultdict(float)
            for it in lin_lf_items:
                lin_by_mat[it["material"]] += it["lf"]
            linear_items = [{"material": m, "lf": round(v, 1)} for m, v in sorted(lin_by_mat.items(), key=lambda x: -x[1])]
            n_lin_sf = len(lin_sf_polys)
            # AUTO-TRIM (blueprint 1c): derive corner/base-top/opening LF straight from the
            # detected faces. SUGGESTIONS only — kept in a separate field so they never touch
            # the estimator's confirmed linearItems or the money total. Needs a trusted scale.
            auto_trim = []
            if auto and scale_conf and scale_val:
                try:
                    auto_trim = auto_trim_mod.compute(polys, pw, ph, float(scale_val) / 72.0)
                    if auto_trim:
                        auto_flags.append("Suggested trim (auto) below — verify each line against your scope before pricing")
                except Exception as _te:
                    jlog(job, f"Page {pi+1}: auto-trim skipped ({_te})", "warn")
            job["takeoffData"].append({
                "pageNumber": pi + 1, "title": f"Sheet page {pi+1}", "sheetRef": f"p{pi+1}",
                "scale": (f"1\"={scale_val}'" if (auto and scale_conf) else "auto (calibrate)") if auto else "from markup",
                "scaleSource": ("default" if (auto and not scale_conf) else ("AI auto — verify" if auto else "estimator markup")),
                "verifiedScale": bool(auto and scale_conf),
                "building": "Building",
                "zones": zones,
                "linearItems": linear_items,
                "autoTrim": auto_trim,
                "flags": auto_flags + sf_warns,
                "levels": page_levels,
                "source": "texture-auto" if auto else "digitize",
                # window/door COUNT surface (count-only takeoffs are a whole bid class):
                # openings the readers detected and cut out of the SF on this page
                "openingsCount": (sum(len(p.get("holes") or []) for p in polys) if auto else None),
            })
            extra = (f" + {n_lin_sf} linear run(s)" if n_lin_sf else "") + (f", {len(linear_items)} trim/LF item(s)" if linear_items else "")
            jlog(job, f"Page {pi+1}: " + (f"{len(polys)} AI-suggested zone(s)" if auto else f"{len(polys)} marked region(s){extra}") + f", {len(zones)} material(s)", "warn" if auto else "ok")
            for w in sf_warns:
                jlog(job, f"Page {pi+1}: ⚠ {w}", "warn")
        doc.close()
        job["legend"] = list(legend.values())
        # THE ARCHITECT'S OWN MATERIAL SCHEDULE — read it off any text-bearing page so the
        # estimator can sanity-check our takeoff against the drawing's stated quantities.
        try:
            sched_best = {}
            sdoc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for pi in range(sdoc.page_count):
                if len(sdoc[pi].get_text() or "") < 60:   # skip textless pages fast
                    continue
                for row in callouts.read_schedule_pg(sdoc[pi]):
                    k = row["key"]
                    if k not in sched_best or row["sf"] > sched_best[k]["sf"]:
                        sched_best[k] = row
            sdoc.close()
            if len(sched_best) >= 2:
                sched = sorted(sched_best.values(), key=lambda r: -r["sf"])
                job["drawingSchedule"] = {"items": sched, "total": sum(r["sf"] for r in sched)}
                jlog(job, f"Read the drawing's own material schedule — {len(sched)} materials, "
                          f"{job['drawingSchedule']['total']:,} SF stated", "ok")
        except Exception:
            pass
        # OCR FALLBACK for FLATTENED sets — auto-detected pages with cladding but NO base text
        # (lettering exported as curves). Reads the material spec off the rendered image so the
        # estimator knows WHAT the cladding is even when the file carries no text. Lazy + capped.
        try:
            odoc = fitz.open(stream=pdf_bytes, filetype="pdf")
            flat_pages = [e["pageNumber"] for e in job["takeoffData"]
                          if e.get("source") == "texture-auto"
                          and len((odoc[e["pageNumber"] - 1].get_text() or "")) < 250][:4]
            odoc.close()
            if flat_pages and not ocr_text.available():
                jlog(job, f"OCR unavailable on server ({len(flat_pages)} flattened page(s) would benefit)", "warn")
            if flat_pages and ocr_text.available():
                seen = set(); ocr_mats = []
                for pn in flat_pages:
                    for m in ocr_text.read_materials(pdf_bytes, pn - 1):
                        k = m["text"].upper()
                        if k not in seen:
                            seen.add(k); ocr_mats.append(m)
                if ocr_mats:
                    job["ocrMaterials"] = ocr_mats[:10]
                    jlog(job, f"Read {len(ocr_mats)} material spec(s) off the drawing with OCR "
                              f"(flattened set): {', '.join(m['text'] for m in ocr_mats[:3])}", "ok")
        except Exception:
            pass
        # TYPICAL-OF-n FLAG (convention census 2026-07-23: 'TYP OF n' pages carry 53
        # not-found walls / 10.2k SF in 6 jobs — the drawing shows ONE instance, the
        # estimator counts n). FLAG ONLY per the two-tier rule: SF never auto-multiplied.
        try:
            import re as _re7
            _tdoc = fitz.open(stream=pdf_bytes, filetype="pdf")
            for _pi7 in range(len(_tdoc)):
                _t7 = (_tdoc[_pi7].get_text() or "").upper()
                for _m7 in _re7.finditer(r"TYP(?:ICAL)?\.?\s*(?:OF|X)\s*\(?(\d+)\)?", _t7):
                    _n7 = int(_m7.group(1))
                    if 2 <= _n7 <= 30:
                        for _e7 in job["takeoffData"]:
                            if _e7["pageNumber"] == _pi7 + 1:
                                _e7.setdefault("flags", []).append(
                                    f"⚠ REPETITION: this sheet says '{_m7.group(0).strip()}' — one instance"
                                    f" is drawn but {_n7} exist. Confirm the count is in your takeoff"
                                    " (multiply or trace the others) before bidding.")
                                break
                        break   # one flag per page
                if "MATCH LINE" in _t7:
                    for _e7 in job["takeoffData"]:
                        if _e7["pageNumber"] == _pi7 + 1:
                            _e7.setdefault("flags", []).append(
                                "⚠ MATCH LINE on this sheet — the view continues on another sheet."
                                " Confirm the continued portion is in the takeoff (censused: match-line"
                                " sheets hide walls).")
                            break
            _tdoc.close()
        except Exception:
            pass
        # JOB-LEVEL SOFFIT/RETURN CROSS-CHECK (the $500K recall guard — BOTH paths):
        # if ANY page of the set mentions soffits (or panel/canopy returns) but the
        # finished takeoff contains no zone named for them, flag it loudly. Flags
        # only — SF is never touched.
        try:
            _sr_pages = []
            _sdoc2 = fitz.open(stream=pdf_bytes, filetype="pdf")
            for _pi2 in range(len(_sdoc2)):
                _t2 = (_sdoc2[_pi2].get_text() or "").upper()
                if "SOFFIT" in _t2 or ("RETURN" in _t2 and ("PANEL" in _t2 or "CANOPY" in _t2 or "ACM" in _t2)):
                    _sr_pages.append(_pi2 + 1)
            _sdoc2.close()
            if _sr_pages and job["takeoffData"]:
                _named = any(("soffit" in str(z.get("materialName") or z.get("category") or "").lower())
                             or ("return" in str(z.get("materialName") or z.get("category") or "").lower())
                             for e in job["takeoffData"] for z in e["zones"])
                if not _named:
                    job["takeoffData"][-1].setdefault("flags", []).append(
                        "⚠ JOB CHECK: the drawings mention soffits/returns on page(s) "
                        + ", ".join(map(str, _sr_pages[:6])) + ("…" if len(_sr_pages) > 6 else "")
                        + " but NO soffit/return zone exists in this takeoff — confirm they are"
                          " measured or excluded before bidding (missed soffits have cost real money).")
        except Exception:
            pass
        total = sum(z["netArea"] for e in job["takeoffData"] for z in e["zones"])
        if auto_tried:  # never fail silently: say what the AI actually looked at
            jlog(job, f"Auto-detect scanned {auto_tried} page(s), found cladding on {auto_hits}", "warn" if auto_hits == 0 else "ok")
        if not job["takeoffData"]:
            jlog(job, "No measurements found — no readable markup, and the AI saw no cladding on the pages it scanned.", "warn")
        else:
            jlog(job, f"Done — {len(job['takeoffData'])} page(s), {round(total):,} SF total", "success")
        job["status"] = "done"; job["phase"] = "done"
        job["progress"] = {"label": "Complete", "pct": 100}
        _persist_job(jid)  # durable: this takeoff now survives a restart/redeploy
    except Exception as e:
        import traceback; traceback.print_exc()
        job["status"] = "error"; job["phase"] = "error"; job["error"] = str(e)
        jlog(job, f"Error: {e}", "error")
        _persist_job(jid)

# ── endpoints ──────────────────────────────────────────────────────────────
@app.post("/analyze")
async def analyze(background_tasks: BackgroundTasks, pdf: UploadFile = File(...)):
    jid = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"  # collision-proof (no same-ms overwrite)
    # STREAM the upload to disk in chunks — never hold a whole giant set in RAM.
    # (await pdf.read() on a 1GB file was the OOM vector for heavy documents.)
    d = _job_dir(jid)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "input.pdf")
    with open(path, "wb") as fh:
        while True:
            chunk = await pdf.read(8 * 1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    data = _load_pdf_bytes(path)
    jobs[jid] = {"status": "queued", "phase": "idle", "log": [], "progress": {"label": "Queued", "pct": 0},
                 "legend": [], "takeoffData": [], "scheduleData": None, "error": None,
                 "polygons_by_page": {}, "dims_by_page": {}, "pdf": data}
    _persist_job(jid); _evict_mem(); _gc_disk()  # durable + bounded RAM + bounded disk
    background_tasks.add_task(process, jid, data)
    return {"jobId": jid}

@app.get("/status/{jid}")
def status(jid: str):
    j = get_job(jid)
    if not j: raise HTTPException(404, "job not found")
    return {"status": j["status"], "phase": j.get("phase", ""), "log": j["log"], "progress": j["progress"],
            "legend": j.get("legend", []), "takeoffData": j.get("takeoffData", []),
            "scheduleData": j.get("scheduleData"), "drawingSchedule": j.get("drawingSchedule"),
            "ocrMaterials": j.get("ocrMaterials"), "error": j.get("error"), "pageCount": j.get("pageCount", 0)}

@app.get("/polygons/{jid}/{page}")
def polygons(jid: str, page: int):
    j = get_job(jid)
    if not j: raise HTTPException(404, "job not found")
    dims = j.get("dims_by_page", {}).get(page) or {"width": 612, "height": 792}
    return {"polygons": j.get("polygons_by_page", {}).get(page, []),
            "width": dims["width"], "height": dims["height"]}

@app.get("/page-image/{jid}/{page}")
def page_image(jid: str, page: int):
    j = get_job(jid)
    if not j or not j.get("pdf"): raise HTTPException(404, "job not found")
    doc = fitz.open(stream=j["pdf"], filetype="pdf")
    if page < 1 or page > doc.page_count:
        doc.close(); raise HTTPException(404, "page out of range")
    pix = doc[page - 1].get_pixmap(matrix=fitz.Matrix(2, 2))  # ~144 dpi
    png = pix.tobytes("png"); doc.close()
    return Response(content=png, media_type="image/png")

@app.get("/page-crop/{jid}/{page}")
def page_crop(jid: str, page: int, x0: float = 0, y0: float = 0, x1: float = 1, y1: float = 1, px: int = 1800):
    """Bluebeam-style deep zoom: render ONLY the viewport rect (normalized coords) at high DPI.
    The frontend swaps this in when the estimator zooms past ~2.5x so textures stay crisp."""
    j = get_job(jid)
    if not j or not j.get("pdf"): raise HTTPException(404, "job not found")
    doc = fitz.open(stream=j["pdf"], filetype="pdf")
    if page < 1 or page > doc.page_count:
        doc.close(); raise HTTPException(404, "page out of range")
    pg = doc[page - 1]
    x0 = max(0.0, min(1.0, x0)); y0 = max(0.0, min(1.0, y0))
    x1 = max(x0 + 0.01, min(1.0, x1)); y1 = max(y0 + 0.01, min(1.0, y1))
    # pg.rect is already the rotated/display rect and clip takes display coords (verified on a 270° page)
    W, H = pg.rect.width, pg.rect.height
    clip = fitz.Rect(x0 * W, y0 * H, x1 * W, y1 * H)
    z = min(8.0, max(1.0, min(2200, px) / max(1, clip.width)))
    pix = pg.get_pixmap(matrix=fitz.Matrix(z, z), clip=clip, alpha=False)
    png = pix.tobytes("png"); doc.close()
    return Response(content=png, media_type="image/png")

@app.post("/material-groups")
def material_groups_route(payload: dict = Body(...)):
    """A PREVIEW of material groups (within-job texture clustering, elevation-fenced). The estimator
    SELECTS the groups she's bidding — her selection carries the accuracy; the preview may be imperfect.
    SF returned is a PREVIEW estimate (approx), clearly flagged. Additive; digitize-markup untouched.
    payload: {jobId, page (1-indexed)}."""
    j = get_job(payload.get("jobId"))
    if not j or not j.get("pdf"):
        raise HTTPException(404, "job not found")
    page = int(payload.get("page", 1)) - 1
    # VECTOR-FIRST (estimator-confirmed: the geometry path beats texture clustering by a mile).
    # Each vector region is emitted as grid patches so the existing select + exact-on-select
    # flow works unchanged; the texture clustering stays as the fallback for scanned pages.
    try:
        vpolys, _, _, _ = vector_hatch.detect(j["pdf"], page)
        if vpolys:
            import numpy as np
            import cv2 as _cv2
            GX, GY = 96, 64
            groups = []
            for gi, p in enumerate(vpolys[:24]):
                pts = np.array([[x * GX, y * GY] for x, y in p["points"]], np.float32).astype(np.int32)
                m = np.zeros((GY, GX), np.uint8)
                _cv2.fillPoly(m, [pts.reshape(-1, 1, 2)], 1)
                ys, xs = np.where(m > 0)
                patches = [[round(x / GX, 4), round(y / GY, 4), round(1 / GX, 4), round(1 / GY, 4)]
                           for x, y in zip(xs.tolist(), ys.tolist())][:1500]
                if not patches:
                    continue
                groups.append({"group": f"vec{gi}", "color": p.get("fill_color") or [0, 0.7, 0.85],
                               "patches": patches, "approx_sf": p.get("area_sf", 0),
                               "material": p.get("material"), "source": "vector"})
            if groups:
                return {"status": "ok", "groups": groups, "engine": "vector"}
    except Exception:
        pass
    try:
        r = material_groups.groups(j["pdf"], page)
    except Exception as e:
        return {"groups": [], "error": str(e)}
    return r

@app.post("/refine-group")
def refine_group_route(payload: dict = Body(...)):
    """Exact-on-select: preview-group patches → corner-snapped exact shapes (+opening deductions).
    payload: {jobId, page (1-indexed), patches:[[nx,ny,nw,nh],...], min_opening_sf}. Additive."""
    j = get_job(payload.get("jobId"))
    if not j or not j.get("pdf"):
        raise HTTPException(404, "job not found")
    try:
        return snap_fill.refine_group(j["pdf"], int(payload.get("page", 1)) - 1, payload.get("patches") or [],
                                      min_opening_sf=float(payload.get("min_opening_sf") or 0))
    except Exception as e:
        return {"status": "error", "shapes": [], "error": str(e)}

@app.post("/snap-fill")
def snap_fill_route(payload: dict = Body(...)):
    """Coloring-book bucket / corner-snap → exact polygon + SF from the drawing's vector geometry.
    payload: {jobId, page (1-indexed), point:[nx,ny]}  OR  {jobId, page, corners:[[nx,ny],...]}.
    Bucket returns status 'ok' (points+area_sf) or 'leak' (caller switches to corner mode). Additive —
    does not touch the digitize-markup pipeline."""
    j = get_job(payload.get("jobId"))
    if not j or not j.get("pdf"):
        raise HTTPException(404, "job not found")
    page = int(payload.get("page", 1)) - 1
    dims = j.get("dims_by_page", {}).get(page + 1) or {"width": 612, "height": 792}
    try:
        if payload.get("corners"):
            r = snap_fill.corners(j["pdf"], page, payload["corners"],
                                  min_opening_sf=float(payload.get("min_opening_sf") or 0))
        else:
            r = snap_fill.bucket(j["pdf"], page, payload.get("point", [0.5, 0.5]))
    except Exception as e:
        return {"status": "error", "error": str(e)}
    r["width"] = dims["width"]; r["height"] = dims["height"]
    return r

@app.post("/split-shape")
def split_shape(payload: dict = Body(...)):
    """ESTIMATOR'S KNIFE: split a face along the structural line nearest the click — where
    HIS material boundary lives (the drawing often can't show it; he knows it). Each half
    keeps its share of the (already net) SF by area ratio; holes follow by containment.
    payload: {jobId, page (1-based), points:[[nx,ny]..], area_sf, holes, click:[nx,ny]}"""
    j = get_job(payload.get("jobId"))
    if not j or not j.get("pdf"):
        raise HTTPException(404, "job not found")
    try:
        from shapely.geometry import Polygon as SPoly, LineString, Point as SPoint
        from shapely.ops import split as sh_split
        page = int(payload.get("page", 1)) - 1
        doc = fitz.open(stream=j["pdf"], filetype="pdf")
        pg = doc[page]
        W, H = pg.rect.width, pg.rect.height
        geo = vector_hatch.page_geometry(pg)
        doc.close()
        pts = [(float(x) * W, float(y) * H) for x, y in payload.get("points", [])]
        poly = SPoly(pts).buffer(0)
        if poly.is_empty:
            return {"status": "error", "error": "bad polygon"}
        cx, cy = float(payload["click"][0]) * W, float(payload["click"][1]) * H
        minx, miny, maxx, maxy = poly.bounds
        # candidate structural lines near the click: long V (split left/right) or H (split bands)
        best = None
        for (x1, y1, x2, y2) in geo.get("segs", []):
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            L = (dx * dx + dy * dy) ** 0.5
            if L < 25:
                continue
            if dx < 0.05 * L:                      # vertical line: x ~ const
                x = (x1 + x2) / 2
                if minx + 4 < x < maxx - 4 and L >= 0.35 * (maxy - miny):
                    d = abs(x - cx)
                    if best is None or d < best[0]:
                        best = (d, "v", x)
            elif dy < 0.05 * L:                    # horizontal line: y ~ const
                y = (y1 + y2) / 2
                if miny + 4 < y < maxy - 4 and L >= 0.35 * (maxx - minx):
                    d = abs(y - cy)
                    if best is None or d < best[0]:
                        best = (d, "h", y)
        if best is None or best[0] > 0.06 * max(W, H):
            # no structural line near the click — split exactly AT the click (still useful)
            best = (0, "v", cx) if (maxy - miny) > (maxx - minx) else (0, "h", cy)
        _, axis, c = best
        cutter = LineString([(c, miny - 10), (c, maxy + 10)]) if axis == "v" else LineString([(minx - 10, c), (maxx + 10, c)])
        parts = [g for g in sh_split(poly, cutter).geoms if g.area > 1e-6]
        if len(parts) < 2:
            return {"status": "error", "error": "line does not split the face"}
        parts.sort(key=lambda g: -g.area)
        parts = parts[:2] if len(parts) == 2 else [parts[0], max(parts[1:], key=lambda g: g.area)]
        total_a = sum(g.area for g in parts)
        sf = float(payload.get("area_sf", 0))
        holes = payload.get("holes") or []
        out = []
        for g in parts:
            ext = list(g.exterior.coords)
            hp = []
            for hh in holes:
                try:
                    hc = SPoly([(float(x) * W, float(y) * H) for x, y in hh]).centroid
                    if g.contains(hc):
                        hp.append(hh)
                except Exception:
                    pass
            out.append({"points": [[round(x / W, 5), round(y / H, 5)] for x, y in ext[:-1]],
                        "area_sf": round(sf * g.area / max(total_a, 1e-9), 1),
                        "holes": hp})
        return {"status": "ok", "shapes": out, "axis": axis,
                "cut_at": round((c / W) if axis == "v" else (c / H), 5)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/learn")
def learn(payload: dict = Body(...)):
    """Capture a manual/corrected takeoff as labeled training data (the flywheel).
    payload: {jobId, page, shapes:[{points,name,color,type}], source}"""
    jid = payload.get("jobId")
    job = get_job(jid)
    try:
        # disk-fill guard: a single correction payload is KBs; anything huge is abuse
        if len(json.dumps(payload)) > 40_000_000:
            return {"ok": False, "error": "payload too large"}
        os.makedirs(CORR_DIR, exist_ok=True)
        ts = str(int(time.time() * 1000))
        d = os.path.join(CORR_DIR, ts)
        os.makedirs(d, exist_ok=True)
        if job and job.get("pdf") and payload.get("shapes"):   # pdf copy only for shape-labeled corrections
            with open(os.path.join(d, "drawing.pdf"), "wb") as fh:
                fh.write(job["pdf"])
        with open(os.path.join(d, "labels.json"), "w", encoding="utf-8") as fh:
            json.dump({**payload, "at": ts}, fh)               # keep the WHOLE payload (final-* answer keys carry takeoffData)
        n = len([x for x in os.listdir(CORR_DIR) if os.path.isdir(os.path.join(CORR_DIR, x))])
        return {"ok": True, "saved": ts, "total": n}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/autonomy-status")
def autonomy_status():
    """THE AUTONOMY METER: for every exported bid (answer key = final-* capture), compare the
    system's FIRST AUTO OUTPUT (job.json as processed, before any human edit) against the
    human-confirmed final. agreement = min/max of total SF. This is the dial that shows the
    learning working — when it lives near 100%, estimators stop reviewing and only price."""
    jobs_seen = {}
    try:
        for d in sorted(os.listdir(CORR_DIR)):
            lp = os.path.join(CORR_DIR, d, "labels.json")
            if not os.path.isfile(lp):
                continue
            try:
                with open(lp, encoding="utf-8") as fh:
                    rec = json.load(fh)
            except Exception:
                continue
            if not str(rec.get("source", "")).startswith("final"):
                continue
            jid = rec.get("jobId")
            if jid:
                jobs_seen[jid] = rec        # latest final per job wins
    except Exception:
        pass
    out = []
    for jid, rec in jobs_seen.items():
        j = get_job(jid)
        if not j:
            continue
        auto_sf = sum(z.get("netArea", 0) for e in (j.get("takeoffData") or []) for z in (e.get("zones") or []))
        fin_sf = sum(z.get("netArea", 0) for e in (rec.get("takeoffData") or []) for z in (e.get("zones") or []))
        if fin_sf <= 0:
            continue
        agree = round(100 * min(auto_sf, fin_sf) / max(auto_sf, fin_sf, 1))
        out.append({"jobId": jid, "projName": rec.get("projName") or j.get("projName") or "Project",
                    "auto_sf": round(auto_sf), "final_sf": round(fin_sf), "agreement": agree})
    out.sort(key=lambda r: r["jobId"], reverse=True)
    avg = round(sum(r["agreement"] for r in out) / len(out)) if out else None
    return {"jobs": out[:40], "avg_agreement": avg, "n": len(out)}

def process_compare(jid, data):
    """THE KILLER DEMO: his marked takeoff vs the AI's blank-sheet takeoff on the SAME
    drawing. Reads his Bluebeam polygons EXACTLY, strips them to a synthetic clean set,
    runs the auto engine on it, grades every wall with the bench's assembled scoring
    (pieces >=50%-inside the wall, summed SF + union coverage) — the frozen-exam
    machinery, live in production, on the estimator's own job."""
    j = get_job(jid)
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        his_by_pg = {}
        for pi in range(min(doc.page_count, 8)):
            pg = doc[pi]
            pw, ph = pg.rect.width, pg.rect.height
            his = [p for p in extract_page_polygons(pg, pw, ph, 8.0)
                   if p.get("sf_exact") and 40 <= p["area_sf"] <= 60000]
            if his:
                his_by_pg[pi] = his
        if not his_by_pg:
            j["status"] = "error"
            j["error"] = "No labeled takeoff polygons found — upload the estimator's marked Bluebeam set."
            _persist_job(jid); return
        for pi in range(doc.page_count):
            pg = doc[pi]
            for _ in range(400):
                a = pg.first_annot
                if not a:
                    break
                try:
                    pg.delete_annot(a)
                except Exception:
                    break
        clean = doc.tobytes()
        doc.close()
        j["pdf"] = clean                    # /page-image now serves the CLEAN drawing
        import vector_hatch
        from shapely.geometry import Polygon as _CP
        from shapely.ops import unary_union as _cu
        pages = []
        for pi, his in sorted(his_by_pg.items()):
            j["progress"] = {"label": f"AI takeoff on page {pi + 1}…", "pct": 20 + int(70 * len(pages) / max(1, len(his_by_pg)))}
            _persist_job(jid)
            try:
                pieces, W, H, sinfo = vector_hatch.detect(clean, pi)
            except Exception:
                pieces, sinfo = [], {}
            pp = []
            for p in pieces:
                try:
                    q = _CP([(x, y) for x, y in p["points"]]).buffer(0)
                    if not q.is_empty:
                        pp.append((q, p))
                except Exception:
                    pass
            walls = []
            for g in his:
                try:
                    gq = _CP([(x, y) for x, y in g["points"]]).buffer(0)
                except Exception:
                    continue
                mine = []
                for q, p in pp:
                    try:
                        if q.intersection(gq).area >= 0.5 * q.area:
                            mine.append((q, p))
                    except Exception:
                        pass
                asf = sum(p.get("area_sf", 0) for _, p in mine)
                cov = 0.0
                try:
                    if mine:
                        cov = _cu([q for q, _ in mine]).intersection(gq).area / max(1e-9, gq.area)
                except Exception:
                    pass
                walls.append({"sf": g["area_sf"], "mat": g.get("material") or "Wall",
                              "got": round(asf, 1), "cov": round(cov, 2),
                              "money": bool(cov >= 0.7 and abs(asf - g["area_sf"]) <= 0.15 * g["area_sf"])})
            pages.append({
                "page": pi + 1,
                "his": [{"points": g["points"], "sf": g["area_sf"],
                         "mat": (g.get("material") or "")[:40]} for g in his],
                "ours": [{"points": p["points"], "sf": p.get("area_sf", 0),
                          "mat": (p.get("material") or "")[:40],
                          "holes": (p.get("holes") or [])[:12]} for p in pieces],
                "hisTotal": round(sum(g["area_sf"] for g in his), 1),
                "ourTotal": round(sum(p.get("area_sf", 0) for p in pieces), 1),
                "walls": walls,
                "scale_confirmed": bool(sinfo.get("scale_confirmed")),
            })
        nw = sum(len(p["walls"]) for p in pages)
        nm = sum(1 for p in pages for w in p["walls"] if w["money"])
        nf = sum(1 for p in pages for w in p["walls"] if w["cov"] >= 0.3)
        j["compare"] = {"pages": pages,
                        "summary": {"walls": nw, "found": nf, "money": nm,
                                    "hisTotal": round(sum(p["hisTotal"] for p in pages), 1),
                                    "ourTotal": round(sum(p["ourTotal"] for p in pages), 1)}}
        j["status"] = "done"
        j["progress"] = {"label": "Comparison ready", "pct": 100}
        _persist_job(jid)
    except Exception as e:
        try:
            j["status"] = "error"; j["error"] = f"compare failed: {e}"
            _persist_job(jid)
        except Exception:
            pass

@app.post("/compare")
async def compare(background_tasks: BackgroundTasks, pdf: UploadFile = File(...)):
    jid = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"
    d = _job_dir(jid)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "input.pdf")
    with open(path, "wb") as fh:
        while True:
            chunk = await pdf.read(8 * 1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    data = _load_pdf_bytes(path)
    jobs[jid] = {"status": "processing", "phase": "compare", "log": [],
                 "progress": {"label": "Reading the estimator's takeoff…", "pct": 5},
                 "legend": [], "takeoffData": [], "scheduleData": None, "error": None,
                 "polygons_by_page": {}, "dims_by_page": {}, "pdf": data}
    _persist_job(jid); _evict_mem(); _gc_disk()
    background_tasks.add_task(process_compare, jid, data)
    return {"jobId": jid}

@app.get("/compare-result/{jid}")
def compare_result(jid: str):
    j = get_job(jid)
    if not j:
        raise HTTPException(404, "job not found")
    return {"status": j["status"], "progress": j.get("progress"),
            "error": j.get("error"), "compare": j.get("compare")}

@app.get("/learn-status")
def learn_status():
    try:
        n = len([x for x in os.listdir(CORR_DIR) if os.path.isdir(os.path.join(CORR_DIR, x))]) if os.path.isdir(CORR_DIR) else 0
    except Exception:
        n = 0
    return {"corrections": n}

@app.get("/evidence-pdf/{jid}")
def evidence_pdf(jid: str, materials: str = ""):
    """Marked-up evidence PDF: cover summary + each page with the regions OUTLINED in their color and
    labeled with SF. Optional ?materials=A,B filters to only what the estimator selected."""
    j = get_job(jid)
    if not j or not j.get("pdf"):
        raise HTTPException(404, "job not found")
    sel = set(m.strip().lower() for m in materials.split(",") if m.strip()) if materials else None
    def _match(name, cat):
        return (str(name or "").lower() in sel) or (str(cat or "").lower() in sel)
    # safety: if the selection matches nothing (e.g. renamed groups), mark everything — never a blank PDF
    if sel is not None and not any(_match(z.get("materialName"), z.get("category"))
                                   for el in j.get("takeoffData", []) for z in el.get("zones", [])):
        sel = None
    def keep(name, cat):
        return True if sel is None else _match(name, cat)
    src = fitz.open(stream=j["pdf"], filetype="pdf")
    out = fitz.open()
    # summary cover page (filtered to her selection when given)
    mats = {}; tot = 0.0
    for el in j.get("takeoffData", []):
        for z in el.get("zones", []):
            if not keep(z.get("materialName"), z.get("category")):
                continue
            k = z.get("materialName", "Material"); mats[k] = mats.get(k, 0) + z.get("netArea", 0); tot += z.get("netArea", 0)
    cov = out.new_page(width=612, height=792)
    cov.insert_text((50, 60), "Boston Facade Systems — Takeoff Evidence", fontsize=17, color=(0.05, 0.11, 0.18))
    cov.insert_text((50, 84), (j.get("projName") or "Project"), fontsize=11, color=(0.4, 0.45, 0.5))
    y = 130
    cov.insert_text((50, y), "Material", fontsize=10, color=(0.4, 0.45, 0.5))
    cov.insert_text((360, y), "Net SF", fontsize=10, color=(0.4, 0.45, 0.5)); y += 8
    cov.draw_line((50, y), (500, y), color=(0.8, 0.83, 0.87)); y += 20
    for k, sf in sorted(mats.items(), key=lambda x: -x[1]):
        cov.insert_text((50, y), str(k)[:48], fontsize=11)
        cov.insert_text((360, y), f"{sf:,.0f}", fontsize=11); y += 22
    y += 6; cov.draw_line((50, y), (500, y), color=(0.8, 0.83, 0.87)); y += 22
    cov.insert_text((50, y), "TOTAL", fontsize=12, color=(0.05, 0.11, 0.18))
    cov.insert_text((360, y), f"{tot:,.0f} SF", fontsize=12, color=(0.05, 0.11, 0.18))
    # BID-READINESS block: provenance + verify-first — the questions a GC exec asks.
    y += 34
    cov.insert_text((50, y), "How these numbers were measured", fontsize=11, color=(0.05, 0.11, 0.18)); y += 16
    cov.insert_text((50, y), "Every region is read from the drawing's own geometry (seams, fills, tags,", fontsize=8.5, color=(0.35, 0.4, 0.45)); y += 12
    cov.insert_text((50, y), "trained boundary model) and carries its arithmetic: gross - openings = net.", fontsize=8.5, color=(0.35, 0.4, 0.45)); y += 18
    nflag = 0
    flagged = []
    for el in j.get("takeoffData", []):
        for fl in (el.get("flags") or []):
            nflag += 1
            if len(flagged) < 6 and not str(fl).startswith("✓"):
                flagged.append((el.get("pageNumber"), str(fl)[:86]))
    unver = [el.get("pageNumber") for el in j.get("takeoffData", [])
             if el.get("scaleSource") == "default"]
    cov.insert_text((50, y), "Verify first:", fontsize=10, color=(0.7, 0.35, 0.05)); y += 14
    if unver:
        cov.insert_text((58, y), f"- Pages {', '.join(str(p) for p in unver[:8])}: scale unconfirmed - calibrate before trusting SF", fontsize=8.5, color=(0.55, 0.3, 0.05)); y += 12
    for pn9, fl in flagged:
        cov.insert_text((58, y), f"- p{pn9}: {fl}", fontsize=8.5, color=(0.55, 0.3, 0.05)); y += 12
    if not unver and not flagged:
        cov.insert_text((58, y), "- No open flags. All scales confirmed.", fontsize=8.5, color=(0.15, 0.5, 0.25)); y += 12
    # PER-WALL TABLE page(s): every zone with its page, SF, and reader provenance —
    # the wall-by-wall receipt a reviewer can check against the drawing in seconds.
    def _reader_of(z):
        m = str(z.get("materialName") or "")
        src9 = str(z.get("source") or "")
        if "bluebeam" in src9:
            return "Your markup (exact)"
        for pre, nm in (("Hatched area", "Hatch reader"), ("Color fill", "Drawn color"),
                        ("Rendered", "Rendered reader"), ("Wall area (AI boundary", "AI boundary model"),
                        ("Wall band", "Story-band"), ("Wall area", "Structural flood"),
                        ("Panel wall", "Drawn fill")):
            if m.startswith(pre):
                return nm
        return "Drawing geometry"
    rows = []
    for el in j.get("takeoffData", []):
        for z in el.get("zones", []):
            if keep(z.get("materialName"), z.get("category")):
                rows.append((el.get("pageNumber"), str(z.get("materialName") or "Material")[:38],
                             z.get("netArea", 0), _reader_of(z)))
    rows.sort(key=lambda r: -r[2])
    ty = 792
    tp = None
    for i9, (pn9, mat9, sf9, rd9) in enumerate(rows[:120]):
        if ty > 720:
            tp = out.new_page(width=612, height=792)
            tp.insert_text((50, 50), "Wall-by-wall detail", fontsize=13, color=(0.05, 0.11, 0.18))
            tp.insert_text((50, 72), "Page", fontsize=8.5, color=(0.4, 0.45, 0.5))
            tp.insert_text((90, 72), "Material / wall", fontsize=8.5, color=(0.4, 0.45, 0.5))
            tp.insert_text((360, 72), "Net SF", fontsize=8.5, color=(0.4, 0.45, 0.5))
            tp.insert_text((430, 72), "Measured by", fontsize=8.5, color=(0.4, 0.45, 0.5))
            tp.draw_line((50, 80), (562, 80), color=(0.8, 0.83, 0.87))
            ty = 96
        tp.insert_text((50, ty), f"p{pn9}", fontsize=9)
        tp.insert_text((90, ty), mat9, fontsize=9)
        tp.insert_text((360, ty), f"{sf9:,.0f}", fontsize=9)
        tp.insert_text((430, ty), rd9, fontsize=8.5, color=(0.35, 0.4, 0.45))
        ty += 15
    # each page: OUTLINE each kept region in its color + label the SF (a real marked-up sheet)
    for el in j.get("takeoffData", []):
        pn = el.get("pageNumber")
        if not pn or pn < 1 or pn > src.page_count:
            continue
        page_polys = [p for p in j.get("polygons_by_page", {}).get(pn, []) if keep(p.get("material"), p.get("category"))]
        if not page_polys:
            continue
        # RENDER the page to a size-capped JPEG background (not the huge vector page) → small, emailable.
        # The region outlines + SF labels are drawn as crisp VECTOR on top, so they stay sharp.
        srcpage = src[pn - 1]; pw, ph = srcpage.rect.width, srcpage.rect.height
        zoom = min(2.5, 2400.0 / max(pw, ph, 1))  # cap long side ~2400px regardless of sheet size
        pix = srcpage.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        pg = out.new_page(width=pw, height=ph)
        pg.insert_image(pg.rect, stream=pix.tobytes("jpg", jpg_quality=80))  # JPEG → colored sheets shrink hugely
        for p in page_polys:
            if p.get("suggest_only"):
                continue      # unaccepted AI suggestions never appear on the evidence PDF
            col = p.get("fill_color") or [0.85, 0.1, 0.1]
            col = tuple(float(c) for c in col[:3])
            pts = [(float(x) * pw, float(y) * ph) for x, y in (p.get("points") or [])]
            if len(pts) >= 3:
                pg.draw_polyline(pts + [pts[0]], color=col, width=1.3)
            cx, cy = p.get("cx", 0.5) * pw, p.get("cy", 0.5) * ph
            pg.insert_text((cx - 18, cy), f"{p.get('area_sf', 0):,.0f} SF", fontsize=8, color=(0.55, 0.0, 0.0))
    data = out.tobytes(); out.close(); src.close()
    return Response(content=data, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="BFS_Evidence_{jid}.pdf"'})

@app.post("/accept-suggestion")
def accept_suggestion(payload: dict = Body(...)):
    """Flip a suggest_only v13 piece into a REAL zone (the human said it's a wall).
    payload: {jobId, page (1-based), pieceId}. Rebuilds that page's zone rows,
    persists, logs the acceptance as flywheel training data."""
    jid = payload.get("jobId"); page = int(payload.get("page") or 0)
    pid = payload.get("pieceId")
    j = get_job(jid)
    if not j or page < 1:
        raise HTTPException(404, "job/page not found")
    polys = (j.get("polygons_by_page") or {}).get(page) or (j.get("polygons_by_page") or {}).get(str(page))
    if polys is None:
        raise HTTPException(404, "no polygons for page")
    hit = next((p for p in polys if str(p.get("id")) == str(pid) and p.get("suggest_only")), None)
    if hit is None:
        raise HTTPException(404, "suggestion not found (already accepted?)")
    hit["suggest_only"] = False
    hit["material"] = "AI wall (accepted)"
    hit["category"] = "AI wall (accepted)"
    hit["group"] = "AI wall (accepted)"
    # rebuild the page's zones from non-suggestion polys (mirror the analyze path)
    from collections import defaultdict as _dd
    bymat = _dd(lambda: {"sf": 0.0, "n": 0, "category": None})
    for p in polys:
        if p.get("suggest_only"):
            continue
        key = p.get("material") or p.get("category") or "Unlabeled"
        bymat[key]["sf"] += p.get("area_sf", 0); bymat[key]["n"] += 1
        bymat[key]["category"] = p.get("category")
    zones = [{"materialName": m, "material_type": m, "category": d["category"] or "Other",
              "netArea": round(d["sf"], 1), "grossArea": round(d["sf"], 1),
              "totalOpeningArea": 0, "description": f"{d['n']} region(s)"}
             for m, d in bymat.items()]
    for e in j.get("takeoffData") or []:
        if e.get("pageNumber") == page:
            e["zones"] = zones
    try:
        _persist_job(jid)
    except Exception:
        pass
    try:  # flywheel: an accepted suggestion is gold-grade boundary supervision
        ts = int(time.time() * 1000)
        os.makedirs(CORR_DIR, exist_ok=True)
        with open(os.path.join(CORR_DIR, f"{ts}_suggest-accept.json"), "w", encoding="utf-8") as fh:
            json.dump({"jobId": jid, "page": page, "source": "suggest-accept",
                       "shapes": [{"points": hit.get("points"), "area_sf": hit.get("area_sf")}],
                       "_ts": ts}, fh)
    except Exception:
        pass
    return {"ok": True, "zones": zones, "accepted_sf": hit.get("area_sf")}

@app.post("/bid-excel")
def bid_excel_endpoint(payload: dict = Body(...)):
    """Template-v2 BFS proposal workbook: clones the estimator's real letterhead
    template (bid_template_v2.xlsx) and fills it her way — per-page quantity
    addition formulas (=2049+321), Amount=Qty*Conv*Rate, TOTAL linked. Validated
    13/13 cell-identical against the submitted 26-262 Malden bid."""
    import bid_excel as _be
    try:
        data = _be.fill_bid(payload or {})
    except Exception as ex:
        raise HTTPException(500, f"bid excel failed: {ex}")
    name = ((payload.get("job_number") or "bid") + " - " + (payload.get("job_name") or "proposal"))
    safe = "".join(ch for ch in name if ch not in '\\/:*?"<>|')[:80]
    return Response(content=data,
                    media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f'attachment; filename="{safe}.xlsx"'})

@app.post("/scope-read")
async def scope_read(pdf: UploadFile = File(...)):
    """Extract text (+ estimator's checkmarks/notes) from a scope PDF so the Scope tab can read PDFs, not just Excel."""
    data = await pdf.read()
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception:
        raise HTTPException(400, "not a readable PDF")
    npages = doc.page_count
    parts = []
    for i in range(min(npages, 40)):
        t = doc[i].get_text() or ""
        anns = [a.info.get("content", "") for a in (doc[i].annots() or []) if a.info.get("content", "")]
        block = t
        if anns:
            block += "\n[estimator marks/notes on page: " + " | ".join(anns[:60]) + "]"
        if block.strip():
            parts.append(f"# Page {i+1}\n{block}")
    doc.close()
    txt = "\n\n".join(parts)
    return {"text": txt, "chars": len(txt), "pages": len(parts)}

@app.get("/snap-points/{jid}/{page}")
def snap_points(jid: str, page: int):
    """Bluebeam-style corner snap: the drawing's real CAD geometry → structural snap corners (endpoints +
    intersections of the LONG horizontal/vertical lines = building outline, floor lines, major openings).
    The estimator's clicks snap to these (pixel-perfect like Bluebeam); auto-markup can snap its edges too."""
    j = get_job(jid)
    if not j or not j.get("pdf"):
        raise HTTPException(404, "job not found")
    doc = fitz.open(stream=j["pdf"], filetype="pdf")
    if page < 1 or page > doc.page_count:
        doc.close(); raise HTTPException(404, "page out of range")
    pg = doc[page - 1]; W, H = pg.rect.width, pg.rect.height
    Lmin = 0.06 * max(W, H)  # "long" = structural, not window-mullion/brick/detail noise
    hlines = []; vlines = []
    try:
        for d in pg.get_drawings():
            for it in d["items"]:
                segs = []
                if it[0] == "l":
                    segs = [(it[1].x, it[1].y, it[2].x, it[2].y)]
                elif it[0] == "re":
                    r = it[1]; segs = [(r.x0, r.y0, r.x1, r.y0), (r.x1, r.y0, r.x1, r.y1),
                                       (r.x1, r.y1, r.x0, r.y1), (r.x0, r.y1, r.x0, r.y0)]
                for (x1, y1, x2, y2) in segs:
                    if (x2 - x1) ** 2 + (y2 - y1) ** 2 < Lmin ** 2:
                        continue
                    if abs(y2 - y1) < abs(x2 - x1) * 0.02:
                        hlines.append((min(x1, x2), max(x1, x2), (y1 + y2) / 2))
                    elif abs(x2 - x1) < abs(y2 - y1) * 0.02:
                        vlines.append((min(y1, y2), max(y1, y2), (x1 + x2) / 2))
    except Exception:
        pass
    doc.close()
    hlines = hlines[:600]; vlines = vlines[:600]  # bound compute
    pts = []
    for (xa, xb, y) in hlines: pts += [(xa, y), (xb, y)]
    for (ya, yb, x) in vlines: pts += [(x, ya), (x, yb)]
    for (xa, xb, yh) in hlines:  # H×V intersections = true corners
        for (ya, yb, xv) in vlines:
            if xa - 2 <= xv <= xb + 2 and ya - 2 <= yh <= yb + 2:
                pts.append((xv, yh))
    keep = []
    for pt in pts[:9000]:
        if not any(abs(pt[0] - k[0]) < 6 and abs(pt[1] - k[1]) < 6 for k in keep):
            keep.append(pt)
        if len(keep) >= 1500:
            break
    norm = [[round(x / W, 5), round(y / H, 5)] for (x, y) in keep]
    return {"points": norm, "width": W, "height": H, "count": len(norm)}

@app.post("/admin/upload-model")
async def upload_model(model: UploadFile = File(...), key: str = "", slot: str = ""):
    """One-time: load a trained ONNX model onto the volume. Guarded by ADMIN_KEY.
    slot="" -> the extent model (/data/model.onnx, v11 path); slot="v13" -> the
    boundary model (/data/model_v13.onnx) — its PRESENCE activates the v13 reader."""
    if key != os.environ.get("ADMIN_KEY", "bfs-model-load"):
        raise HTTPException(403, "bad key")
    data = await model.read()
    if slot == "v13":
        path = os.environ.get("V13_ONNX", "/data/model_v13.onnx")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(data)
        return {"ok": True, "bytes": len(data), "slot": "v13"}
    path = os.environ.get("MODEL_ONNX", "/data/model.onnx")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    model_infer.reset()
    return {"ok": True, "bytes": len(data), "model_available": model_infer.available()}

@app.post("/split-suggest")
def split_suggest(payload: dict = Body(...)):
    """✂ SUGGESTIONS: v13's boundary channel proposes cut lines inside a selected
    piece (probe-proven: boundary ink sits ON the estimator's wall edges at median
    1.00). The estimator accepts with one click — and every accepted cut becomes
    boundary supervision for v14. payload: {jobId, page(1-based), points([[x,y]..]
    normalized)} → {cuts: [{axis:'v'|'h', pos: normalized, span:[a,b]}]}"""
    jid = payload.get("jobId")
    page = int(payload.get("page") or 0)
    pts = payload.get("points") or []
    j = get_job(jid)
    if not j or not j.get("pdf") or page < 1 or len(pts) < 3:
        raise HTTPException(400, "bad request")
    import numpy as np, cv2
    mp = os.environ.get("V13_ONNX", "/data/model_v13.onnx")
    if not os.path.isfile(mp):
        return {"cuts": [], "note": "boundary model not present"}
    try:
        doc = fitz.open(stream=j["pdf"], filetype="pdf")
        pg = doc[page - 1]
        W, H = pg.rect.width, pg.rect.height
        z = 1536.0 / max(W, H)
        pix = pg.get_pixmap(matrix=fitz.Matrix(z, z), alpha=False)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].copy()
        doc.close()
        import vector_hatch as _vh
        if _vh._V13_SESS is None:
            import onnxruntime as ort
            _vh._V13_SESS = ort.InferenceSession(mp, providers=["CPUExecutionProvider"])
        ih, iw = img.shape[:2]
        T = 768
        prob = np.zeros((3, ih, iw), np.float32)
        cnt = np.zeros((ih, iw), np.float32)
        for y0 in sorted(set(list(range(0, max(1, ih - T + 1), T // 2)) + [max(0, ih - T)])):
            for x0 in sorted(set(list(range(0, max(1, iw - T + 1), T // 2)) + [max(0, iw - T)])):
                tile = img[y0:y0 + T, x0:x0 + T]
                th, tw = tile.shape[:2]
                pad = np.zeros((T, T, 3), np.uint8)
                pad[:th, :tw] = tile
                x = (pad.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
                lg = _vh._V13_SESS.run(None, {"input": x})[0][0]
                e = np.exp(lg - lg.max(0, keepdims=True))
                sm = e / e.sum(0, keepdims=True)
                prob[:, y0:y0 + th, x0:x0 + tw] += sm[:, :th, :tw]
                cnt[y0:y0 + th, x0:x0 + tw] += 1
        prob /= np.maximum(cnt, 1)[None]
        bnd = (prob.argmax(0) == 2).astype(np.uint8)
        m = np.zeros((ih, iw), np.uint8)
        cv2.fillPoly(m, [np.array([[int(px * W * z), int(py * H * z)] for px, py in pts], np.int32)], 1)
        inside = bnd & m
        col = inside.sum(0); hgt = m.sum(0)
        row = inside.sum(1); wid = m.sum(1)
        xs = [px * W * z for px, py in pts]; ys = [py * H * z for px, py in pts]
        x0b, x1b, y0b, y1b = int(min(xs)), int(max(xs)), int(min(ys)), int(max(ys))
        raw = []
        for cx in range(x0b + 4, min(x1b - 4, iw)):
            if hgt[cx] > 8 and col[cx] >= 0.6 * hgt[cx]:
                raw.append(("v", cx))
        for cy in range(y0b + 4, min(y1b - 4, ih)):
            if wid[cy] > 8 and row[cy] >= 0.6 * wid[cy]:
                raw.append(("h", cy))
        cuts = []
        for ax in ("v", "h"):
            ps = sorted(c for a, c in raw if a == ax)
            i = 0
            while i < len(ps):
                k = i
                while k + 1 < len(ps) and ps[k + 1] - ps[k] <= 3:
                    k += 1
                c = (ps[i] + ps[k]) / 2
                if ax == "v":
                    cuts.append({"axis": "v", "pos": round(c / z / W, 5),
                                 "span": [round(y0b / z / H, 5), round(y1b / z / H, 5)]})
                else:
                    cuts.append({"axis": "h", "pos": round(c / z / H, 5),
                                 "span": [round(x0b / z / W, 5), round(x1b / z / W, 5)]})
                i = k + 1
        return {"cuts": cuts[:8]}
    except Exception as e:
        return {"cuts": [], "note": f"suggest failed: {e}"}

@app.get("/admin/export-corrections")
def export_corrections(key: str = "", since: str = ""):
    """Flywheel export: every /learn label captured by the live app (renames, deletes,
    splits, bucket confirms, final-* answer keys) as one JSON payload — the raw
    material for the v14 dataset. Labels only (PDFs stay on the volume; fetch a
    specific correction's drawing separately if needed). Key-gated like upload-model."""
    if key != os.environ.get("ADMIN_KEY", "bfs-model-load"):
        raise HTTPException(403, "bad key")
    out = []
    try:
        for d in sorted(os.listdir(CORR_DIR)):
            if since and d <= since:
                continue
            lp = os.path.join(CORR_DIR, d, "labels.json")
            if not os.path.isfile(lp):
                continue
            try:
                with open(lp, encoding="utf-8") as fh:
                    rec = json.load(fh)
                rec["_ts"] = d
                rec["_has_pdf"] = os.path.isfile(os.path.join(CORR_DIR, d, "drawing.pdf"))
                out.append(rec)
            except Exception:
                continue
    except Exception:
        pass
    return {"n": len(out), "corrections": out}

@app.get("/health")
def health():
    try:
        on_disk = len([d for d in os.listdir(JOBS_DIR) if os.path.isdir(os.path.join(JOBS_DIR, d))]) if os.path.isdir(JOBS_DIR) else 0
    except Exception:
        on_disk = 0
    return {"status": "ok", "engine": "digitize-markup", "deps": "pymupdf+onnx",
            "auto_engine": "model" if model_infer.available() else "texture",
            "ocr": ocr_text.available(), "ocr_err": ocr_text.last_error(),
            "jobs_in_mem": len(jobs), "jobs_on_disk": on_disk, "mem_cap": MAX_MEM_JOBS}
