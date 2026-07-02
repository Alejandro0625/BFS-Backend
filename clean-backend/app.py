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
import snap_fill  # coloring-book BUCKET fill + corner-snap → exact SF from vector geometry (assist layer)
import material_groups  # within-job texture grouping → a selectable PREVIEW of material groups (assist layer)
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
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
jobs = {}  # jobId -> dict  (in-memory hot cache, bounded to MAX_MEM_JOBS)

# ── durable, bounded job store ───────────────────────────────────────────────
# The old code kept every uploaded PDF in `jobs` forever → unbounded RAM → OOM.
# Now: keep a small hot set in RAM, persist each job to the volume, rehydrate on demand.
def _job_dir(jid): return os.path.join(JOBS_DIR, str(jid))

def _persist_job(jid):
    """Snapshot a job to the volume so it survives eviction + restarts. Best-effort (no-op if no volume)."""
    job = jobs.get(jid)
    if not job: return
    try:
        d = _job_dir(jid); os.makedirs(d, exist_ok=True)
        if job.get("pdf") and not os.path.exists(os.path.join(d, "input.pdf")):
            with open(os.path.join(d, "input.pdf"), "wb") as fh: fh.write(job["pdf"])
        meta = {k: job.get(k) for k in ("status", "phase", "progress", "legend", "takeoffData",
                "scheduleData", "error", "polygons_by_page", "dims_by_page", "log", "projName")}
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
        meta["pdf"] = open(p, "rb").read() if os.path.exists(p) else None
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

def _gc_disk():
    """Keep the volume bounded — prune the oldest persisted job folders."""
    try:
        if not os.path.isdir(JOBS_DIR): return
        dirs = [os.path.join(JOBS_DIR, d) for d in os.listdir(JOBS_DIR)]
        dirs = [d for d in dirs if os.path.isdir(d)]
        if len(dirs) <= MAX_DISK_JOBS: return
        dirs.sort(key=lambda d: os.path.getmtime(d))
        for d in dirs[:-MAX_DISK_JOBS]:
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
            if not polys and not doc_has_markup and page_is_elev and auto_used < MAX_AUTO_PAGES:
                # RAW/clean page -> auto-markup. Engine order: VECTOR (reads the drawn pattern —
                # exact geometry, openings netted) -> trained MODEL -> texture heuristics.
                try:
                    auto_tried += 1
                    tpolys = []; sinfo = {}
                    try:
                        tpolys, _, _, sinfo = vector_hatch.detect(pdf_bytes, pi)
                        auto_engine = "vector"
                    except Exception:
                        tpolys = []
                    if not tpolys:
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
                        polys = tpolys; auto = True; auto_used += 1; auto_hits += 1
                        scale_conf = bool(sinfo.get("scale_confirmed")); scale_val = sinfo.get("ft_per_in")
                except Exception as te:
                    jlog(job, f"Page {pi+1}: auto-detect skipped ({te})", "warn")
            sf_warns = flag_label_outliers(polys, pw, ph) if not auto else []  # catch typo'd markup labels
            job["polygons_by_page"][pi + 1] = polys
            job["dims_by_page"][pi + 1] = {"width": pw, "height": ph}
            if not polys and not lin_lf_items:  # keep pages that have linear (trim/LF) measurements even with no area polygons
                continue
            bymat = defaultdict(lambda: {"sf": 0.0, "n": 0, "category": None})
            for p in polys:
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
            if auto:
                auto_flags.append("Read from the drawing's pattern vectors — confirm which walls are in your scope"
                                  if auto_engine == "vector" else "AI suggestion — verify SF before bidding")
                if not scale_conf:
                    # scale could NOT be read → SF used a default 8.0 and is unreliable. Force a calibrate.
                    auto_flags.append("Scale could not be read on this sheet — calibrate before trusting SF (SF may be off several ×)")
                else:
                    auto_flags.append(f"Scale read as 1\"={scale_val}' — verify")
            # aggregate the estimator's LINEAR (LF) measurements — trim/soffit/fascia (priced per LF)
            lin_by_mat = defaultdict(float)
            for it in lin_lf_items:
                lin_by_mat[it["material"]] += it["lf"]
            linear_items = [{"material": m, "lf": round(v, 1)} for m, v in sorted(lin_by_mat.items(), key=lambda x: -x[1])]
            n_lin_sf = len(lin_sf_polys)
            job["takeoffData"].append({
                "pageNumber": pi + 1, "title": f"Sheet page {pi+1}", "sheetRef": f"p{pi+1}",
                "scale": (f"1\"={scale_val}'" if (auto and scale_conf) else "auto (calibrate)") if auto else "from markup",
                "scaleSource": ("default" if (auto and not scale_conf) else ("AI auto — verify" if auto else "estimator markup")),
                "verifiedScale": bool(auto and scale_conf),
                "building": "Building",
                "zones": zones,
                "linearItems": linear_items,
                "flags": auto_flags + sf_warns,
                "source": "texture-auto" if auto else "digitize",
            })
            extra = (f" + {n_lin_sf} linear run(s)" if n_lin_sf else "") + (f", {len(linear_items)} trim/LF item(s)" if linear_items else "")
            jlog(job, f"Page {pi+1}: " + (f"{len(polys)} AI-suggested zone(s)" if auto else f"{len(polys)} marked region(s){extra}") + f", {len(zones)} material(s)", "warn" if auto else "ok")
            for w in sf_warns:
                jlog(job, f"Page {pi+1}: ⚠ {w}", "warn")
        doc.close()
        job["legend"] = list(legend.values())
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
    data = await pdf.read()
    jid = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}"  # collision-proof (no same-ms overwrite)
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
            "scheduleData": j.get("scheduleData"), "error": j.get("error")}

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

@app.post("/learn")
def learn(payload: dict = Body(...)):
    """Capture a manual/corrected takeoff as labeled training data (the flywheel).
    payload: {jobId, page, shapes:[{points,name,color,type}], source}"""
    jid = payload.get("jobId")
    job = get_job(jid)
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
async def upload_model(model: UploadFile = File(...), key: str = ""):
    """One-time: load the trained ONNX model onto the volume so RAW drawings get model auto-markup.
    Guarded by ADMIN_KEY. Persists to /data (survives redeploys)."""
    if key != os.environ.get("ADMIN_KEY", "bfs-model-load"):
        raise HTTPException(403, "bad key")
    data = await model.read()
    path = os.environ.get("MODEL_ONNX", "/data/model.onnx")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    model_infer.reset()
    return {"ok": True, "bytes": len(data), "model_available": model_infer.available()}

@app.get("/health")
def health():
    try:
        on_disk = len([d for d in os.listdir(JOBS_DIR) if os.path.isdir(os.path.join(JOBS_DIR, d))]) if os.path.isdir(JOBS_DIR) else 0
    except Exception:
        on_disk = 0
    return {"status": "ok", "engine": "digitize-markup", "deps": "pymupdf+onnx",
            "auto_engine": "model" if model_infer.available() else "texture",
            "jobs_in_mem": len(jobs), "jobs_on_disk": on_disk, "mem_cap": MAX_MEM_JOBS}
