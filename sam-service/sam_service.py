"""
BPS Measurement Service — Vector-First (Bluebeam-level accuracy)
-----------------------------------------------------------------
Architectural PDFs from CAD software contain actual vector geometry —
the same data Bluebeam reads. We extract those paths directly and
calculate SF from real coordinates, not pixels.

Measurement priority:
  1. Vector paths from PDF (exact — same as Bluebeam)
  2. SAM pixel segmentation (fallback for scanned/raster PDFs)

POST /measure
  body: {
    image_b64:  string,          # base64 JPEG of rendered page (for SAM fallback)
    pdf_b64:    string,          # base64 of the raw PDF page bytes (for vector extraction)
    page_number: int,            # 1-based page number
    zones: [{ id, x0pct, y0pct, x1pct, y1pct }],
    scale_str:  "1/8\"=1'-0\"",
    dpi:        150
  }
  returns: {
    zones: [{ id, gross_sf, opening_sf, net_sf, method }],
    method: "vector" | "sam"
  }

GET /health
"""

import os, base64, json, math, io
from pathlib import Path
from typing import List, Optional
import numpy as np
import cv2
import torch
import fitz  # PyMuPDF
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

CHECKPOINT  = os.environ.get("SAM_CHECKPOINT", r"C:\Users\User\Downloads\sam_vit_b_01ec64.pth")
MEMORY_FILE = os.environ.get("SAM_MEMORY", r"C:\Users\User\Downloads\sam_memory.json")
DPI_DEFAULT = 150

# ── Load SAM (used as fallback for raster PDFs) ───────────────────────────────
print("Loading SAM...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
sam.to(device)
predictor = SamPredictor(sam)
auto_gen = SamAutomaticMaskGenerator(
    sam, points_per_side=32, pred_iou_thresh=0.88,
    stability_score_thresh=0.90, min_mask_region_area=500,
)
print("SAM ready.")

app = FastAPI(title="BPS Measurement Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Models ────────────────────────────────────────────────────────────────────
class ZoneBox(BaseModel):
    id: int
    x0pct: float
    y0pct: float
    x1pct: float
    y1pct: float

class PolygonRequest(BaseModel):
    pdf_b64: str
    page_number: int = 1
    scale_str: Optional[str] = "1/8\"=1'-0\""

class MeasureRequest(BaseModel):
    image_b64: str
    pdf_b64: Optional[str] = None   # raw PDF bytes base64 — enables vector extraction
    page_number: Optional[int] = 1
    zones: List[ZoneBox]
    scale_str: Optional[str] = "1/8\"=1'-0\""
    dpi: Optional[int] = DPI_DEFAULT

class ZoneResult(BaseModel):
    id: int
    gross_sf: float
    opening_sf: float
    net_sf: float
    method: str  # "vector" or "sam"

class MeasureResponse(BaseModel):
    zones: List[ZoneResult]
    method: str


# ── Scale helpers ─────────────────────────────────────────────────────────────
def scale_to_ft_per_inch(s: str) -> float:
    """'1/8\"=1\'-0\"' → 8.0 (feet per inch on paper)"""
    import re
    m = re.search(r'(\d+)/(\d+)', s)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        if num > 0:
            return den / num
    m = re.search(r'([\d.]+)', s)
    if m:
        v = float(m.group(1))
        if 0 < v < 1:
            return 1.0 / v
    return 8.0

def pdf_pts_to_sf(area_pts: float, ft_per_inch: float) -> float:
    """
    Convert area in PDF points² to square feet.
    1 PDF point = 1/72 inch.
    At scale ft_per_inch: 1 point on paper = ft_per_inch/72 real feet.
    1 pt² = (ft_per_inch/72)² sq ft.
    """
    ft_per_pt = ft_per_inch / 72.0
    return area_pts * (ft_per_pt ** 2)

def px_to_sf(px: int, ft_per_inch: float, dpi: int) -> float:
    ppf = dpi / ft_per_inch
    return px / (ppf * ppf)

def pct_to_pts(x0pct, y0pct, x1pct, y1pct, page_width, page_height):
    """Convert % bounds to PDF point bounds."""
    return (
        page_width  * x0pct / 100,
        page_height * y0pct / 100,
        page_width  * x1pct / 100,
        page_height * y1pct / 100,
    )

def pct_to_px(x0pct, y0pct, x1pct, y1pct, w, h):
    return (
        max(0, int(w * x0pct / 100)),
        max(0, int(h * y0pct / 100)),
        min(w, int(w * x1pct / 100)),
        min(h, int(h * y1pct / 100)),
    )


# ── Polygon area (shoelace formula) ───────────────────────────────────────────
def polygon_area(points) -> float:
    """Exact area of a polygon given its vertices. Units = input units²."""
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2.0

def rect_area(rect) -> float:
    """Area of a fitz.Rect."""
    return abs(rect.width * rect.height)


# ── Vector extraction from PDF ────────────────────────────────────────────────
def extract_vector_zones(pdf_bytes: bytes, page_number: int, zone_boxes: list, ft_per_inch: float) -> dict:
    """
    Extract closed vector paths from the PDF page.
    For each zone box, find the largest closed polygon whose centroid
    falls within the box — that's the wall surface.
    Also find smaller enclosed paths inside it — those are openings.

    Returns: { zone_id: { gross_sf, opening_sf, net_sf } }
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_number - 1]
    pw, ph = page.rect.width, page.rect.height  # page dimensions in PDF points

    # Extract all drawing paths
    drawings = page.get_drawings()
    doc.close()

    # Build list of closed polygons with their areas and centroids
    polygons = []
    for d in drawings:
        # Skip very thin lines (dimension lines, annotation lines)
        if d.get("width", 0) > 0 and d.get("fill") is None and d.get("width", 0) < 2:
            continue

        # Use the rect as a proxy for area when path is complex
        r = d.get("rect")
        if r is None:
            continue

        r = fitz.Rect(r)
        if r.is_empty or r.width < 10 or r.height < 10:
            continue

        area = rect_area(r)
        if area < 500:  # too small — dimension annotation etc.
            continue

        cx = (r.x0 + r.x1) / 2
        cy = (r.y0 + r.y1) / 2

        # Try to get precise polygon area from path items
        precise_area = area  # default to rect area
        pts = []
        for item in d.get("items", []):
            if item[0] == "l":   # line segment
                pts.append(item[1])
            elif item[0] == "re": # rectangle
                rr = fitz.Rect(item[1])
                precise_area = rect_area(rr)
                pts = [rr.tl, rr.tr, rr.br, rr.bl]
                cx = (rr.x0 + rr.x1) / 2
                cy = (rr.y0 + rr.y1) / 2
                break

        if len(pts) >= 3:
            pa = polygon_area([(p.x, p.y) for p in pts])
            if pa > 100:
                precise_area = pa
                cx = sum(p.x for p in pts) / len(pts)
                cy = sum(p.y for p in pts) / len(pts)

        polygons.append({
            "area": precise_area,
            "cx": cx, "cy": cy,
            "rect": r,
        })

    if not polygons:
        return {}

    # Sort largest first
    polygons.sort(key=lambda p: p["area"], reverse=True)

    results = {}
    for zbox in zone_boxes:
        bx0, by0, bx1, by1 = pct_to_pts(zbox.x0pct, zbox.y0pct, zbox.x1pct, zbox.y1pct, pw, ph)

        # Find polygons whose centroid is inside the zone box
        in_zone = [p for p in polygons
                   if bx0 <= p["cx"] <= bx1 and by0 <= p["cy"] <= by1]

        if not in_zone:
            continue

        # Largest = the wall surface
        wall = in_zone[0]
        gross_sf = pdf_pts_to_sf(wall["area"], ft_per_inch)

        # Smaller polygons inside the wall rect = openings (windows, doors)
        wr = wall["rect"]
        openings = [
            p for p in in_zone[1:]
            if p["area"] < wall["area"] * 0.5
            and wr.contains(fitz.Point(p["cx"], p["cy"]))
        ]
        opening_sf = sum(pdf_pts_to_sf(o["area"], ft_per_inch) for o in openings)
        opening_sf = min(opening_sf, gross_sf * 0.85)  # sanity cap

        net_sf = max(gross_sf - opening_sf, 0)

        results[zbox.id] = {
            "gross_sf":   round(gross_sf, 1),
            "opening_sf": round(opening_sf, 1),
            "net_sf":     round(net_sf, 1),
        }

    return results


# ── SAM fallback (raster PDFs / scanned drawings) ────────────────────────────
def decode_image(b64: str) -> np.ndarray:
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def sam_measure_zones(img_rgb, zone_boxes, ft_per_inch, dpi):
    """Box-prompted SAM segmentation for each zone."""
    h, w = img_rgb.shape[:2]
    predictor.set_image(img_rgb)
    results = {}
    for z in zone_boxes:
        x0, y0, x1, y1 = pct_to_px(z.x0pct, z.y0pct, z.x1pct, z.y1pct, w, h)
        if x1 <= x0 or y1 <= y0:
            continue
        box = np.array([x0, y0, x1, y1])
        masks, scores, _ = predictor.predict(box=box, multimask_output=True)
        if masks is None or len(masks) == 0:
            continue
        wall_mask = masks[int(np.argmax(scores))]
        gross_px = int(wall_mask.sum())

        # Find openings by running auto-gen on the crop
        ys, xs = np.where(wall_mask)
        if len(xs) == 0: continue
        cy0, cy1 = max(0, ys.min()-10), min(h, ys.max()+10)
        cx0, cx1 = max(0, xs.min()-10), min(w, xs.max()+10)
        crop = img_rgb[cy0:cy1, cx0:cx1]

        opening_px = 0
        if crop.shape[0] > 80 and crop.shape[1] > 80 and gross_px > 30000:
            zone_crop = wall_mask[cy0:cy1, cx0:cx1]
            raw = auto_gen.generate(crop)
            for m in raw:
                seg = m["segmentation"]
                area = m["area"]
                if area < 3000 or area > gross_px * 0.40: continue
                overlap = np.logical_and(seg, zone_crop).sum()
                if overlap / max(area, 1) > 0.75 and area / max(gross_px, 1) < 0.50:
                    opening_px += area

        opening_px = min(opening_px, int(gross_px * 0.85))
        net_px = max(gross_px - opening_px, 0)
        results[z.id] = {
            "gross_sf":   round(px_to_sf(gross_px,   ft_per_inch, dpi), 1),
            "opening_sf": round(px_to_sf(opening_px, ft_per_inch, dpi), 1),
            "net_sf":     round(px_to_sf(net_px,     ft_per_inch, dpi), 1),
        }
    return results

def is_vector_pdf(pdf_bytes: bytes, page_number: int) -> bool:
    """Check if this PDF page has meaningful vector geometry."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_number - 1]
    drawings = page.get_drawings()
    doc.close()
    # If there are large closed paths, it's a vector PDF
    large = [d for d in drawings
             if d.get("rect") and fitz.Rect(d["rect"]).get_area() > 1000]
    return len(large) > 5


# ── Endpoint ──────────────────────────────────────────────────────────────────
@app.post("/measure", response_model=MeasureResponse)
def measure(req: MeasureRequest):
    ft_per_inch = scale_to_ft_per_inch(req.scale_str or "")
    dpi = req.dpi or DPI_DEFAULT
    print(f"/measure: {len(req.zones)} zones  scale={req.scale_str} ({ft_per_inch} ft/in)")

    method = "sam"
    raw_results = {}

    # ── Try vector extraction first ──
    if req.pdf_b64:
        try:
            pdf_bytes = base64.b64decode(req.pdf_b64)
            if is_vector_pdf(pdf_bytes, req.page_number or 1):
                print("  Vector PDF detected — using exact coordinate measurement")
                raw_results = extract_vector_zones(pdf_bytes, req.page_number or 1, req.zones, ft_per_inch)
                if raw_results:
                    method = "vector"
                    print(f"  Vector measurement complete: {len(raw_results)} zones")
                else:
                    print("  Vector extraction found no matching polygons — falling back to SAM")
            else:
                print("  Raster PDF — using SAM pixel measurement")
        except Exception as e:
            print(f"  Vector extraction failed ({e}) — falling back to SAM")

    # ── SAM fallback ──
    if method == "sam":
        img_rgb = decode_image(req.image_b64)
        raw_results = sam_measure_zones(img_rgb, req.zones, ft_per_inch, dpi)

    # Build response
    out = []
    for z in req.zones:
        r = raw_results.get(z.id, {"gross_sf": 0, "opening_sf": 0, "net_sf": 0})
        print(f"  Zone {z.id} [{method}]: gross={r['gross_sf']} opening={r['opening_sf']} net={r['net_sf']} SF")
        out.append(ZoneResult(id=z.id, method=method, **r))

    return MeasureResponse(zones=out, method=method)


def _normalize_verts(verts, pw, ph):
    """Convert PyMuPDF vertices (Points or tuples) to normalized [0-1] coord pairs."""
    out = []
    for v in verts:
        if isinstance(v, (tuple, list)):
            out.append([round(v[0] / pw, 4), round(v[1] / ph, 4)])
        else:
            out.append([round(v.x / pw, 4), round(v.y / ph, 4)])
    return out


def extract_bluebeam_polygons(pdf_bytes: bytes, page_number: int, ft_per_inch: float = 8.0) -> list:
    """
    Read Bluebeam (and PDF-native) Polygon annotations from a page.
    These store the estimator's exact traced shapes AND the measured SF value
    in the annotation content label (e.g. "871 sf").
    Returns list of polygon dicts ready for the interactive view.
    """
    import re
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_number - 1]
    pw, ph = page.rect.width, page.rect.height

    results = []
    for a in page.annots():
        if a.type[0] != 6:   # only Polygon annotations
            continue

        verts = a.vertices or []
        if len(verts) < 3:
            continue

        content  = a.info.get("content", "")
        fill     = a.colors.get("fill", [])
        stroke   = a.colors.get("stroke", [])
        rect     = a.rect

        # Parse SF from label ("493 sf", "2,209 sf", etc.) — use Bluebeam's own measurement
        sf_match = re.search(r"([\d,]+)\s*sf", content, re.IGNORECASE)
        if sf_match:
            area_sf = float(sf_match.group(1).replace(",", ""))
        else:
            # No SF label — calculate from polygon vertices
            raw_pts = []
            for v in verts:
                if isinstance(v, (tuple, list)):
                    raw_pts.append((v[0], v[1]))
                else:
                    raw_pts.append((v.x, v.y))
            area_sf = round(pdf_pts_to_sf(polygon_area(raw_pts), ft_per_inch), 1)

        norm = _normalize_verts(verts, pw, ph)
        cx   = round(sum(v[0] for v in norm) / len(norm), 4)
        cy   = round(sum(v[1] for v in norm) / len(norm), 4)

        results.append({
            "id":         len(results),
            "points":     norm,
            "area_sf":    area_sf,
            "cx":         cx,
            "cy":         cy,
            "bbox":       [round(rect.x0/pw,4), round(rect.y0/ph,4),
                           round(rect.x1/pw,4), round(rect.y1/ph,4)],
            "source":     "bluebeam",
            "label":      content,
            "fill_color": [round(c, 3) for c in fill] if fill else None,
        })

    doc.close()
    return results


@app.post("/polygons")
def get_page_polygons(req: PolygonRequest):
    """
    Extract surface polygons from a PDF page.

    Priority:
      1. Bluebeam/PDF polygon annotations  — exact shapes + SF from estimator's markup
      2. Vector paths from CAD PDF          — exact geometry, SF from scale
    """
    pdf_bytes = base64.b64decode(req.pdf_b64)
    ft_per_inch = scale_to_ft_per_inch(req.scale_str or "")

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[req.page_number - 1]
    pw, ph = page.rect.width, page.rect.height
    doc.close()

    # ── Priority 1: Bluebeam polygon annotations ────────────────────────────
    bb = extract_bluebeam_polygons(pdf_bytes, req.page_number, ft_per_inch)
    if bb:
        print(f"/polygons [bluebeam]: page {req.page_number} → {len(bb)} annotation polygons")
        return {"polygons": bb, "page_width": pw, "page_height": ph,
                "count": len(bb), "method": "bluebeam"}

    # ── Priority 2: Vector path extraction ──────────────────────────────────
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[req.page_number - 1]
    drawings = page.get_drawings()
    doc.close()

    MIN_AREA_PTS = 300
    raw = []
    for d in drawings:
        r = d.get("rect")
        if r is None:
            continue
        r = fitz.Rect(r)
        if r.is_empty or r.width < 12 or r.height < 12:
            continue

        pts = []
        for item in d.get("items", []):
            if item[0] == "l":
                pts.append([float(item[1].x), float(item[1].y)])
            elif item[0] == "c":
                pts.append([float(item[3].x), float(item[3].y)])
            elif item[0] == "re":
                rr = fitz.Rect(item[1])
                pts = [[rr.x0, rr.y0], [rr.x1, rr.y0], [rr.x1, rr.y1], [rr.x0, rr.y1]]
                break

        if len(pts) < 3:
            pts = [[r.x0, r.y0], [r.x1, r.y0], [r.x1, r.y1], [r.x0, r.y1]]

        poly_area = polygon_area(pts)
        if poly_area < MIN_AREA_PTS:
            continue

        area_sf = round(pdf_pts_to_sf(poly_area, ft_per_inch), 1)
        norm = [[round(x / pw, 4), round(y / ph, 4)] for x, y in pts]
        cx   = round(sum(p[0] for p in norm) / len(norm), 4)
        cy   = round(sum(p[1] for p in norm) / len(norm), 4)

        raw.append({
            "id": len(raw), "points": norm, "area_sf": area_sf,
            "cx": cx, "cy": cy,
            "bbox": [round(r.x0/pw,4), round(r.y0/ph,4), round(r.x1/pw,4), round(r.y1/ph,4)],
            "source": "vector",
        })

    raw.sort(key=lambda p: p["area_sf"], reverse=True)
    raw = raw[:60]
    for i, p in enumerate(raw):
        p["id"] = i

    print(f"/polygons [vector]: page {req.page_number} → {len(raw)} polygons")
    return {"polygons": raw, "page_width": pw, "page_height": ph,
            "count": len(raw), "method": "vector"}


@app.get("/health")
def health():
    return {"status": "ok", "device": device, "model": "vector-first + SAM fallback"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8001)))
