"""
BFS Measurement Service — Multi-Modal Detection Pipeline
---------------------------------------------------------
Detection priority per page:
  1. Bluebeam/PDF polygon annotations  (exact shapes + SF from estimator markup)
  2. CAD vector fill clustering         (exact geometry, SF from scale)
  3. Claude Vision (claude-opus-4-8)    (reads hatching patterns on any PDF)
  4. SAM fallback                       (raster/scanned PDFs)

Scale detection:
  - EasyOCR reads the title block for "1/8" = 1'-0"" automatically
  - Falls back to server default or 1/8"=1'-0"

Area calculation:
  - Shapely for precise polygon math, proper hole/opening subtraction
"""

import os, base64, json, re, io, threading
from typing import List, Optional
import numpy as np
import cv2
import torch
import fitz  # PyMuPDF
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from shapely.geometry import Polygon as SPoly
from shapely.ops import unary_union
from shapely.validation import make_valid
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

CHECKPOINT  = os.environ.get("SAM_CHECKPOINT", "/app/sam_vit_b_01ec64.pth")
DPI_DEFAULT = 150
CLAUDE_MODEL = "claude-opus-4-8"   # always use the latest

# ── Load SAM ──────────────────────────────────────────────────────────────────
print("Loading SAM...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
sam.to(device)
predictor   = SamPredictor(sam)
auto_gen    = SamAutomaticMaskGenerator(
    sam, points_per_side=32, pred_iou_thresh=0.88,
    stability_score_thresh=0.90, min_mask_region_area=500,
)
print("SAM ready.")

# ── Lazy-load EasyOCR (heavy — only init on first use) ────────────────────────
_ocr_reader = None
_ocr_lock   = threading.Lock()

def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        with _ocr_lock:
            if _ocr_reader is None:
                import easyocr
                print("Loading EasyOCR...")
                _ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
                print("EasyOCR ready.")
    return _ocr_reader

app = FastAPI(title="BFS Measurement Service")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Pydantic models ────────────────────────────────────────────────────────────
class ZoneBox(BaseModel):
    id: int
    x0pct: float; y0pct: float; x1pct: float; y1pct: float

class PolygonRequest(BaseModel):
    pdf_b64: str
    page_number: int = 1
    scale_str: Optional[str] = None

class MeasureRequest(BaseModel):
    image_b64: str
    pdf_b64: Optional[str] = None
    page_number: Optional[int] = 1
    zones: List[ZoneBox]
    scale_str: Optional[str] = None
    dpi: Optional[int] = DPI_DEFAULT

class ZoneResult(BaseModel):
    id: int; gross_sf: float; opening_sf: float; net_sf: float; method: str

class MeasureResponse(BaseModel):
    zones: List[ZoneResult]; method: str


# ── Scale helpers ──────────────────────────────────────────────────────────────
_SCALE_PATTERNS = [
    # "1/8" = 1'-0"", "3/16"=1'", "1/4"=1'-0""
    (r'(\d+)\s*/\s*(\d+)\s*["″]?\s*=\s*(\d+)', lambda m: int(m.group(2)) / int(m.group(1))),
    # "1:96", "1:48"
    (r'1\s*:\s*(\d+)', lambda m: int(m.group(1)) / 12.0),
    # "scale 1/8", "scale: 1/4"
    (r'scale\s*:?\s*(\d+)\s*/\s*(\d+)', lambda m: int(m.group(2)) / int(m.group(1))),
]

def parse_scale_str(s: str) -> float:
    """Parse any common scale string → ft per paper inch."""
    if not s:
        return 8.0
    for pattern, calc in _SCALE_PATTERNS:
        m = re.search(pattern, s, re.IGNORECASE)
        if m:
            try:
                return calc(m)
            except ZeroDivisionError:
                pass
    # plain fraction like "1/8"
    m = re.search(r'(\d+)/(\d+)', s)
    if m and int(m.group(1)):
        return int(m.group(2)) / int(m.group(1))
    return 8.0

def detect_scale_ocr(pdf_bytes: bytes, page_number: int) -> Optional[str]:
    """
    Use EasyOCR to read the scale annotation from the drawing title block.
    Returns a scale string like '1/8"=1\'-0"' or None if not found.
    """
    try:
        reader = get_ocr_reader()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[page_number - 1]
        # Render at 2x — good balance of OCR accuracy vs speed
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        doc.close()

        results = reader.readtext(img)
        # Combine adjacent text into lines for better pattern matching
        texts = [text for _, text, conf in results if conf > 0.4]
        full_text = " ".join(texts)

        for pattern, _ in _SCALE_PATTERNS:
            m = re.search(pattern, full_text, re.IGNORECASE)
            if m:
                detected = m.group(0).strip()
                print(f"  EasyOCR detected scale: '{detected}'")
                return detected

        # Also search each text box individually
        for _, text, conf in results:
            if conf > 0.4:
                for pattern, _ in _SCALE_PATTERNS:
                    if re.search(pattern, text, re.IGNORECASE):
                        print(f"  EasyOCR detected scale: '{text.strip()}'")
                        return text.strip()
    except Exception as e:
        print(f"  EasyOCR scale detection failed: {e}")
    return None

def get_scale(scale_str: Optional[str], pdf_bytes: bytes, page_number: int) -> float:
    """
    Resolve ft_per_inch from:
    1. Caller-provided scale string
    2. EasyOCR reading from the drawing
    3. Default 1/8"=1'-0" (ft_per_inch = 8)
    """
    if scale_str:
        return parse_scale_str(scale_str)
    # Try OCR
    ocr_str = detect_scale_ocr(pdf_bytes, page_number)
    if ocr_str:
        return parse_scale_str(ocr_str)
    print("  Scale not found — defaulting to 1/8\"=1'-0\" (8 ft/in)")
    return 8.0


# ── Area calculation with Shapely ─────────────────────────────────────────────
def pts_to_shapely(points, pw=1.0, ph=1.0, normalized=True):
    """Convert point list to Shapely Polygon. Points can be normalized [0-1] or PDF pts."""
    if normalized:
        coords = [(x * pw, y * ph) for x, y in points]
    else:
        coords = [(x, y) for x, y in points]
    if len(coords) < 3:
        return None
    try:
        p = SPoly(coords)
        if not p.is_valid:
            p = make_valid(p)
        return p if p.area > 0 else None
    except Exception:
        return None

def shapely_area_sf(poly: SPoly, ft_per_inch: float) -> float:
    """Convert Shapely polygon area (PDF pts²) → square feet."""
    ft_per_pt = ft_per_inch / 72.0
    return poly.area * (ft_per_pt ** 2)

def polygon_area_pts(points) -> float:
    """Shoelace formula fallback — area in whatever units points are in."""
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2.0

def pdf_pts_to_sf(area_pts: float, ft_per_inch: float) -> float:
    ft_per_pt = ft_per_inch / 72.0
    return area_pts * (ft_per_pt ** 2)

def px_to_sf(px: int, ft_per_inch: float, dpi: int) -> float:
    ppf = dpi / ft_per_inch
    return px / (ppf * ppf)


# ── Coordinate helpers ────────────────────────────────────────────────────────
def pct_to_pts(x0pct, y0pct, x1pct, y1pct, pw, ph):
    return pw*x0pct/100, ph*y0pct/100, pw*x1pct/100, ph*y1pct/100

def pct_to_px(x0pct, y0pct, x1pct, y1pct, w, h):
    return (max(0,int(w*x0pct/100)), max(0,int(h*y0pct/100)),
            min(w,int(w*x1pct/100)), min(h,int(h*y1pct/100)))

def _normalize_verts(verts, pw, ph):
    out = []
    for v in verts:
        if isinstance(v, (tuple, list)):
            out.append([round(v[0]/pw, 4), round(v[1]/ph, 4)])
        else:
            out.append([round(v.x/pw, 4), round(v.y/ph, 4)])
    return out


# ── Bluebeam annotation extraction ────────────────────────────────────────────
def extract_bluebeam_polygons(pdf_bytes: bytes, page_number: int, ft_per_inch: float = 8.0) -> list:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_number - 1]
    pw, ph = page.rect.width, page.rect.height
    results = []
    for a in page.annots():
        if a.type[0] != 6:
            continue
        verts = a.vertices or []
        if len(verts) < 3:
            continue
        content = a.info.get("content", "")
        fill    = a.colors.get("fill", [])
        rect    = a.rect
        sf_match = re.search(r"([\d,]+)\s*sf", content, re.IGNORECASE)
        if sf_match:
            area_sf = float(sf_match.group(1).replace(",", ""))
        else:
            raw_pts = [(v[0],v[1]) if isinstance(v,(tuple,list)) else (v.x,v.y) for v in verts]
            # Use Shapely for accurate area
            poly = pts_to_shapely(raw_pts, normalized=False)
            area_sf = round(shapely_area_sf(poly, ft_per_inch), 1) if poly else 0.0

        norm = _normalize_verts(verts, pw, ph)
        cx = round(sum(v[0] for v in norm)/len(norm), 4)
        cy = round(sum(v[1] for v in norm)/len(norm), 4)
        results.append({
            "id": len(results), "points": norm, "area_sf": area_sf,
            "cx": cx, "cy": cy,
            "bbox": [round(rect.x0/pw,4), round(rect.y0/ph,4), round(rect.x1/pw,4), round(rect.y1/ph,4)],
            "source": "bluebeam", "label": content,
            "fill_color": [round(c,3) for c in fill] if fill else None,
        })
    doc.close()
    return results


# ── CAD vector texture clustering ─────────────────────────────────────────────
def cluster_by_texture(drawings, pw, ph, ft_per_inch: float, min_cluster_sf: float = 10.0) -> list:
    clusters: dict = {}
    for d in drawings:
        fill = d.get("fill")
        if fill is None or len(fill) < 3:
            continue
        r_val = d.get("rect")
        if r_val is None:
            continue
        rect = fitz.Rect(r_val)
        if rect.is_empty or rect.width < 10 or rect.height < 10:
            continue
        if rect.get_area() < 200:
            continue
        key = tuple(round(c, 1) for c in fill[:3])
        if key == (0.0, 0.0, 0.0) or all(c >= 0.95 for c in key) or all(c < 0.15 for c in key):
            continue

        pts: list = []
        for item in d.get("items", []):
            if item[0] == "l":
                pts.append([float(item[1].x), float(item[1].y)])
            elif item[0] == "c":
                pts.append([float(item[3].x), float(item[3].y)])
            elif item[0] == "re":
                rr = fitz.Rect(item[1])
                pts = [[rr.x0,rr.y0],[rr.x1,rr.y0],[rr.x1,rr.y1],[rr.x0,rr.y1]]
                break
        if len(pts) < 3:
            pts = [[rect.x0,rect.y0],[rect.x1,rect.y0],[rect.x1,rect.y1],[rect.x0,rect.y1]]

        # Shapely area calculation
        poly = pts_to_shapely(pts, normalized=False)
        if poly is None or poly.area < 200:
            continue
        area_sf = pdf_pts_to_sf(poly.area, ft_per_inch)
        cx_val  = poly.centroid.x
        cy_val  = poly.centroid.y
        norm = [[round(x/pw,4), round(y/ph,4)] for x,y in pts]

        if key not in clusters:
            clusters[key] = []
        clusters[key].append({
            "points": norm, "area_sf": round(area_sf, 1),
            "cx": round(cx_val/pw, 4), "cy": round(cy_val/ph, 4),
            "bbox": [round(rect.x0/pw,4), round(rect.y0/ph,4), round(rect.x1/pw,4), round(rect.y1/ph,4)],
        })

    significant = {k: v for k, v in clusters.items() if sum(p["area_sf"] for p in v) >= min_cluster_sf}
    sorted_clusters = sorted(significant.items(), key=lambda x: sum(p["area_sf"] for p in x[1]), reverse=True)[:8]
    result: list = []
    for cluster_id, (fill_key, polys) in enumerate(sorted_clusters):
        for poly in sorted(polys, key=lambda p: p["area_sf"], reverse=True)[:30]:
            result.append({
                "id": len(result), "points": poly["points"], "area_sf": poly["area_sf"],
                "cx": poly["cx"], "cy": poly["cy"], "bbox": poly["bbox"],
                "source": "vector_cluster", "cluster_id": cluster_id, "fill_color": list(fill_key),
            })
    return result


# ── Claude Vision (claude-opus-4-8) ───────────────────────────────────────────
def claude_vision_segment(pdf_bytes: bytes, page_number: int, ft_per_inch: float) -> list:
    """
    Send the rendered page to Claude Vision for material region detection.
    Uses the most capable model for maximum accuracy on blank/CAD drawings.
    """
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return []

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_number - 1]
    pw, ph = page.rect.width, page.rect.height
    # 2x zoom = ~144 DPI — good detail without huge token cost
    pix     = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img_b64 = base64.b64encode(pix.tobytes("jpeg", jpg_quality=88)).decode()
    doc.close()

    client = anthropic.Anthropic(api_key=api_key)

    prompt = """You are analyzing an architectural exterior elevation drawing for a facade panel estimator.

Your task: identify every distinct material surface region in the ELEVATION VIEWS ONLY.
Ignore: title block, revision table, notes column, scale bar, north arrow, section cuts, floor plan portions.

MATERIAL IDENTIFICATION GUIDE — look for these drawing conventions:
- Brick / masonry: dense horizontal lines with staggered short vertical joints. Brick hatching.
- Metal panels / ACM: smooth rectangular areas divided by thin straight panel joint lines
- Glass / windows / curtain wall: rectangular openings, often with diagonal lines or X pattern, or just outlined
- Doors: rectangular openings at grade level, sometimes with swing arc
- Concrete / EIFS / stucco: plain fill, lightly stippled, or very light line texture
- Ribbed / corrugated metal: many closely-spaced parallel lines (horizontal or vertical)
- Soffit: underside horizontal surfaces, usually at eave or overhang
- Stone / precast: large block pattern, thicker outlines

CRITICAL RULES:
1. Trace the FULL extent of each material — roof line to grade, full width of that material zone
2. Windows and doors = separate "glass" and "door" polygons INSIDE the wall polygons
3. If you see the same material in multiple disconnected areas, create one polygon per area
4. Use more vertices for L-shapes, setbacks, or complex outlines — minimum 4 vertices
5. Coordinates: (0,0) = top-left of image, (1,1) = bottom-right. Normalized 0–1 range.
6. Be precise — wrong polygons mean wrong square footage on a real bid

Return ONLY this JSON (no markdown, no explanation):
{
  "regions": [
    {
      "material_type": "brick",
      "polygon": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],
      "confidence": 0.92,
      "notes": "main wall surface, north elevation"
    }
  ]
}

Allowed material_type values: brick, metal_panel, acm_panel, glass, door, concrete, eifs, stucco, ribbed_metal, soffit, stone, precast, fiber_cement, other"""

    try:
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                    {"type": "text", "text": prompt}
                ]
            }]
        )

        text = resp.content[0].text.strip()
        # Strip markdown code fence if present
        m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
        if m:
            text = m.group(1)
        # Find first { ... } block
        m = re.search(r'\{[\s\S]+\}', text)
        if m:
            text = m.group(0)

        data    = json.loads(text)
        regions = data.get("regions", [])
        if not regions:
            return []

        mat_to_cluster: dict = {}
        cluster_counter = 0
        result = []

        for region in regions:
            mat_type   = region.get("material_type", "other").lower().strip()
            poly_pts   = region.get("polygon", [])
            confidence = region.get("confidence", 0.8)

            if len(poly_pts) < 3 or confidence < 0.4:
                continue

            # Assign cluster ID by material type
            if mat_type not in mat_to_cluster:
                mat_to_cluster[mat_type] = cluster_counter
                cluster_counter += 1
            cluster_id = mat_to_cluster[mat_type]

            # Area via Shapely (normalized → PDF pts → SF)
            poly = pts_to_shapely(poly_pts, pw=pw, ph=ph, normalized=True)
            if poly is None:
                continue
            area_sf = round(shapely_area_sf(poly, ft_per_inch), 1)

            cx = round(poly.centroid.x / pw, 4)
            cy = round(poly.centroid.y / ph, 4)

            result.append({
                "id":            len(result),
                "points":        [[round(x, 4), round(y, 4)] for x, y in poly_pts],
                "area_sf":       area_sf,
                "cx":            cx,
                "cy":            cy,
                "bbox":          [round(min(p[0] for p in poly_pts), 4), round(min(p[1] for p in poly_pts), 4),
                                  round(max(p[0] for p in poly_pts), 4), round(max(p[1] for p in poly_pts), 4)],
                "source":        "claude_vision",
                "cluster_id":    cluster_id,
                "material_type": mat_type,
                "confidence":    confidence,
            })

        print(f"  Claude Vision ({CLAUDE_MODEL}): {len(result)} regions, {len(mat_to_cluster)} material types")
        return result

    except Exception as e:
        print(f"  Claude Vision failed: {e}")
        return []


# ── SAM fallback (raster/scanned PDFs) ───────────────────────────────────────
def decode_image(b64: str) -> np.ndarray:
    data = base64.b64decode(b64)
    arr  = np.frombuffer(data, dtype=np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def sam_measure_zones(img_rgb, zone_boxes, ft_per_inch, dpi):
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
        gross_px  = int(wall_mask.sum())

        ys, xs = np.where(wall_mask)
        if len(xs) == 0:
            continue
        cy0, cy1 = max(0, ys.min()-10), min(h, ys.max()+10)
        cx0, cx1 = max(0, xs.min()-10), min(w, xs.max()+10)
        crop = img_rgb[cy0:cy1, cx0:cx1]

        opening_px = 0
        if crop.shape[0] > 80 and crop.shape[1] > 80 and gross_px > 30000:
            zone_crop = wall_mask[cy0:cy1, cx0:cx1]
            raw = auto_gen.generate(crop)
            for m in raw:
                seg  = m["segmentation"]
                area = m["area"]
                if area < 3000 or area > gross_px * 0.40:
                    continue
                overlap = np.logical_and(seg, zone_crop).sum()
                if overlap / max(area, 1) > 0.75 and area / max(gross_px, 1) < 0.50:
                    opening_px += area

        opening_px = min(opening_px, int(gross_px * 0.85))
        net_px     = max(gross_px - opening_px, 0)
        results[z.id] = {
            "gross_sf":   round(px_to_sf(gross_px,   ft_per_inch, dpi), 1),
            "opening_sf": round(px_to_sf(opening_px, ft_per_inch, dpi), 1),
            "net_sf":     round(px_to_sf(net_px,     ft_per_inch, dpi), 1),
        }
    return results

def is_vector_pdf(pdf_bytes: bytes, page_number: int) -> bool:
    doc   = fitz.open(stream=pdf_bytes, filetype="pdf")
    page  = doc[page_number - 1]
    drawings = page.get_drawings()
    doc.close()
    large = [d for d in drawings if d.get("rect") and fitz.Rect(d["rect"]).get_area() > 1000]
    return len(large) > 5


# ── Vector zone extraction ────────────────────────────────────────────────────
def extract_vector_zones(pdf_bytes: bytes, page_number: int, zone_boxes: list, ft_per_inch: float) -> dict:
    doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_number - 1]
    pw, ph = page.rect.width, page.rect.height
    drawings = page.get_drawings()
    doc.close()

    polygons = []
    for d in drawings:
        if d.get("width", 0) > 0 and d.get("fill") is None and d.get("width", 0) < 2:
            continue
        r = d.get("rect")
        if r is None:
            continue
        r = fitz.Rect(r)
        if r.is_empty or r.width < 10 or r.height < 10 or r.get_area() < 500:
            continue

        pts = []
        for item in d.get("items", []):
            if item[0] == "l":
                pts.append((float(item[1].x), float(item[1].y)))
            elif item[0] == "re":
                rr = fitz.Rect(item[1])
                pts = [(rr.x0,rr.y0),(rr.x1,rr.y0),(rr.x1,rr.y1),(rr.x0,rr.y1)]
                break

        if len(pts) < 3:
            pts = [(r.x0,r.y0),(r.x1,r.y0),(r.x1,r.y1),(r.x0,r.y1)]

        poly = pts_to_shapely(pts, normalized=False)
        if poly is None or poly.area < 100:
            continue

        polygons.append({"poly": poly, "area": poly.area, "cx": poly.centroid.x, "cy": poly.centroid.y, "rect": r})

    polygons.sort(key=lambda p: p["area"], reverse=True)

    results = {}
    for zbox in zone_boxes:
        bx0, by0, bx1, by1 = pct_to_pts(zbox.x0pct, zbox.y0pct, zbox.x1pct, zbox.y1pct, pw, ph)
        in_zone = [p for p in polygons if bx0 <= p["cx"] <= bx1 and by0 <= p["cy"] <= by1]
        if not in_zone:
            continue

        wall = in_zone[0]
        # Use Shapely to subtract openings
        openings_polys = [
            p["poly"] for p in in_zone[1:]
            if p["area"] < wall["area"] * 0.5 and wall["poly"].contains(p["poly"].centroid)
        ]
        if openings_polys:
            opening_shape = unary_union(openings_polys)
            net_poly  = wall["poly"].difference(opening_shape)
            opening_sf = round(shapely_area_sf(make_valid(opening_shape), ft_per_inch), 1)
            gross_sf   = round(shapely_area_sf(wall["poly"], ft_per_inch), 1)
            net_sf     = round(shapely_area_sf(make_valid(net_poly), ft_per_inch), 1)
        else:
            gross_sf   = round(shapely_area_sf(wall["poly"], ft_per_inch), 1)
            opening_sf = 0.0
            net_sf     = gross_sf

        opening_sf = min(opening_sf, gross_sf * 0.85)
        net_sf     = max(gross_sf - opening_sf, 0.0)
        results[zbox.id] = {"gross_sf": gross_sf, "opening_sf": opening_sf, "net_sf": net_sf}

    return results


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/measure", response_model=MeasureResponse)
def measure(req: MeasureRequest):
    pdf_bytes   = base64.b64decode(req.pdf_b64) if req.pdf_b64 else None
    ft_per_inch = get_scale(req.scale_str, pdf_bytes, req.page_number or 1) if pdf_bytes else parse_scale_str(req.scale_str or "")
    dpi         = req.dpi or DPI_DEFAULT
    print(f"/measure: {len(req.zones)} zones  ft_per_inch={ft_per_inch:.2f}")

    method = "sam"
    raw_results = {}

    if pdf_bytes:
        try:
            if is_vector_pdf(pdf_bytes, req.page_number or 1):
                raw_results = extract_vector_zones(pdf_bytes, req.page_number or 1, req.zones, ft_per_inch)
                if raw_results:
                    method = "vector"
        except Exception as e:
            print(f"  Vector extraction failed: {e}")

    if method == "sam":
        img_rgb = decode_image(req.image_b64)
        raw_results = sam_measure_zones(img_rgb, req.zones, ft_per_inch, dpi)

    out = []
    for z in req.zones:
        r = raw_results.get(z.id, {"gross_sf": 0, "opening_sf": 0, "net_sf": 0})
        out.append(ZoneResult(id=z.id, method=method, **r))
    return MeasureResponse(zones=out, method=method)


@app.post("/polygons")
def get_page_polygons(req: PolygonRequest):
    """
    Priority:
      1. Bluebeam polygon annotations
      2. CAD vector texture clustering
      3. Claude Vision (claude-opus-4-8)
    """
    pdf_bytes   = base64.b64decode(req.pdf_b64)
    ft_per_inch = get_scale(req.scale_str, pdf_bytes, req.page_number)

    doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[req.page_number - 1]
    pw, ph = page.rect.width, page.rect.height
    drawings = page.get_drawings()
    doc.close()

    # 1. Bluebeam
    bb = extract_bluebeam_polygons(pdf_bytes, req.page_number, ft_per_inch)
    if bb:
        print(f"/polygons [bluebeam]: {len(bb)} polygons")
        return {"polygons": bb, "width": pw, "height": ph, "count": len(bb), "method": "bluebeam"}

    # 2. Vector clustering
    clustered = cluster_by_texture(drawings, pw, ph, ft_per_inch)
    if clustered:
        n = len(set(p["cluster_id"] for p in clustered))
        print(f"/polygons [vector-cluster]: {len(clustered)} polygons, {n} texture groups")
        return {"polygons": clustered, "width": pw, "height": ph, "count": len(clustered), "method": "vector_cluster"}

    # 3. Claude Vision
    print(f"/polygons [claude-vision {CLAUDE_MODEL}]: sending page {req.page_number}")
    vision = claude_vision_segment(pdf_bytes, req.page_number, ft_per_inch)
    if vision:
        n = len(set(p["cluster_id"] for p in vision))
        return {"polygons": vision, "width": pw, "height": ph, "count": len(vision), "method": "claude_vision"}

    print(f"/polygons: no surfaces detected on page {req.page_number}")
    return {"polygons": [], "width": pw, "height": ph, "count": 0, "method": "none"}


@app.get("/health")
def health():
    return {"status": "ok", "device": device, "claude_model": CLAUDE_MODEL,
            "capabilities": ["bluebeam", "vector_cluster", "claude_vision", "sam", "easyocr_scale", "shapely"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8001)))
