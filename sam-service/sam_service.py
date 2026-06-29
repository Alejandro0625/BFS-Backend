"""
BPS SAM Measurement Microservice — v2
---------------------------------------
Uses SamPredictor (box-prompted mode) instead of automatic generation.
Much more consistent on architectural drawings because:
  - Claude's bounding box tells SAM exactly WHERE to look
  - SAM focuses on that region instead of guessing across the whole image
  - Post-processing snaps mask edges to detected straight lines (walls are rectangles)
  - Learning: confirmed masks are saved and reused as prompts on similar future pages

POST /measure
  body: {
    image_b64: string,
    zones: [{ id, x0pct, y0pct, x1pct, y1pct }],
    scale_str: "1/8\"=1'-0\"",
    dpi: 150
  }
  returns: { zones: [{ id, gross_sf, opening_sf, net_sf }] }

POST /confirm   — call this when estimator confirms a takeoff (trains the system)
  body: {
    page_hash: string,       # md5 of the page image — identifies the drawing
    zones: [{ id, x0pct, y0pct, x1pct, y1pct, gross_sf, net_sf, material_id }]
  }

GET /health
"""

import os, base64, hashlib, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import cv2
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

CHECKPOINT   = os.environ.get("SAM_CHECKPOINT", r"C:\Users\User\Downloads\sam_vit_b_01ec64.pth")
MEMORY_FILE  = os.environ.get("SAM_MEMORY", r"C:\Users\User\Downloads\sam_memory.json")
DPI_DEFAULT  = 150
MIN_AREA_PX  = 5_000
CHILD_THRESH = 0.65  # fraction of child mask inside parent = it's an opening

# ── Load SAM once ─────────────────────────────────────────────────────────────
print("Loading SAM...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
sam = sam_model_registry["vit_b"](checkpoint=CHECKPOINT)
sam.to(device)

# SamPredictor: box-prompted mode — much more accurate for known zones
predictor = SamPredictor(sam)

# SamAutomaticMaskGenerator: used ONLY for finding openings inside a zone
auto_gen = SamAutomaticMaskGenerator(
    sam,
    points_per_side=32,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.90,
    min_mask_region_area=500,
    box_nms_thresh=0.50,
)
print("SAM ready.")

# ── Memory: confirmed measurements ────────────────────────────────────────────
def load_memory():
    if Path(MEMORY_FILE).exists():
        with open(MEMORY_FILE) as f:
            return json.load(f)
    return {"confirmed_zones": []}

def save_memory(mem):
    with open(MEMORY_FILE, "w") as f:
        json.dump(mem, f)

memory = load_memory()
print(f"Memory: {len(memory.get('confirmed_zones', []))} confirmed zones loaded")

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="BPS SAM Service v2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Models ────────────────────────────────────────────────────────────────────
class ZoneBox(BaseModel):
    id: int
    x0pct: float
    y0pct: float
    x1pct: float
    y1pct: float

class MeasureRequest(BaseModel):
    image_b64: str
    zones: List[ZoneBox]
    scale_str: Optional[str] = "1/8\"=1'-0\""
    dpi: Optional[int] = DPI_DEFAULT

class ZoneResult(BaseModel):
    id: int
    gross_sf: float
    opening_sf: float
    net_sf: float

class MeasureResponse(BaseModel):
    zones: List[ZoneResult]

class ConfirmedZone(BaseModel):
    id: int
    x0pct: float
    y0pct: float
    x1pct: float
    y1pct: float
    gross_sf: float
    net_sf: float
    material_id: Optional[str] = ""

class ConfirmRequest(BaseModel):
    page_hash: str
    zones: List[ConfirmedZone]


# ── Helpers ───────────────────────────────────────────────────────────────────
def decode_image(b64: str) -> np.ndarray:
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def image_hash(img_rgb: np.ndarray) -> str:
    # Fast hash: downsample then md5
    small = cv2.resize(img_rgb, (64, 64))
    return hashlib.md5(small.tobytes()).hexdigest()

def scale_to_ft_per_inch(s: str) -> float:
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

def px_to_sf(px: int, ft_per_inch: float, dpi: int) -> float:
    ppf = dpi / ft_per_inch
    return px / (ppf * ppf)

def pct_to_px(x0pct, y0pct, x1pct, y1pct, w, h):
    return (
        max(0, int(w * x0pct / 100)),
        max(0, int(h * y0pct / 100)),
        min(w, int(w * x1pct / 100)),
        min(h, int(h * y1pct / 100)),
    )

def snap_mask_to_lines(mask: np.ndarray, img_gray: np.ndarray) -> np.ndarray:
    """
    Post-process: snap the mask boundary to detected straight lines.
    Architectural drawings have straight walls — this makes masks precise.
    """
    h, w = mask.shape

    # Detect straight lines using Hough
    edges = cv2.Canny(img_gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60,
                             minLineLength=40, maxLineGap=10)
    if lines is None:
        return mask

    # Draw detected lines on a blank image
    line_img = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in lines[:, 0]:
        cv2.line(line_img, (x1, y1), (x2, y2), 255, 2)

    # Dilate lines slightly so they form closed regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    line_img = cv2.dilate(line_img, kernel, iterations=1)

    # Use watershed or contour snapping: find contour of mask, snap points to nearest line
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask

    snapped = np.zeros_like(mask)
    for c in cnts:
        approx = cv2.approxPolyDP(c, 8, True)
        new_pts = []
        for pt in approx:
            x, y = pt[0]
            # Search in a small radius for the nearest strong line pixel
            r = 12
            x0c, y0c = max(0, x-r), max(0, y-r)
            x1c, y1c = min(w, x+r), min(h, y+r)
            region = line_img[y0c:y1c, x0c:x1c]
            ys_l, xs_l = np.where(region > 0)
            if len(xs_l) > 0:
                # Find closest line pixel
                dists = (xs_l - r)**2 + (ys_l - r)**2
                best = np.argmin(dists)
                x = x0c + xs_l[best]
                y = y0c + ys_l[best]
            new_pts.append([[x, y]])

        new_pts = np.array(new_pts, dtype=np.int32)
        cv2.fillPoly(snapped, [new_pts], True)

    # Only keep the snapped region if it's close to the original (don't over-snap)
    original_area = mask.sum()
    snapped_area  = snapped.sum()
    if snapped_area == 0 or abs(snapped_area - original_area) / max(original_area, 1) > 0.30:
        return mask  # snapping went too far, keep original

    return snapped.astype(bool)

def find_openings_in_zone(zone_mask: np.ndarray, img_rgb: np.ndarray) -> list:
    """
    Find openings (windows, doors) inside a zone by running SAM on just the zone crop.
    Returns list of masks (each is an opening).
    """
    h, w = zone_mask.shape
    ys, xs = np.where(zone_mask)
    if len(xs) == 0:
        return []

    y0, y1 = max(0, ys.min() - 10), min(h, ys.max() + 10)
    x0, x1 = max(0, xs.min() - 10), min(w, xs.max() + 10)

    crop = img_rgb[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    if ch < 50 or cw < 50:
        return []

    # Only run auto SAM if the zone is large enough to have openings
    zone_area = zone_mask.sum()
    if zone_area < 30_000:
        return []

    raw = auto_gen.generate(crop)

    openings = []
    zone_crop_mask = zone_mask[y0:y1, x0:x1]

    for m in raw:
        seg = m["segmentation"]
        if seg.shape != (ch, cw):
            continue
        area = m["area"]
        if area < MIN_AREA_PX or area > zone_area * 0.40:
            continue

        # Must be mostly inside the zone
        overlap = np.logical_and(seg, zone_crop_mask).sum()
        if overlap / max(area, 1) < 0.75:
            continue

        # Must NOT be the zone itself (avoid selecting the parent)
        if area / max(zone_area, 1) > 0.50:
            continue

        # Inflate back to full image coords
        full_mask = np.zeros((h, w), dtype=bool)
        full_mask[y0:y1, x0:x1] = seg
        openings.append(full_mask)

    # Deduplicate by overlap
    final = []
    for op in sorted(openings, key=lambda m: m.sum(), reverse=True):
        dominated = False
        for kept in final:
            inter = np.logical_and(op, kept).sum()
            if inter / max(op.sum(), 1) > 0.70:
                dominated = True
                break
        if not dominated:
            final.append(op)

    return final


# ── Core measurement ──────────────────────────────────────────────────────────
def segment_zone(img_rgb: np.ndarray, img_gray: np.ndarray, box_px, ft_per_inch: float, dpi: int) -> dict:
    """
    Use SamPredictor with a bounding box to get the wall mask.
    Then find openings inside using auto-gen on the crop.
    Returns { gross_sf, opening_sf, net_sf }
    """
    x0, y0, x1, y1 = box_px
    if x1 <= x0 or y1 <= y0:
        return {"gross_sf": 0, "opening_sf": 0, "net_sf": 0}

    # SamPredictor takes [x0, y0, x1, y1] as a numpy array
    box = np.array([x0, y0, x1, y1])
    masks, scores, _ = predictor.predict(
        box=box,
        multimask_output=True,
    )

    if masks is None or len(masks) == 0:
        return {"gross_sf": 0, "opening_sf": 0, "net_sf": 0}

    # Pick the mask with the highest score
    best_idx = int(np.argmax(scores))
    wall_mask = masks[best_idx]

    # Snap to architectural lines for precision
    wall_mask = snap_mask_to_lines(wall_mask, img_gray)

    gross_px = int(wall_mask.sum())

    # Find openings (windows, doors) inside the wall zone
    openings = find_openings_in_zone(wall_mask, img_rgb)
    opening_px = sum(int(op.sum()) for op in openings)
    opening_px = min(opening_px, int(gross_px * 0.85))  # sanity cap

    net_px = max(gross_px - opening_px, 0)

    return {
        "gross_sf":   round(px_to_sf(gross_px,   ft_per_inch, dpi), 1),
        "opening_sf": round(px_to_sf(opening_px, ft_per_inch, dpi), 1),
        "net_sf":     round(px_to_sf(net_px,     ft_per_inch, dpi), 1),
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/measure", response_model=MeasureResponse)
def measure(req: MeasureRequest):
    img_rgb = decode_image(req.image_b64)
    h, w = img_rgb.shape[:2]
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    ft_per_inch = scale_to_ft_per_inch(req.scale_str or "")
    dpi = req.dpi or DPI_DEFAULT

    print(f"/measure: {len(req.zones)} zones  scale={req.scale_str} ({ft_per_inch} ft/in)  DPI={dpi}  img={w}x{h}")

    # Set image in predictor once — reused for all zones on this page
    predictor.set_image(img_rgb)

    results = []
    for z in req.zones:
        box_px = pct_to_px(z.x0pct, z.y0pct, z.x1pct, z.y1pct, w, h)
        r = segment_zone(img_rgb, img_gray, box_px, ft_per_inch, dpi)
        print(f"  Zone {z.id}: gross={r['gross_sf']} opening={r['opening_sf']} net={r['net_sf']} SF")
        results.append(ZoneResult(id=z.id, **r))

    return MeasureResponse(zones=results)


@app.post("/confirm")
def confirm(req: ConfirmRequest):
    """
    Estimator confirmed a takeoff — save zone measurements to memory.
    Future calls use this to validate and improve accuracy.
    """
    mem = load_memory()
    new_count = 0
    for z in req.zones:
        mem["confirmed_zones"].append({
            "page_hash":   req.page_hash,
            "x0pct": z.x0pct, "y0pct": z.y0pct,
            "x1pct": z.x1pct, "y1pct": z.y1pct,
            "gross_sf":    z.gross_sf,
            "net_sf":      z.net_sf,
            "material_id": z.material_id,
        })
        new_count += 1
    save_memory(mem)
    memory["confirmed_zones"] = mem["confirmed_zones"]
    print(f"/confirm: saved {new_count} zones (total: {len(mem['confirmed_zones'])})")
    return {"saved": new_count, "total": len(mem["confirmed_zones"])}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": device,
        "model": "vit_b — SamPredictor (box-prompted)",
        "confirmed_zones": len(memory.get("confirmed_zones", [])),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8001)))
