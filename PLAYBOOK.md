# BFS AI Estimator — System Playbook
*Written 2026-07-06. The complete operating manual. Any future Claude session also has this context via memory files at `C:\Users\User\.claude\projects\...\memory\` — the project brain persists across models and sessions.*

## The system
- **App (live)**: https://bfs-estimator.vercel.app
- **Frontend repo**: `Alejandro0625/BFS--Estimator` → `src/App.jsx` (local: `C:\Users\User\Downloads\BFS-Estimator\src\App.jsx`) — Vercel auto-deploys `main`
- **Backend repo**: `Alejandro0625/BFS-Backend` → `clean-backend/` (local: `C:\Users\User\Downloads\bfs-clean-backend\`) — Railway auto-deploys, service `cheerful-generosity`, live URL `https://cheerful-generosity-production-96e0.up.railway.app`
- **Tokens** (GitHub/Vercel/Railway/Kaggle): in the memory file `project_bps_deploy.md`

## How a drawing is read (engine order on clean/unmarked pages)
1. **vector_hatch.py** — reads DRAWN geometry: gray-fill wall slabs + seam-line trains. Faces welded into one piece per pattern (shapely exact-geometry union, mitred = square corners), windows/doors detected by evidence (nested frame or mullions, plausible size) and subtracted, borders snapped to structural lines. This is the primary engine and matches the estimator's gold standard within ~5% on Fleet.
2. **model_infer.py** — trained U-Net (v10 weights on the Railway volume `/data/model.onnx`), tiled 768px @ 2560 render. Fallback for scanned/no-vector pages.
3. **texture.py** — classical CV texture, last resort.
- **Marked Bluebeam drawings** never touch the engines: `extract_page_polygons/polylines` reads the estimator's own measurements EXACTLY (verified constantly: Fleet 1364/2209/4630/1653).
- **callouts.py** — names regions from the drawing's text: legend keys (M1→spec) first, then arrowhead/leader, strict material-name gate. Reads material quantity schedules ("M1 2,049 sf" rows) → `drawingSchedule`.
- **ocr_text.py** — for flattened sets (text drawn as curves): RapidOCR **vendored** in `rapidocr_vendor.zip` (lazy-extracted; NEVER pip-install rapidocr directly — it pulls full opencv which cannot boot on Railway and took the service down once). Live-verified reading "METL-SPAN CF LIGHT MESA" off Fleet.
- **snap_fill.py** — bucket = click a pattern → ALL faces with that pattern on the page (siblings), welded, net of openings. Corner tool + exact-on-select unchanged.

## The learning flywheel (all live)
- Renames, deletes (NOT-CLADDING negatives), draw shapes, bucket confirms → POST `/learn` → `/data/corrections/`
- **Every Excel export saves the complete confirmed takeoff** (`source: final-*`) = answer keys
- **Estimator gold example** banked: jobId `estimator-gold-fleet` (his hand-marked Fleet, 10 polygons with SF+colors)
- `/autonomy-status` compares first auto output vs answer keys → Model tab shows the meter (appears after first export)

## v12 MODEL TRAINING — RUNNING NOW on Kaggle
- Kernel `bostonfacades/bfs-takeoff-v12` (v10's proven recipe + grad clip + lr 7e-4, on the big `takeoff-tiles-v6` dataset). Expect ~6–10h.
- **When COMPLETE**: check log — if `BEST HONEST extent-IoU` **> 0.741** (v10's score), deploy it:
  1. Copy kernel `bostonfacades/bfs-onnx-export-v10`, point its input at v12's output, run → download `model.onnx`
  2. `POST {backend}/admin/upload-model?key=bfs-model-load` (multipart field `model`) — **requires explicit user approval in-session (classifier-gated)**
  3. Verify `/health` shows `auto_engine: "model"`, then run the test battery.
- If ≤ 0.741: keep v10, note results, adjust (more epochs / different seed).

## Deploy discipline (the rules that prevent outages)
1. **NEVER ship a dependency change without the fresh-venv deploy simulation**: `python -m venv t; t\Scripts\pip install -r requirements.txt; t\Scripts\python -c "import app"` (from clean-backend, plus feature smoke test).
2. **NEVER let anything pull `opencv-python` (full)** — backend must stay `opencv-python-headless` only. Shapely is safe (wheels bundle GEOS).
3. **NEVER edit App.jsx with PowerShell string ops** — it mojibakes UTF-8 (emoji/dashes). Use proper editors; check with a search for `ðŸ` before pushing.
4. **Zips created on Windows PowerShell use backslash entries** — build zips with Python `zipfile` (forward slashes) or Linux can't extract them.
5. Railway health-gate is NOT reliable protection — a failed deploy CAN take the service down (~4 min to revert: push previous file versions back).
6. After EVERY backend deploy run the battery: marked file must return exactly 1364/2209/4630/1653; clean Fleet ≈ 6697/5756; `/health` ok.

## Test assets
- Marked (exact-path): `C:\Users\User\Downloads\fleet_garage_elevations.pdf`
- Clean (vector engine): `V:\Bids 2026\Siding Bids 2026\00 - Submitted\26-231 RH White...rebid\Only elevations no mark ups.pdf`
- GOLD STANDARD (his hand markup, the target look + totals 11,843 SF): same folder, `Only elevations.pdf`
- Text-bearing CD set (naming/legend/schedule): `26-004A Haycon...Malden\1A Arch drawings- crop CD set.pdf` (+ clean copy in the session scratchpad)

## Known open items (next sessions)
1. Small-face coverage: MIN_SF=100 hides faces under 100 SF. Do NOT just lower it (caused weld-sprawl regression once — anti-sprawl guard now exists, but add a junk discriminator: keep small faces only when their pattern matches a big face's group).
2. Mixed-style faces (part slab + part ribbed) = 2 pieces / 2 clicks; teach equivalence via rename data.
3. OCR noise ("OPENNG KITH BRICK") — tighten the cladding filter or add spell-normalization.
4. Retrain with HER corrections + gold examples requires USER to upload new data to Kaggle (agent cannot upload company data).
5. Autonomy meter fills as exports happen — show the boss after ~5 real bids.

## The vision (unchanged)
Estimator uploads → system finds every face, cuts windows, names materials → she picks scope, prices, exports. Every correction trains it. When the autonomy meter lives near 100%: estimators only price.
