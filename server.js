import express from "express";
import multer from "multer";
import cors from "cors";
import fetch from "node-fetch";
import { createCanvas } from "canvas";
import { PDFDocument, rgb, StandardFonts } from "pdf-lib";
import fs from "fs";
import path from "path";
import pkg from "pdfjs-dist/legacy/build/pdf.js";
const { getDocument, GlobalWorkerOptions } = pkg;
GlobalWorkerOptions.workerSrc = "";

const app = express();
const upload = multer({
  storage: multer.diskStorage({
    destination: '/tmp',
    filename: (req, file, cb) => cb(null, Date.now() + '.pdf')
  }),
  limits: { fileSize: 500 * 1024 * 1024 }
});

app.use(cors());
app.use(express.json({ limit: "50mb" }));
app.use((req, res, next) => { req.setTimeout(0); res.setTimeout(0); next(); });

const API_KEY = process.env.ANTHROPIC_API_KEY;

// Store completed evidence PDF path for download
let evidencePdfPath = null;

const IGNORE_MATERIALS = [
  "brick", "masonry", "stone", "cast stone", "eifs", "stucco",
  "concrete", "cmu", "glass", "curtainwall", "storefront",
  "roofing", "shingle", "tile", "wood siding", "vinyl"
];

const MATERIAL_COLORS = {
  "ACM Panel":              { rgb: [0.78, 0.63, 0.19], hex: "#c8a030" },
  "MCM Panel":              { rgb: [0.78, 0.63, 0.19], hex: "#c8a030" },
  "Fiber Cement Panel":     { rgb: [0.35, 0.54, 0.35], hex: "#5a8a5a" },
  "Fiber Cement Plank":     { rgb: [0.29, 0.48, 0.42], hex: "#4a7a6a" },
  "Nichiha Panel":          { rgb: [0.48, 0.42, 0.67], hex: "#7a6aaa" },
  "Aluminum Wall Panel":    { rgb: [0.42, 0.60, 0.67], hex: "#6a99aa" },
  "Perforated Metal Panel": { rgb: [0.67, 0.48, 0.35], hex: "#aa7a5a" },
  "Soffit Panel":           { rgb: [0.35, 0.48, 0.67], hex: "#5a7aaa" },
  "Return/Trim":            { rgb: [0.67, 0.35, 0.48], hex: "#aa5a7a" },
  "Other":                  { rgb: [0.48, 0.48, 0.48], hex: "#7a7a7a" },
};

async function renderPage(pdfDoc, pageNum, scale) {
  const page = await pdfDoc.getPage(pageNum);
  const viewport = page.getViewport({ scale });
  const canvas = createCanvas(Math.floor(viewport.width), Math.floor(viewport.height));
  const ctx = canvas.getContext("2d");
  await page.render({ canvasContext: ctx, viewport }).promise;
  const b64 = canvas.toDataURL("image/jpeg", 0.80).split(",")[1];
  page.cleanup();
  return b64;
}

async function renderPageToCanvas(pdfDoc, pageNum, scale) {
  const page = await pdfDoc.getPage(pageNum);
  const viewport = page.getViewport({ scale });
  const canvas = createCanvas(Math.floor(viewport.width), Math.floor(viewport.height));
  const ctx = canvas.getContext("2d");
  await page.render({ canvasContext: ctx, viewport }).promise;
  page.cleanup();
  return canvas;
}

async function getPageText(pdfDoc, pageNum) {
  try {
    const page = await pdfDoc.getPage(pageNum);
    const content = await page.getTextContent();
    const text = content.items.map(i => i.str).join(" ");
    page.cleanup();
    return text;
  } catch(e) { return ""; }
}

async function claude(content, system) {
  const res = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": API_KEY,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({
      model: "claude-opus-4-5",
      max_tokens: 4000,
      system,
      messages: [{ role: "user", content }],
    }),
  });
  const data = await res.json();
  if (data.error) throw new Error(data.error.message);
  return data.content?.find(b => b.type === "text")?.text || "";
}

function parseJSON(text) {
  try {
    const m = text.match(/```json\s*([\s\S]*?)```/);
    return JSON.parse(m ? m[1] : text);
  } catch {
    const s = text.indexOf("{"), e = text.lastIndexOf("}");
    if (s !== -1 && e !== -1) try { return JSON.parse(text.slice(s, e + 1)); } catch {}
    return null;
  }
}

function send(res, data) {
  res.write("data: " + JSON.stringify(data) + "\n\n");
}

// Build one evidence page for an elevation
async function buildEvidencePage(outputPdf, font, fontReg, elevCanvas, elev) {
  const zones = (elev.zones || []).filter(z => (z.netArea || 0) > 0);
  if (!zones.length) return;

  // Legend strip height
  const legendH = 36 + zones.length * 24 + 14;
  const combined = createCanvas(elevCanvas.width, elevCanvas.height + legendH);
  const ctx = combined.getContext("2d");

  // Draw elevation drawing
  ctx.drawImage(elevCanvas, 0, 0);

  // Legend strip background
  ctx.fillStyle = "#0f0e0b";
  ctx.fillRect(0, elevCanvas.height, elevCanvas.width, legendH);

  // Title line
  ctx.fillStyle = "#e0cc80";
  ctx.font = "bold 14px Arial";
  const title = (elev.title || "Elevation") + "   " + (elev.sheetRef || "") + "   Scale: " + (elev.scale || "N/A");
  ctx.fillText(title, 12, elevCanvas.height + 22);

  // Material rows with color swatches
  zones.forEach((z, i) => {
    const y = elevCanvas.height + 40 + i * 24;
    const color = MATERIAL_COLORS[z.category] || MATERIAL_COLORS["Other"];

    // Color swatch
    ctx.fillStyle = color.hex;
    ctx.fillRect(12, y - 14, 16, 16);
    ctx.strokeStyle = "#ffffff30";
    ctx.lineWidth = 1;
    ctx.strokeRect(12, y - 14, 16, 16);

    // Material label + SF
    ctx.fillStyle = "#ccc4aa";
    ctx.font = "12px Arial";
    const matLabel = (z.materialId ? z.materialId + ": " : "") + (z.materialName || z.category);
    const sfLabel = "   Gross: " + Math.round(z.grossArea || 0) + " SF   Openings: (" + Math.round(z.totalOpeningArea || 0) + ")   Net: " + Math.round(z.netArea) + " SF   Adj (+15%): " + Math.round(z.netArea * 1.15) + " SF";
    ctx.fillText(matLabel + sfLabel, 34, y);
  });

  // Total line
  const totalNet = zones.reduce((s, z) => s + (z.netArea || 0), 0);
  ctx.fillStyle = "#7ab87a";
  ctx.font = "bold 13px Arial";
  ctx.fillText("TOTAL: " + Math.round(totalNet) + " SF net   /   " + Math.round(totalNet * 1.15) + " SF adjusted (+15%)", 12, elevCanvas.height + legendH - 6);

  // Embed into PDF page
  const imgBytes = combined.toBuffer("image/png");
  const embeddedImg = await outputPdf.embedPng(imgBytes);

  const pageW = 1188;  // A3 landscape width
  const pageH = 840;
  const scale = Math.min(pageW / combined.width, pageH / combined.height);

  const newPage = outputPdf.addPage([pageW, pageH]);
  newPage.drawRectangle({ x: 0, y: 0, width: pageW, height: pageH, color: rgb(0.05, 0.05, 0.04) });
  newPage.drawImage(embeddedImg, {
    x: (pageW - combined.width * scale) / 2,
    y: (pageH - combined.height * scale) / 2,
    width: combined.width * scale,
    height: combined.height * scale,
  });

  // BPS watermark top right
  newPage.drawText("BOSTON PANEL SYSTEMS — AI TAKEOFF", {
    x: pageW - 280,
    y: pageH - 16,
    size: 8,
    font,
    color: rgb(0.35, 0.30, 0.18),
  });
}

// ─── Main Analysis + Evidence PDF endpoint ───────────────────────────────────
app.post("/analyze", upload.single("pdf"), async (req, res) => {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders();

  try {
    send(res, { type: "log", msg: "Loading PDF...", level: "info" });

    const pdfDoc = await getDocument({ url: "file://" + req.file.path }).promise;
    const total = pdfDoc.numPages;
    send(res, { type: "log", msg: "Pages loaded: " + total + " — " + req.file.originalname, level: "ok" });

    // Initialize evidence PDF
    const outputPdf = await PDFDocument.create();
    const font = await outputPdf.embedFont(StandardFonts.HelveticaBold);
    const fontReg = await outputPdf.embedFont(StandardFonts.Helvetica);

    // ── STEP 1: Extract text from all pages (FREE) ────────────────────────────
    send(res, { type: "phase", phase: "filtering" });
    send(res, { type: "log", msg: "Reading all page title blocks (free — no API cost)...", level: "info" });

    const pageIndex = [];
    for (let p = 1; p <= total; p++) {
      send(res, { type: "progress", label: "Reading title blocks " + p + "/" + total, pct: Math.round((p / total) * 15) });
      const text = await getPageText(pdfDoc, p);
      if (text.trim().length > 10) {
        pageIndex.push({ page: p, text: text.slice(0, 400) });
      }
    }

    send(res, { type: "log", msg: "Extracted text from " + pageIndex.length + " pages — classifying with AI (1 API call)...", level: "info" });

    // ── STEP 2: ONE Claude call to classify all pages ─────────────────────────
    const indexSummary = pageIndex.map(p => "Page " + p.page + ": " + p.text.replace(/\s+/g, " ").trim()).join("\n");

    const classifyResult = parseJSON(await claude(
      [{ type: "text", text: "You are a commercial panel siding estimator. Below is text from every page title block of a " + total + "-page architectural blueprint set.\n\nIdentify which pages contain:\n- EXTERIOR ELEVATIONS: outside building faces showing panel siding/cladding. Look for words like: elevation, facade, exterior, building face, enlarged elevation\n- RETURN ELEVATIONS: corner/return details. Look for: return, corner return, balcony return\n- MATERIAL LEGEND or FINISH SCHEDULE: exterior material key\n- FLOOR PLANS: building floor plans (needed only for soffit/return locations)\n- 3D VIEWS: exterior renderings or perspective views\n\nWe ONLY care about panel cladding materials: ACM, fiber cement, Nichiha, aluminum wall panels, perforated metal, soffits, returns.\nIGNORE: structural, mechanical, plumbing, electrical, interior, sections, civil, landscape, roofing.\n\nPAGE DATA:\n" + indexSummary + "\n\nReturn ONLY JSON: {\"exteriorElevations\":[page numbers],\"returnElevations\":[page numbers],\"materialLegend\":[page numbers],\"floorPlans\":[page numbers],\"views3d\":[page numbers]}" }],
      "You classify architectural blueprint pages for exterior panel siding estimation. Return ONLY valid JSON."
    ));

    const relevant = {
      floorPlans: (classifyResult && classifyResult.floorPlans) || [],
      exteriorElevations: (classifyResult && classifyResult.exteriorElevations) || [],
      returnElevations: (classifyResult && classifyResult.returnElevations) || [],
      materialLegend: (classifyResult && classifyResult.materialLegend) || [],
      views3d: (classifyResult && classifyResult.views3d) || [],
    };

    send(res, { type: "log", msg: "Found: " + relevant.exteriorElevations.length + " elevations | " + relevant.returnElevations.length + " returns | " + relevant.materialLegend.length + " legend | " + relevant.floorPlans.length + " floor plans | " + relevant.views3d.length + " 3D views", level: "ok" });

    if (!relevant.exteriorElevations.length && !relevant.returnElevations.length) {
      throw new Error("No exterior elevation pages found. Check that PDF has readable title block text.");
    }

    // ── STEP 3: Read material legend ──────────────────────────────────────────
    send(res, { type: "phase", phase: "legend" });
    send(res, { type: "log", msg: "Reading material legend...", level: "info" });
    send(res, { type: "progress", label: "Reading legend", pct: 20 });

    let legend = [];
    const legendPages = relevant.materialLegend.length ? relevant.materialLegend : relevant.exteriorElevations.slice(0, 2);

    for (const p of legendPages.slice(0, 3)) {
      const b64 = await renderPage(pdfDoc, p, 1.5);
      const raw = await claude(
        [
          { type: "image", source: { type: "base64", media_type: "image/jpeg", data: b64 } },
          { type: "text", text: "Find the EXTERIOR BUILDING MATERIALS LEGEND or FINISH SCHEDULE. Extract ONLY panel cladding materials: ACM panels, MCM panels, fiber cement panels/planks, Nichiha, aluminum wall panels, perforated metal panels, soffit panels, returns/trim. IGNORE: brick, masonry, stone, EIFS, stucco, concrete, glass, curtainwall, roofing, vapor barriers. Return ONLY JSON: {\"projectName\":\"if visible\",\"materials\":[{\"id\":\"e.g. 1 or ACM-1\",\"name\":\"full material name\",\"category\":\"ACM Panel|Fiber Cement Panel|Fiber Cement Plank|Nichiha Panel|Aluminum Wall Panel|Perforated Metal Panel|Soffit Panel|Return/Trim\",\"color\":\"color/finish if noted\",\"notes\":\"spec notes\"}]}" },
        ],
        "Extract exterior panel material legends from architectural drawings. Return ONLY valid JSON."
      );
      const parsed = parseJSON(raw);
      if (parsed && parsed.materials && parsed.materials.length) {
        legend = parsed.materials;
        send(res, { type: "log", msg: "Found " + legend.length + " panel materials: " + legend.map(m => m.id + " — " + m.name).join(", "), level: "ok" });
        send(res, { type: "legend", legend });
        break;
      }
    }

    if (!legend.length) {
      send(res, { type: "log", msg: "No dedicated legend found — will identify materials from drawing callouts", level: "warn" });
    }

    // ── STEP 4: Check floor plans for soffits/returns ─────────────────────────
    let soffitNotes = [];
    if (relevant.floorPlans.length) {
      send(res, { type: "log", msg: "Checking floor plans for soffit and return locations...", level: "info" });
      for (const p of relevant.floorPlans.slice(0, 2)) {
        const b64 = await renderPage(pdfDoc, p, 1.0);
        const raw = await claude(
          [
            { type: "image", source: { type: "base64", media_type: "image/jpeg", data: b64 } },
            { type: "text", text: "This is a floor plan for a commercial building. Identify ALL soffit locations (canopies, overhangs, covered walkways, bump-outs with undersides) and return locations (where panel wraps around building corners). For each note approximate dimensions if visible. Return ONLY JSON: {\"soffits\":[{\"location\":\"description\",\"width\":\"ft\",\"depth\":\"ft\"}],\"returns\":[{\"location\":\"description\",\"height\":\"ft\",\"depth\":\"ft\"}]}" },
          ],
          "Identify soffit and return locations from architectural floor plans for panel siding takeoff."
        );
        const parsed = parseJSON(raw);
        if (parsed && ((parsed.soffits && parsed.soffits.length) || (parsed.returns && parsed.returns.length))) {
          soffitNotes.push(parsed);
          send(res, { type: "log", msg: "Floor plan p." + p + ": " + (parsed.soffits || []).length + " soffits, " + (parsed.returns || []).length + " returns", level: "ok" });
        }
      }
    }

    // ── STEP 5: Analyze each elevation AND build evidence PDF page together ────
    send(res, { type: "phase", phase: "analyzing" });

    const elevPages = [
      ...relevant.exteriorElevations.map(p => ({ p, type: "elevation" })),
      ...relevant.returnElevations.map(p => ({ p, type: "return" })),
    ];

    send(res, { type: "log", msg: "Analyzing " + elevPages.length + " elevation pages + building evidence PDF simultaneously...", level: "info" });

    const legendCtx = legend.length ? "PANEL MATERIAL LEGEND: " + JSON.stringify(legend) : "Identify panel materials from callouts and labels on the drawing.";
    const soffitCtx = soffitNotes.length ? "\nSOFFIT & RETURN LOCATIONS FROM FLOOR PLANS: " + JSON.stringify(soffitNotes) : "";
    const takeoffData = [];

    for (let i = 0; i < elevPages.length; i++) {
      const { p, type } = elevPages[i];
      send(res, { type: "progress", label: "Analyzing + rendering page " + (i + 1) + " of " + elevPages.length, pct: 30 + Math.round((i / elevPages.length) * 60) });

      const b64 = await renderPage(pdfDoc, p, 1.5);

      const prompt = `You are a senior commercial PANEL SIDING estimator performing a precise material takeoff, following the same process as Bluebeam Revu manual takeoffs.

${legendCtx}${soffitCtx}

Page ${p} — type: ${type}.

TAKEOFF PROCESS:
1. Read the drawing TITLE and SHEET REFERENCE from the title block
2. Read the SCALE from the drawing (e.g. 1/8"=1'-0", 1/4"=1'-0", 1/16"=1'-0")
3. For each material zone visible on this elevation:
   - Identify the material using the legend (match by hatch pattern, label, or callout)
   - Calculate GROSS area using the scale: width × height in real-world feet → SF
   - List every opening to subtract: windows, doors, louvers, curtainwall, storefronts
   - NET area = Gross − Total Openings
4. SOFFITS: measure any underside of overhang or canopy (width × depth = SF) — flag separately
5. RETURNS: measure any corner wrap (height × return depth = SF) — flag separately  
6. BUMP-OUTS: treat as separate zones if they project from the main wall plane
7. HIDDEN ELEVATIONS: flag any condition where a face may not be fully visible
8. If scale is very small (1/32" or 1/16") note it as a flag — measurement is approximate
9. IGNORE: brick, masonry, stone, EIFS, stucco, concrete, glass, curtainwall, roofing, vapor barriers
10. If this is a BUILDING SECTION or WALL SECTION — return 0 zones

Return ONLY valid JSON:
{"pageNumber":${p},"elevations":[{"title":"","sheetRef":"","scale":"","building":"","direction":"","zones":[{"materialId":"","materialName":"","category":"","description":"","grossArea":0,"totalOpeningArea":0,"netArea":0}],"flags":[]}]}`;

      const raw = await claude(
        [
          { type: "image", source: { type: "base64", media_type: "image/jpeg", data: b64 } },
          { type: "text", text: prompt },
        ],
        "You are a senior commercial panel siding estimator. Perform precise Bluebeam-style material takeoffs. Focus ONLY on panel cladding materials. Return ONLY valid JSON."
      );

      const parsed = parseJSON(raw);
      if (parsed && parsed.elevations && parsed.elevations.length) {
        parsed.elevations.forEach(e => {
          e.pageNumber = p;
          e.zones = (e.zones || []).filter(z => {
            const n = (z.materialName || "").toLowerCase();
            return !IGNORE_MATERIALS.some(ig => n.includes(ig));
          });
        });

        takeoffData.push(...parsed.elevations);

        // Build evidence PDF page immediately for this elevation
        for (const elev of parsed.elevations) {
          const sf = (elev.zones || []).reduce((s, z) => s + (z.netArea || 0), 0);
          if (sf > 0) {
            try {
              const elevCanvas = await renderPageToCanvas(pdfDoc, p, 1.5);
              await buildEvidencePage(outputPdf, font, fontReg, elevCanvas, elev);
              send(res, { type: "log", msg: "  ✓ " + elev.title + " (" + elev.sheetRef + ") — " + (elev.zones || []).length + " zones, " + Math.round(sf) + " SF — evidence page added", level: "ok" });
            } catch(pdfErr) {
              send(res, { type: "log", msg: "  ✓ " + elev.title + " — " + Math.round(sf) + " SF (evidence page skipped: " + pdfErr.message + ")", level: "ok" });
            }
          }
          (elev.flags || []).filter(Boolean).forEach(f => send(res, { type: "log", msg: "    ⚠ " + f, level: "warn" }));
        }

        send(res, { type: "elevation", data: parsed.elevations });
      } else {
        send(res, { type: "log", msg: "  Page " + p + ": could not read — manual review needed", level: "warn" });
      }
    }

    // ── STEP 6: 3D cross-reference ────────────────────────────────────────────
    if (relevant.views3d.length) {
      send(res, { type: "progress", label: "Cross-referencing 3D views", pct: 92 });
      send(res, { type: "log", msg: "Cross-checking 3D views for missed soffits, returns, bump-outs...", level: "info" });
      const b64 = await renderPage(pdfDoc, relevant.views3d[0], 1.0);
      const cr = parseJSON(await claude(
        [
          { type: "image", source: { type: "base64", media_type: "image/jpeg", data: b64 } },
          { type: "text", text: legendCtx + "\n\nThis is a 3D exterior rendering. Look specifically for:\n1. SOFFITS — underside of overhangs, canopies, covered walkways\n2. RETURNS — where panel wraps around building corners\n3. BUMP-OUTS — wall projections that create additional surfaces\n4. HIDDEN ELEVATIONS — faces not clearly visible in flat elevation drawings\n\nFlag anything that may have been missed. Return ONLY JSON: {\"warnings\":[\"specific items with location\"],\"notes\":\"overall description of exterior\"}" },
        ],
        "Review 3D exterior renderings to catch missed soffits, returns, and bump-outs for panel siding takeoff."
      ));
      if (cr && cr.warnings && cr.warnings.length) {
        cr.warnings.forEach(w => send(res, { type: "log", msg: "3D CHECK: " + w, level: "warn" }));
      } else {
        send(res, { type: "log", msg: "3D cross-reference complete — no additional items flagged", level: "ok" });
      }
    }

    // ── STEP 7: Save evidence PDF ─────────────────────────────────────────────
    send(res, { type: "progress", label: "Saving evidence PDF...", pct: 96 });
    const pdfBytes = await outputPdf.save();
    evidencePdfPath = "/tmp/evidence_" + Date.now() + ".pdf";
    fs.writeFileSync(evidencePdfPath, Buffer.from(pdfBytes));
    send(res, { type: "log", msg: "Evidence PDF saved — " + outputPdf.getPageCount() + " pages", level: "ok" });

    send(res, { type: "done", takeoffData, legend, soffitNotes, evidenceReady: true });
    send(res, { type: "progress", label: "Complete", pct: 100 });
    send(res, { type: "log", msg: "✅ Done — " + takeoffData.length + " elevations processed — Excel + PDF ready", level: "success" });
    res.end();

  } catch (err) {
    send(res, { type: "error", msg: err.message });
    send(res, { type: "log", msg: "Error: " + err.message, level: "error" });
    res.end();
  }
});

// ─── Download evidence PDF ────────────────────────────────────────────────────
app.post("/evidence-pdf", (req, res) => {
  if (!evidencePdfPath || !fs.existsSync(evidencePdfPath)) {
    return res.status(404).json({ error: "No evidence PDF available. Run analysis first." });
  }
  res.setHeader("Content-Type", "application/pdf");
  res.setHeader("Content-Disposition", "attachment; filename=BPS_Takeoff_Evidence.pdf");
  fs.createReadStream(evidencePdfPath).pipe(res);
});

app.get("/health", (req, res) => res.json({ status: "ok" }));

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log("BPS Estimator backend running on port " + PORT));
