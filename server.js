import express from "express";
import multer from "multer";
import cors from "cors";
import fetch from "node-fetch";
import { createCanvas } from "canvas";
import { PDFDocument, rgb, StandardFonts } from "pdf-lib";
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
app.use((req, res, next) => { req.setTimeout(600000); res.setTimeout(600000); next(); });

const API_KEY = process.env.ANTHROPIC_API_KEY;

// Store last analyzed PDF path for PDF generation
let lastPdfPath = null;

const IGNORE_MATERIALS = [
  "brick", "masonry", "stone", "cast stone", "eifs", "stucco",
  "concrete", "cmu", "glass", "curtainwall", "storefront",
  "roofing", "shingle", "tile", "wood siding", "vinyl"
];

const MATERIAL_COLORS = {
  "ACM Panel":              [0.78, 0.63, 0.19],
  "MCM Panel":              [0.78, 0.63, 0.19],
  "Fiber Cement Panel":     [0.35, 0.54, 0.35],
  "Fiber Cement Plank":     [0.29, 0.48, 0.42],
  "Nichiha Panel":          [0.48, 0.42, 0.67],
  "Aluminum Wall Panel":    [0.42, 0.60, 0.67],
  "Perforated Metal Panel": [0.67, 0.48, 0.35],
  "Soffit Panel":           [0.35, 0.48, 0.67],
  "Return/Trim":            [0.67, 0.35, 0.48],
  "Other":                  [0.48, 0.48, 0.48],
};

async function renderPage(pdfDoc, pageNum, scale) {
  const page = await pdfDoc.getPage(pageNum);
  const viewport = page.getViewport({ scale });
  const canvas = createCanvas(Math.floor(viewport.width), Math.floor(viewport.height));
  const ctx = canvas.getContext("2d");
  await page.render({ canvasContext: ctx, viewport }).promise;
  const b64 = canvas.toDataURL("image/jpeg", 0.75).split(",")[1];
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
  return data.content?.find((b) => b.type === "text")?.text || "";
}

function parseJSON(text) {
  try {
    const m = text.match(/```json\s*([\s\S]*?)```/);
    return JSON.parse(m ? m[1] : text);
  } catch {
    const s = text.indexOf("{");
    const e = text.lastIndexOf("}");
    if (s !== -1 && e !== -1) {
      try { return JSON.parse(text.slice(s, e + 1)); } catch {}
    }
    return null;
  }
}

function send(res, data) {
  res.write("data: " + JSON.stringify(data) + "\n\n");
}

// ─── Analysis endpoint ────────────────────────────────────────────────────────
app.post("/analyze", upload.single("pdf"), async (req, res) => {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders();

  try {
    send(res, { type: "log", msg: "Loading PDF...", level: "info" });

    lastPdfPath = req.file.path;
    const pdfDoc = await getDocument({ url: "file://" + req.file.path }).promise;
    const total = pdfDoc.numPages;

    send(res, { type: "log", msg: "Pages loaded: " + total + " — " + req.file.originalname, level: "ok" });
    send(res, { type: "phase", phase: "filtering" });
    send(res, { type: "log", msg: "Reading sheet index to identify relevant pages...", level: "info" });

    const relevant = { floorPlans: [], exteriorElevations: [], returnElevations: [], materialLegend: [], views3d: [], enlargedDetails: [] };

    const indexImages = [];
    for (let p = 1; p <= Math.min(5, total); p++) {
      send(res, { type: "progress", label: "Reading sheet index page " + p, pct: p * 4 });
      const b64 = await renderPage(pdfDoc, p, 0.8);
      indexImages.push({ type: "image", source: { type: "base64", media_type: "image/jpeg", data: b64 } });
    }

    indexImages.push({
      type: "text",
      text: "These are the first pages of a commercial architectural blueprint set with " + total + " total pages. Find the SHEET INDEX, DRAWING LIST, TABLE OF CONTENTS, or DRAWING SCHEDULE. We ONLY want sheets in the A-200 to A-228 range. Exterior elevations are A-200 through A-211. Return elevations are A-220 through A-228. Material legend is usually on the first elevation sheet. DO NOT include any A-300 series, A-350 series, or any wall sections, building sections, or construction details. Only exterior elevation and return elevation sheets. Look carefully at the sheet index table for these sheet numbers and match them to their PDF page numbers. Return ONLY JSON: {\"floorPlans\":[page numbers],\"exteriorElevations\":[page numbers],\"returnElevations\":[page numbers],\"materialLegend\":[page numbers],\"views3d\":[page numbers],\"enlargedDetails\":[page numbers]}"
    });

    const filterResult = parseJSON(await claude(indexImages, "You are a commercial panel siding estimator reading architectural sheet indexes. Identify only pages relevant to exterior panel cladding in the A-200 to A-228 sheet range. Return ONLY valid JSON."));
    if (filterResult) {
      Object.keys(relevant).forEach(k => { if (filterResult[k]) relevant[k].push(...filterResult[k]); });
    }

    send(res, { type: "log", msg: "Sheet index read: " + relevant.exteriorElevations.length + " elevations | " + relevant.returnElevations.length + " returns | " + relevant.materialLegend.length + " legend | " + relevant.views3d.length + " 3D views", level: "ok" });

    if (!relevant.exteriorElevations.length && !relevant.returnElevations.length) {
      throw new Error("No exterior elevation pages found. Make sure the PDF has a sheet index on the first few pages.");
    }

    // Read legend
    send(res, { type: "phase", phase: "legend" });
    send(res, { type: "log", msg: "Reading material legend...", level: "info" });
    send(res, { type: "progress", label: "Reading legend", pct: 25 });

    let legend = [];
    const legendPages = relevant.materialLegend.length ? relevant.materialLegend : relevant.exteriorElevations.slice(0, 2);

    for (const p of legendPages.slice(0, 3)) {
      const b64 = await renderPage(pdfDoc, p, 1.5);
      const raw = await claude(
        [
          { type: "image", source: { type: "base64", media_type: "image/jpeg", data: b64 } },
          { type: "text", text: "Find the EXTERIOR BUILDING MATERIALS LEGEND or FINISH SCHEDULE on this page. Extract ONLY panel materials: ACM panels, MCM panels, fiber cement panels, fiber cement plank, Nichiha panels, aluminum wall panels, perforated metal panels, soffit panels, returns and trim. IGNORE: brick, masonry, stone, EIFS, stucco, concrete, glass, curtainwall, roofing. Return ONLY JSON: {\"projectName\":\"if visible\",\"materials\":[{\"id\":\"e.g. 1 or ACM-1\",\"name\":\"full material name\",\"category\":\"ACM Panel|Fiber Cement Panel|Fiber Cement Plank|Nichiha Panel|Aluminum Wall Panel|Perforated Metal Panel|Soffit Panel|Return/Trim\",\"color\":\"color or finish if noted\",\"notes\":\"any spec notes\"}]}" },
        ],
        "You are a commercial panel siding estimator extracting material legends. Return ONLY valid JSON."
      );
      const parsed = parseJSON(raw);
      if (parsed && parsed.materials && parsed.materials.length) {
        legend = parsed.materials;
        send(res, { type: "log", msg: "Found " + legend.length + " panel materials: " + legend.map(function(m) { return m.id + " (" + m.name + ")"; }).join(", "), level: "ok" });
        send(res, { type: "legend", legend });
        break;
      }
    }

    if (!legend.length) {
      send(res, { type: "log", msg: "No legend found — will identify panel materials from drawing callouts", level: "warn" });
    }

    // Check floor plans for soffits
    let soffitNotes = [];
    if (relevant.floorPlans.length) {
      send(res, { type: "log", msg: "Checking floor plans for soffit and return locations...", level: "info" });
      for (const p of relevant.floorPlans.slice(0, 2)) {
        const b64 = await renderPage(pdfDoc, p, 1.0);
        const raw = await claude(
          [
            { type: "image", source: { type: "base64", media_type: "image/jpeg", data: b64 } },
            { type: "text", text: "This is a floor plan. Identify ALL soffit locations (overhangs, canopies, covered areas) and return locations (where panel wraps around corners). Return ONLY JSON: {\"soffits\":[{\"location\":\"description\",\"width\":\"dimension if visible\",\"depth\":\"dimension if visible\"}],\"returns\":[{\"location\":\"description\",\"height\":\"if visible\",\"depth\":\"if visible\"}]}" },
          ],
          "You are identifying soffit and return locations from architectural floor plans for a panel siding takeoff."
        );
        const parsed = parseJSON(raw);
        if (parsed && ((parsed.soffits && parsed.soffits.length) || (parsed.returns && parsed.returns.length))) {
          soffitNotes.push(parsed);
          send(res, { type: "log", msg: "Floor plan page " + p + ": " + (parsed.soffits ? parsed.soffits.length : 0) + " soffits, " + (parsed.returns ? parsed.returns.length : 0) + " returns", level: "ok" });
        }
      }
    }

    // Analyze elevations
    send(res, { type: "phase", phase: "analyzing" });
    const elevPages = [
      ...relevant.exteriorElevations.map(function(p) { return { p: p, type: "elevation" }; }),
      ...relevant.returnElevations.map(function(p) { return { p: p, type: "return" }; }),
      ...relevant.enlargedDetails.map(function(p) { return { p: p, type: "detail" }; }),
    ];

    send(res, { type: "log", msg: "Analyzing " + elevPages.length + " elevation pages...", level: "info" });

    const legendCtx = legend.length ? "PANEL MATERIAL LEGEND: " + JSON.stringify(legend) : "Identify panel materials from callouts and labels on the drawing.";
    const soffitCtx = soffitNotes.length ? "SOFFIT AND RETURN NOTES FROM FLOOR PLANS: " + JSON.stringify(soffitNotes) : "";
    const takeoffData = [];

    for (let i = 0; i < elevPages.length; i++) {
      const ep = elevPages[i];
      const p = ep.p;
      const type = ep.type;
      send(res, { type: "progress", label: "Analyzing elevation " + (i + 1) + " of " + elevPages.length, pct: 35 + Math.round((i / elevPages.length) * 55) });

      const b64 = await renderPage(pdfDoc, p, 1.5);
      const prompt = "You are a senior commercial PANEL SIDING estimator doing a material takeoff.\n\n" + legendCtx + "\n" + soffitCtx + "\n\nPage " + p + " type: " + type + ".\n\nINSTRUCTIONS:\n1. Read the drawing title\n2. Read the sheet reference\n3. Read the SCALE printed on the drawing\n4. For EACH panel material zone: identify material, calculate GROSS area using scale, list ALL openings, NET = Gross minus Openings\n5. SOFFITS: measure underside of overhangs (width x depth = SF)\n6. RETURNS: measure corner wraps (height x return depth = SF)\n7. IGNORE: brick, masonry, stone, EIFS, stucco, concrete, glass, curtainwall\n8. If this is a BUILDING SECTION or WALL SECTION not an elevation, return 0 zones\n\nReturn ONLY JSON:\n{\"pageNumber\":" + p + ",\"elevations\":[{\"title\":\"\",\"sheetRef\":\"\",\"scale\":\"\",\"building\":\"\",\"direction\":\"\",\"zones\":[{\"materialId\":\"\",\"materialName\":\"\",\"category\":\"\",\"description\":\"\",\"grossArea\":0,\"totalOpeningArea\":0,\"netArea\":0}],\"flags\":[]}]}";

      const raw = await claude(
        [
          { type: "image", source: { type: "base64", media_type: "image/jpeg", data: b64 } },
          { type: "text", text: prompt },
        ],
        "You are a senior commercial panel siding estimator. Focus ONLY on panel materials. Return ONLY valid JSON."
      );

      const parsed = parseJSON(raw);
      if (parsed && parsed.elevations && parsed.elevations.length) {
        parsed.elevations.forEach(function(e) {
          e.pageNumber = p;
          e.zones = (e.zones || []).filter(function(z) {
            const nameLower = (z.materialName || "").toLowerCase();
            return !IGNORE_MATERIALS.some(function(ig) { return nameLower.includes(ig); });
          });
        });
        takeoffData.push(...parsed.elevations);
        parsed.elevations.forEach(function(e) {
          const sf = (e.zones || []).reduce(function(s, z) { return s + (z.netArea || 0); }, 0);
          if (sf > 0) send(res, { type: "log", msg: "  " + e.title + " (" + e.sheetRef + ") — " + (e.zones ? e.zones.length : 0) + " zones, " + Math.round(sf) + " SF", level: "ok" });
          (e.flags || []).filter(Boolean).forEach(function(f) { send(res, { type: "log", msg: "    " + f, level: "warn" }); });
        });
        send(res, { type: "elevation", data: parsed.elevations });
      } else {
        send(res, { type: "log", msg: "Page " + p + ": could not read — manual review needed", level: "warn" });
      }
    }

    // 3D cross reference
    if (relevant.views3d.length) {
      send(res, { type: "progress", label: "Cross-referencing 3D views", pct: 93 });
      send(res, { type: "log", msg: "Cross-checking 3D views for missed soffits and returns...", level: "info" });
      const b64 = await renderPage(pdfDoc, relevant.views3d[0], 1.0);
      const cr = parseJSON(await claude(
        [
          { type: "image", source: { type: "base64", media_type: "image/jpeg", data: b64 } },
          { type: "text", text: legendCtx + "\n\nThis is a 3D exterior view. Look for SOFFITS and RETURNS that may be missed. Return ONLY JSON: {\"warnings\":[\"specific items\"],\"notes\":\"description\"}" },
        ],
        "Review 3D exterior renderings for missed soffit and return conditions."
      ));
      if (cr && cr.warnings && cr.warnings.length) {
        cr.warnings.forEach(function(w) { send(res, { type: "log", msg: "3D CHECK: " + w, level: "warn" }); });
      } else {
        send(res, { type: "log", msg: "3D cross-reference complete", level: "ok" });
      }
    }

    send(res, { type: "done", takeoffData: takeoffData, legend: legend, soffitNotes: soffitNotes, pdfPath: req.file.path });
    send(res, { type: "progress", label: "Complete", pct: 100 });
    send(res, { type: "log", msg: "Analysis complete — " + takeoffData.length + " elevations processed", level: "success" });
    res.end();

  } catch (err) {
    send(res, { type: "error", msg: err.message });
    send(res, { type: "log", msg: "Error: " + err.message, level: "error" });
    res.end();
  }
});

// ─── PDF Generation endpoint ──────────────────────────────────────────────────
app.post("/generate-pdf", express.json({ limit: "50mb" }), async (req, res) => {
  try {
    const { takeoffData } = req.body;
    const pdfPath = lastPdfPath;

    if (!pdfPath || !takeoffData) throw new Error("No PDF available. Run analysis first.");

    const pdfDoc = await getDocument({ url: "file://" + pdfPath }).promise;
    const outputPdf = await PDFDocument.create();
    const font = await outputPdf.embedFont(StandardFonts.HelveticaBold);
    const fontReg = await outputPdf.embedFont(StandardFonts.Helvetica);

    for (const elev of takeoffData) {
      if (!elev.pageNumber) continue;
      if (!elev.zones || elev.zones.length === 0) continue;
      const sf = elev.zones.reduce(function(s, z) { return s + (z.netArea || 0); }, 0);
      if (sf === 0) continue;

      const pageNum = elev.pageNumber;
      const elevCanvas = await renderPageToCanvas(pdfDoc, pageNum, 1.5);

      // Build combined canvas: elevation + legend strip at bottom
      const legendH = 32 + elev.zones.length * 22 + 12;
      const combined = createCanvas(elevCanvas.width, elevCanvas.height + legendH);
      const ctx = combined.getContext("2d");

      // Draw elevation
      ctx.drawImage(elevCanvas, 0, 0);

      // Legend strip background
      ctx.fillStyle = "#111108";
      ctx.fillRect(0, elevCanvas.height, elevCanvas.width, legendH);

      // Title
      ctx.fillStyle = "#e0cc80";
      ctx.font = "bold 15px Arial";
      ctx.fillText(elev.title + "  |  " + (elev.sheetRef || "") + "  |  Scale: " + (elev.scale || "N/A"), 12, elevCanvas.height + 22);

      // Material rows
      elev.zones.forEach(function(z, i) {
        const y = elevCanvas.height + 42 + i * 22;
        const color = MATERIAL_COLORS[z.category] || [0.48, 0.48, 0.48];
        const hex = "#" + color.map(function(c) { return Math.round(c * 255).toString(16).padStart(2, "0"); }).join("");

        ctx.fillStyle = hex;
        ctx.fillRect(12, y - 13, 14, 14);

        ctx.fillStyle = "#ccc4aa";
        ctx.font = "12px Arial";
        const label = (z.materialId ? z.materialId + ": " : "") + z.materialName +
          "   Net: " + Math.round(z.netArea || 0) + " SF" +
          "   Adj (+15%): " + Math.round((z.netArea || 0) * 1.15) + " SF";
        ctx.fillText(label, 32, y);
      });

      const imgBytes = combined.toBuffer("image/png");
      const embeddedImg = await outputPdf.embedPng(imgBytes);

      // Landscape letter page
      const pageW = 1056;
      const pageH = 816;
      const scale = Math.min(pageW / combined.width, pageH / combined.height);

      const newPage = outputPdf.addPage([pageW, pageH]);

      // Dark background
      newPage.drawRectangle({ x: 0, y: 0, width: pageW, height: pageH, color: rgb(0.05, 0.05, 0.04) });

      newPage.drawImage(embeddedImg, {
        x: (pageW - combined.width * scale) / 2,
        y: (pageH - combined.height * scale) / 2,
        width: combined.width * scale,
        height: combined.height * scale,
      });
    }

    const pdfBytes = await outputPdf.save();
    res.setHeader("Content-Type", "application/pdf");
    res.setHeader("Content-Disposition", "attachment; filename=BPS_Takeoff_Verified.pdf");
    res.send(Buffer.from(pdfBytes));

  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get("/health", (req, res) => res.json({ status: "ok" }));

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log("BPS Estimator backend running on port " + PORT));
