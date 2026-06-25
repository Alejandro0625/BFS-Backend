import express from "express";
import multer from "multer";
import cors from "cors";
import fetch from "node-fetch";
import { createCanvas } from "canvas";
import { PDFDocument, rgb, StandardFonts } from "pdf-lib";
import fs from "fs";
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
app.use(express.json({ limit: "10mb" }));

const API_KEY = process.env.ANTHROPIC_API_KEY;
const jobs = {};

function createJob(id) {
  jobs[id] = {
    id, status: "running",
    log: [], progress: { label: "", pct: 0 },
    phase: "filtering",
    takeoffData: [], legend: [], soffitNotes: [],
    evidencePdfPath: null, error: null,
  };
  return jobs[id];
}

function jobLog(job, msg, level = "info") {
  job.log.push({ msg, level, ts: Date.now() });
}

const IGNORE_MATERIALS = [
  "brick", "masonry", "stone", "cast stone", "eifs", "stucco",
  "concrete", "cmu", "glass", "curtainwall", "storefront",
  "roofing", "shingle", "tile", "wood siding", "vinyl"
];

const MATERIAL_COLORS = {
  "ACM Panel":              { hex: "#c8a030" },
  "MCM Panel":              { hex: "#c8a030" },
  "Fiber Cement Panel":     { hex: "#5a8a5a" },
  "Fiber Cement Plank":     { hex: "#4a7a6a" },
  "Nichiha Panel":          { hex: "#7a6aaa" },
  "Aluminum Wall Panel":    { hex: "#6a99aa" },
  "Perforated Metal Panel": { hex: "#aa7a5a" },
  "Soffit Panel":           { hex: "#5a7aaa" },
  "Return/Trim":            { hex: "#aa5a7a" },
  "Other":                  { hex: "#7a7a7a" },
};

async function renderPageOnce(pdfDoc, pageNum, scale) {
  const page = await pdfDoc.getPage(pageNum);
  const viewport = page.getViewport({ scale });
  const canvas = createCanvas(Math.floor(viewport.width), Math.floor(viewport.height));
  const ctx = canvas.getContext("2d");
  await page.render({ canvasContext: ctx, viewport }).promise;
  page.cleanup();
  const b64 = canvas.toDataURL("image/jpeg", 0.85).split(",")[1];
  return { canvas, b64 };
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

async function addEvidencePage(outputPdf, canvas, elev) {
  const zones = (elev.zones || []).filter(z => (z.netArea || 0) > 0);
  if (!zones.length) return;

  const legendH = 36 + zones.length * 24 + 18;
  const combined = createCanvas(canvas.width, canvas.height + legendH);
  const ctx = combined.getContext("2d");

  ctx.drawImage(canvas, 0, 0);
  ctx.fillStyle = "#0f0e0b";
  ctx.fillRect(0, canvas.height, canvas.width, legendH);
  ctx.fillStyle = "#e0cc80";
  ctx.font = "bold 14px Arial";
  ctx.fillText((elev.title || "") + "   " + (elev.sheetRef || "") + "   Scale: " + (elev.scale || "N/A"), 12, canvas.height + 22);

  zones.forEach((z, i) => {
    const y = canvas.height + 42 + i * 24;
    const color = MATERIAL_COLORS[z.category] || MATERIAL_COLORS["Other"];
    ctx.fillStyle = color.hex;
    ctx.fillRect(12, y - 14, 16, 16);
    ctx.fillStyle = "#ccc4aa";
    ctx.font = "12px Arial";
    const label = (z.materialId ? z.materialId + ": " : "") + (z.materialName || z.category) +
      "   Gross: " + Math.round(z.grossArea || 0) + " SF" +
      "   (-" + Math.round(z.totalOpeningArea || 0) + " openings)" +
      "   Net: " + Math.round(z.netArea) + " SF" +
      "   Adj +15%: " + Math.round(z.netArea * 1.15) + " SF";
    ctx.fillText(label, 34, y);
  });

  const totalNet = zones.reduce((s, z) => s + (z.netArea || 0), 0);
  ctx.fillStyle = "#7ab87a";
  ctx.font = "bold 13px Arial";
  ctx.fillText("TOTAL: " + Math.round(totalNet) + " SF net   /   " + Math.round(totalNet * 1.15) + " SF adj (+15%)", 12, canvas.height + legendH - 5);

  const imgBytes = combined.toBuffer("image/png");
  combined.width = 0; combined.height = 0;

  const embeddedImg = await outputPdf.embedPng(imgBytes);
  const pageW = 1188, pageH = 840;
  const newPage = outputPdf.addPage([pageW, pageH]);
  newPage.drawRectangle({ x: 0, y: 0, width: pageW, height: pageH, color: rgb(0.05, 0.05, 0.04) });
  newPage.drawImage(embeddedImg, { x: 0, y: 0, width: pageW, height: pageH });
}

// ─── Main analysis function ───────────────────────────────────────────────────
async function runAnalysis(job, pdfPath, originalName) {
  try {
    jobLog(job, "Loading PDF...", "info");
    const pdfDoc = await getDocument({ url: "file://" + pdfPath }).promise;
    const total = pdfDoc.numPages;
    jobLog(job, "Pages loaded: " + total + " — " + originalName, "ok");

    const outputPdf = await PDFDocument.create();
    await outputPdf.embedFont(StandardFonts.HelveticaBold);

    // ── STEP 1: Read drawing index visually at HIGH RES ───────────────────────
    job.phase = "filtering";
    job.progress = { label: "Reading drawing index", pct: 2 };
    jobLog(job, "Reading drawing index at high resolution...", "info");

    // Render first 8 pages at high res to catch the drawing index/list
    const indexPageImages = [];
    for (let p = 1; p <= Math.min(8, total); p++) {
      job.progress = { label: "Reading index page " + p, pct: p };
      const { canvas, b64 } = await renderPageOnce(pdfDoc, p, 2.0); // HIGH RES
      canvas.width = 0; canvas.height = 0;
      indexPageImages.push({ type: "image", source: { type: "base64", media_type: "image/jpeg", data: b64 } });
    }

    // Ask Claude to read the drawing index and extract ALL sheet names
    jobLog(job, "Extracting sheet list from drawing index...", "info");
    const sheetListResult = parseJSON(await claude(
      [
        ...indexPageImages,
        { type: "text", text: "These are the first pages of a commercial architectural blueprint set with " + total + " total pages.\n\nFind the DRAWING INDEX, SHEET INDEX, DRAWING LIST, or TABLE OF CONTENTS — it is a table listing every drawing with a sheet number and description.\n\nWe are a PANEL SIDING contractor. We ONLY care about ARCHITECTURAL sheets (sheets starting with 'A' or labeled as architectural). Ignore all civil, structural, mechanical, plumbing, electrical, landscape sheets.\n\nFrom the architectural sheets, identify which ones are:\n- EXTERIOR ELEVATIONS: outside building faces showing panel cladding (e.g. 'Exterior Elevations', 'Building Elevations', 'Enlarged Elevations')\n- RETURN ELEVATIONS: corner/return details (e.g. 'Return Elevations', 'Corner Details')\n- MATERIAL LEGEND or FINISH SCHEDULE: exterior material key\n- FLOOR PLANS: building floor plans\n- 3D VIEWS: exterior renderings or perspective views\n\nReturn ONLY JSON:\n{\n  \"sheets\": [\n    {\"number\": \"A1-1\", \"name\": \"Exterior Elevations\", \"type\": \"EXTERIOR_ELEVATION|RETURN_ELEVATION|MATERIAL_LEGEND|FLOOR_PLAN|3D_VIEW|IRRELEVANT\"}\n  ]\n}" }
      ],
      "You are reading an architectural drawing index to extract a complete sheet list. Read carefully — the index may have small text. Return ONLY valid JSON."
    ));

    if (!sheetListResult || !sheetListResult.sheets || sheetListResult.sheets.length === 0) {
      jobLog(job, "Could not read drawing index — falling back to text scan of all pages", "warn");
    } else {
      jobLog(job, "Drawing index read — found " + sheetListResult.sheets.length + " sheets", "ok");
      const relevant_types = sheetListResult.sheets.filter(s => s.type !== "IRRELEVANT");
      jobLog(job, "Relevant sheets: " + relevant_types.map(s => s.number + " (" + s.type + ")").join(", "), "dim");
    }

    // ── STEP 2: Scan every page text for the relevant sheet numbers ──────────
    job.progress = { label: "Scanning pages for sheet numbers", pct: 15 };
    jobLog(job, "Scanning all pages for sheet numbers (free)...", "info");

    const relevant = {
      exteriorElevations: [],
      returnElevations: [],
      materialLegend: [],
      floorPlans: [],
      views3d: [],
    };

    // Get the relevant A-sheets from the index
    const relevantSheets = (sheetListResult && sheetListResult.sheets)
      ? sheetListResult.sheets.filter(s => s.type !== "IRRELEVANT")
      : [];

    // Visually scan all pages in batches to match sheet numbers
    jobLog(job, "Scanning all pages visually to match sheet numbers...", "info");
    const sheetNums = relevantSheets.map(s => s.number);
    const BATCH = 15;

    for (let start = 1; start <= total; start += BATCH) {
      const end = Math.min(start + BATCH - 1, total);
      job.progress = { label: "Scanning pages " + start + "-" + end + " of " + total, pct: 15 + Math.round((start / total) * 10) };

      const content = [];
      for (let p = start; p <= end; p++) {
        const { canvas, b64 } = await renderPageOnce(pdfDoc, p, 0.2);
        canvas.width = 0; canvas.height = 0;
        content.push({ type: "text", text: "PAGE " + p + ":" });
        content.push({ type: "image", source: { type: "base64", media_type: "image/jpeg", data: b64 } });
      }

      content.push({ type: "text", text: "Look at the BOTTOM RIGHT corner title block of each page. Read the sheet number shown there (e.g. A1-1, A1-2, A-200, A2.1 etc). We are looking for these specific sheet numbers: " + sheetNums.join(", ") + ". Return ONLY JSON: {\"matches\":[{\"page\":1,\"sheetNumber\":\"A1-1\"}]}. Only include pages where the sheet number in the bottom right matches one from our list exactly." });

      const matchResult = parseJSON(await claude(content, "Match PDF pages to sheet numbers by reading title blocks. Return ONLY valid JSON."));

      if (matchResult && matchResult.matches) {
        matchResult.matches.forEach(match => {
          const sheet = relevantSheets.find(s => s.number === match.sheetNumber);
          if (sheet && match.page) {
            jobLog(job, "Matched " + match.sheetNumber + " → page " + match.page, "dim");
            if (sheet.type === "EXTERIOR_ELEVATION") relevant.exteriorElevations.push(match.page);
            else if (sheet.type === "RETURN_ELEVATION") relevant.returnElevations.push(match.page);
            else if (sheet.type === "MATERIAL_LEGEND") relevant.materialLegend.push(match.page);
            else if (sheet.type === "FLOOR_PLAN") relevant.floorPlans.push(match.page);
            else if (sheet.type === "3D_VIEW") relevant.views3d.push(match.page);
          }
        });
      }
    }

    // Fallback: keyword scan if visual matching found nothing
    if (!relevant.exteriorElevations.length && !relevant.returnElevations.length) {
      jobLog(job, "Visual matching found nothing — trying keyword fallback...", "warn");
      for (let p = 1; p <= total; p++) {
        const text = (await getPageText(pdfDoc, p)).toUpperCase();
        if (text.includes("EXTERIOR ELEVATION") || text.includes("BUILDING ELEVATION") || text.includes("ENLARGED ELEVATION")) {
          relevant.exteriorElevations.push(p);
        } else if (text.includes("RETURN ELEVATION") || text.includes("BALCONY RETURN")) {
          relevant.returnElevations.push(p);
        } else if (text.includes("FINISH SCHEDULE") || text.includes("MATERIAL LEGEND") || text.includes("EXTERIOR MATERIALS")) {
          relevant.materialLegend.push(p);
        }
      }
    }

    // Remove duplicates and sort
    Object.keys(relevant).forEach(k => { relevant[k] = [...new Set(relevant[k])].sort((a, b) => a - b); });

    jobLog(job, "Found: " + relevant.exteriorElevations.length + " elevations | " + relevant.returnElevations.length + " returns | " + relevant.materialLegend.length + " legend | " + relevant.floorPlans.length + " floor plans | " + relevant.views3d.length + " 3D views", "ok");

    if (!relevant.exteriorElevations.length && !relevant.returnElevations.length) {
      throw new Error("No exterior elevation pages found. The drawing index may not be readable or may be on a later page.");
    }

    // ── STEP 3: Read material legend ──────────────────────────────────────────
    job.phase = "legend";
    job.progress = { label: "Reading legend", pct: 28 };
    jobLog(job, "Reading material legend...", "info");

    let legend = [];
    const legendPages = relevant.materialLegend.length ? relevant.materialLegend : relevant.exteriorElevations.slice(0, 2);
    for (const p of legendPages.slice(0, 3)) {
      const { canvas, b64 } = await renderPageOnce(pdfDoc, p, 1.8);
      canvas.width = 0; canvas.height = 0;
      const raw = await claude(
        [
          { type: "image", source: { type: "base64", media_type: "image/jpeg", data: b64 } },
          { type: "text", text: "Find the EXTERIOR BUILDING MATERIALS LEGEND or FINISH SCHEDULE on this page. It may be a small table or legend box. Extract ONLY panel cladding materials: ACM panels, MCM panels, fiber cement panels/planks, Nichiha, aluminum wall panels, perforated metal panels, soffit panels, returns/trim. IGNORE: brick, masonry, stone, EIFS, stucco, concrete, glass, curtainwall, roofing, vapor barriers. Return ONLY JSON: {\"projectName\":\"if visible\",\"materials\":[{\"id\":\"material number/code e.g. 1 or ACM-1\",\"name\":\"full material name exactly as shown\",\"category\":\"ACM Panel|Fiber Cement Panel|Fiber Cement Plank|Nichiha Panel|Aluminum Wall Panel|Perforated Metal Panel|Soffit Panel|Return/Trim\",\"color\":\"color/finish if noted\",\"notes\":\"\"}]}" },
        ],
        "Extract exterior panel material legends from architectural drawings. Return ONLY valid JSON."
      );
      const parsed = parseJSON(raw);
      if (parsed && parsed.materials && parsed.materials.length) {
        legend = parsed.materials;
        jobLog(job, "Found " + legend.length + " materials: " + legend.map(m => m.id + " — " + m.name).join(", "), "ok");
        break;
      }
    }

    if (!legend.length) jobLog(job, "No dedicated legend found — will identify materials from drawing callouts", "warn");
    job.legend = legend;

    // ── STEP 4: Floor plans for soffits ──────────────────────────────────────
    let soffitNotes = [];
    if (relevant.floorPlans.length) {
      jobLog(job, "Checking floor plans for soffits and returns...", "info");
      for (const p of relevant.floorPlans.slice(0, 2)) {
        const { canvas, b64 } = await renderPageOnce(pdfDoc, p, 1.0);
        canvas.width = 0; canvas.height = 0;
        const raw = await claude(
          [
            { type: "image", source: { type: "base64", media_type: "image/jpeg", data: b64 } },
            { type: "text", text: "This is a floor plan. Identify ALL soffit locations (canopies, overhangs, covered walkways, bump-outs with undersides) and return locations (where panel wraps around building corners). Note approximate dimensions if visible. Return ONLY JSON: {\"soffits\":[{\"location\":\"description\",\"width\":\"ft if visible\",\"depth\":\"ft if visible\"}],\"returns\":[{\"location\":\"description\",\"height\":\"ft if visible\",\"depth\":\"ft if visible\"}]}" },
          ],
          "Identify soffit and return locations from floor plans for panel siding takeoff. Return ONLY valid JSON."
        );
        const parsed = parseJSON(raw);
        if (parsed && ((parsed.soffits && parsed.soffits.length) || (parsed.returns && parsed.returns.length))) {
          soffitNotes.push(parsed);
          jobLog(job, "Floor plan p." + p + ": " + (parsed.soffits || []).length + " soffits, " + (parsed.returns || []).length + " returns", "ok");
        }
      }
    }
    job.soffitNotes = soffitNotes;

    // ── STEP 5: Analyze each elevation ───────────────────────────────────────
    job.phase = "analyzing";
    const elevPages = [
      ...relevant.exteriorElevations.map(p => ({ p, type: "elevation" })),
      ...relevant.returnElevations.map(p => ({ p, type: "return" })),
    ];

    jobLog(job, "Analyzing " + elevPages.length + " elevation pages...", "info");

    const legendCtx = legend.length ? "PANEL MATERIAL LEGEND: " + JSON.stringify(legend) : "Identify panel materials from callouts and labels on the drawing.";
    const soffitCtx = soffitNotes.length ? "\nSOFFIT/RETURN NOTES FROM FLOOR PLANS: " + JSON.stringify(soffitNotes) : "";

    for (let i = 0; i < elevPages.length; i++) {
      const { p, type } = elevPages[i];
      job.progress = { label: "Page " + (i + 1) + " of " + elevPages.length, pct: 32 + Math.round((i / elevPages.length) * 60) };

      const { canvas, b64 } = await renderPageOnce(pdfDoc, p, 1.5);

      const prompt = "You are a senior commercial PANEL SIDING estimator performing a precise material takeoff — the same process as a manual Bluebeam Revu takeoff.\n\n" + legendCtx + soffitCtx + "\n\nPage " + p + " — " + type + ".\n\nTAKEOFF PROCESS:\n1. Read drawing TITLE and SHEET REFERENCE from the title block\n2. Read the SCALE printed on the drawing (e.g. 1/8\"=1'-0\", 1/4\"=1'-0\", 1/16\"=1'-0\")\n3. For each panel material zone:\n   - Identify material using the legend above\n   - GROSS area = width x height using the scale\n   - List ALL openings: windows, doors, louvers, curtainwall, storefronts\n   - NET = Gross minus Total Openings\n4. SOFFITS: measure underside of any overhang or canopy (width x depth = SF) — note separately\n5. RETURNS: measure any corner wrap (height x return depth = SF) — note separately\n6. BUMP-OUTS: treat projecting wall sections as separate zones\n7. IGNORE completely: brick, masonry, stone, EIFS, stucco, concrete, glass, curtainwall, roofing, vapor barriers\n8. If this is a BUILDING SECTION or WALL SECTION — return 0 zones\n\nReturn ONLY valid JSON:\n{\"pageNumber\":" + p + ",\"elevations\":[{\"title\":\"\",\"sheetRef\":\"\",\"scale\":\"\",\"building\":\"\",\"direction\":\"\",\"zones\":[{\"materialId\":\"\",\"materialName\":\"\",\"category\":\"\",\"description\":\"\",\"grossArea\":0,\"totalOpeningArea\":0,\"netArea\":0}],\"flags\":[]}]}";

      const raw = await claude(
        [
          { type: "image", source: { type: "base64", media_type: "image/jpeg", data: b64 } },
          { type: "text", text: prompt },
        ],
        "Senior commercial panel siding estimator. Precise Bluebeam-style takeoffs. Focus ONLY on panel cladding materials. Return ONLY valid JSON."
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

        job.takeoffData.push(...parsed.elevations);

        for (const elev of parsed.elevations) {
          const sf = (elev.zones || []).reduce((s, z) => s + (z.netArea || 0), 0);
          if (sf > 0) {
            try {
              await addEvidencePage(outputPdf, canvas, elev);
              jobLog(job, "  ✓ " + elev.title + " (" + elev.sheetRef + ") — " + Math.round(sf) + " SF", "ok");
            } catch(e) {
              jobLog(job, "  ✓ " + elev.title + " — " + Math.round(sf) + " SF", "ok");
            }
          }
          (elev.flags || []).filter(Boolean).forEach(f => jobLog(job, "    ⚠ " + f, "warn"));
        }
      } else {
        jobLog(job, "  Page " + p + ": could not read — manual review needed", "warn");
      }

      canvas.width = 0; canvas.height = 0;
    }

    // ── STEP 6: 3D cross-reference ────────────────────────────────────────────
    if (relevant.views3d.length) {
      job.progress = { label: "3D cross-reference", pct: 94 };
      jobLog(job, "Cross-checking 3D views for missed soffits and returns...", "info");
      const { canvas, b64 } = await renderPageOnce(pdfDoc, relevant.views3d[0], 1.0);
      canvas.width = 0; canvas.height = 0;
      const cr = parseJSON(await claude(
        [
          { type: "image", source: { type: "base64", media_type: "image/jpeg", data: b64 } },
          { type: "text", text: legendCtx + "\n\nThis is a 3D exterior rendering. Look specifically for:\n1. SOFFITS — underside of overhangs, canopies, covered walkways\n2. RETURNS — where panel wraps around building corners\n3. BUMP-OUTS — wall projections creating additional surfaces\n4. HIDDEN ELEVATIONS — faces not clearly visible in flat drawings\n\nFlag anything that may have been missed. Return ONLY JSON: {\"warnings\":[\"specific items with location\"],\"notes\":\"overall description\"}" },
        ],
        "Review 3D exterior renderings for missed panel siding areas."
      ));
      if (cr && cr.warnings && cr.warnings.length) {
        cr.warnings.forEach(w => jobLog(job, "3D: " + w, "warn"));
      } else {
        jobLog(job, "3D check complete — no additional items flagged", "ok");
      }
    }

    // ── STEP 7: Save evidence PDF ─────────────────────────────────────────────
    job.progress = { label: "Saving evidence PDF...", pct: 97 };
    jobLog(job, "Saving evidence PDF...", "info");
    const pdfBytes = await outputPdf.save();
    const evidencePdfPath = "/tmp/evidence_" + job.id + ".pdf";
    fs.writeFileSync(evidencePdfPath, Buffer.from(pdfBytes));
    job.evidencePdfPath = evidencePdfPath;
    jobLog(job, "Evidence PDF ready — " + outputPdf.getPageCount() + " pages", "ok");

    job.status = "done";
    job.progress = { label: "Complete", pct: 100 };
    jobLog(job, "Done — " + job.takeoffData.length + " elevations analyzed — Excel + PDF ready", "success");

  } catch (err) {
    job.status = "error";
    job.error = err.message;
    jobLog(job, "Error: " + err.message, "error");
  }
}

// ─── Endpoints ────────────────────────────────────────────────────────────────
app.post("/analyze", upload.single("pdf"), (req, res) => {
  const jobId = Date.now().toString();
  const job = createJob(jobId);
  runAnalysis(job, req.file.path, req.file.originalname);
  res.json({ jobId });
});

app.get("/status/:jobId", (req, res) => {
  const job = jobs[req.params.jobId];
  if (!job) return res.status(404).json({ error: "Job not found" });
  res.json({
    status: job.status, phase: job.phase, progress: job.progress,
    log: job.log, legend: job.legend, takeoffData: job.takeoffData,
    soffitNotes: job.soffitNotes, evidenceReady: !!job.evidencePdfPath, error: job.error,
  });
});

app.get("/evidence-pdf/:jobId", (req, res) => {
  const job = jobs[req.params.jobId];
  if (!job || !job.evidencePdfPath || !fs.existsSync(job.evidencePdfPath)) {
    return res.status(404).json({ error: "Evidence PDF not ready yet." });
  }
  res.setHeader("Content-Type", "application/pdf");
  res.setHeader("Content-Disposition", "attachment; filename=BPS_Takeoff_Evidence.pdf");
  fs.createReadStream(job.evidencePdfPath).pipe(res);
});

app.get("/health", (req, res) => res.json({ status: "ok" }));

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log("BPS Estimator backend running on port " + PORT));
