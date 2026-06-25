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

// Find the drawing index page using free text scan first
let indexPageNum = null;
for (let p = 1; p <= Math.min(10, total); p++) {
  const text = (await getPageText(pdfDoc, p)).toLowerCase();
  if (
    text.includes("drawing index") || text.includes("sheet index") ||
    text.includes("drawing list") || text.includes("sheet list") ||
    text.includes("drawing schedule") || text.includes("index of drawings") ||
    text.includes("list of drawings") || text.includes("table of contents")
  ) {
    indexPageNum = p;
    jobLog(job, "Found drawing index on page " + p, "ok");
    break;
  }
}

// If not found by keyword, default to page 2
if (!indexPageNum) {
  indexPageNum = 2;
  jobLog(job, "Drawing index keyword not found — trying page 2", "warn");
}

// Render ONLY that one page at high resolution
jobLog(job, "Rendering drawing index page " + indexPageNum + " at high resolution...", "info");
const { canvas: idxCanvas, b64: idxB64 } = await renderPageOnce(pdfDoc, indexPageNum, 2.5);
idxCanvas.width = 0; idxCanvas.height = 0;
const indexPageImages = [{ type: "image", source: { type: "base64", media_type: "image/jpeg", data: idxB64 } }];

    // Ask Claude to read the drawing index and extract ALL sheet names
    jobLog(job, "Extracting sheet list from drawing index...", "info");
    const sheetListResult = parseJSON(await claude(
      [
        ...indexPageImages,
        { type: "text", text: "These are the first pages of a commercial architectural blueprint set with " + total + " total pages.\n\nFind the DRAWING INDEX, SHEET INDEX, DRAWING LIST, or TABLE OF CONTENTS — it is a table listing every drawing with a sheet number and sheet name/description.\n\nExtract the COMPLETE list of every sheet. For each sheet include:\n- The sheet number exactly as shown (e.g. A-200, A1.1, E-101, etc.)\n- The sheet name/description exactly as shown\n\nAlso look for sheet numbers that indicate exterior elevations, return elevations, material legends, floor plans, and 3D views. These are the ones we need for a panel siding takeoff.\n\nReturn ONLY JSON:\n{\n  \"sheets\": [\n    {\"number\": \"A-200\", \"name\": \"Building 1 Overall Exterior Elevations\", \"type\": \"EXTERIOR_ELEVATION|RETURN_ELEVATION|MATERIAL_LEGEND|FLOOR_PLAN|3D_VIEW|IRRELEVANT\"}\n  ],\n  \"notes\": \"any notes about the drawing set\"\n}" }
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

    // ── STEP 2: Map sheet numbers to PDF page numbers using text scan ─────────
    job.progress = { label: "Mapping sheets to pages", pct: 15 };
    jobLog(job, "Scanning all pages to find which PDF page = which sheet number...", "info");

    // Extract text from every page title block (free, no API)
    const pageSheetMap = {}; // pageNum -> sheet number
    for (let p = 1; p <= total; p++) {
      job.progress = { label: "Scanning page " + p + " of " + total, pct: 15 + Math.round((p / total) * 10) };
      const text = await getPageText(pdfDoc, p);
      if (text.trim().length > 5) {
        // Look for any sheet number pattern in the text
        const matches = text.match(/\b[A-Z]{0,3}-?\d{1,4}(?:\.\d{1,2})?\b/g) || [];
        // Also look for common sheet number formats
        const sheetMatches = text.match(/\b(?:A|S|M|P|E|C|L|I|FP|EX|EL|ELEV|ARCH)-?\d{1,4}(?:\.\d{1,2})?\b/gi) || [];
        const allMatches = [...new Set([...matches, ...sheetMatches])];
        if (allMatches.length > 0) {
          pageSheetMap[p] = { text: text.slice(0, 300), matches: allMatches };
        }
      }
    }

    // Now match the sheet list from the index to actual PDF pages
    const relevant = {
      exteriorElevations: [],
      returnElevations: [],
      materialLegend: [],
      floorPlans: [],
      views3d: [],
    };

    if (sheetListResult && sheetListResult.sheets) {
      const relevantSheets = sheetListResult.sheets.filter(s => s.type !== "IRRELEVANT");

      for (const sheet of relevantSheets) {
        const sheetNum = sheet.number.toUpperCase().replace(/\s+/g, "");
        // Find which PDF page contains this sheet number
        let foundPage = null;
        for (const [pageNum, data] of Object.entries(pageSheetMap)) {
          const pageMatches = data.matches.map(m => m.toUpperCase().replace(/\s+/g, ""));
          if (pageMatches.some(m => m === sheetNum || m.includes(sheetNum) || sheetNum.includes(m))) {
            foundPage = parseInt(pageNum);
            break;
          }
        }

        if (foundPage) {
          if (sheet.type === "EXTERIOR_ELEVATION") relevant.exteriorElevations.push(foundPage);
          else if (sheet.type === "RETURN_ELEVATION") relevant.returnElevations.push(foundPage);
          else if (sheet.type === "MATERIAL_LEGEND") relevant.materialLegend.push(foundPage);
          else if (sheet.type === "FLOOR_PLAN") relevant.floorPlans.push(foundPage);
          else if (sheet.type === "3D_VIEW") relevant.views3d.push(foundPage);
        } else {
          jobLog(job, "Could not find page for sheet " + sheet.number + " — " + sheet.name, "warn");
        }
      }
    }

    // If index reading failed or found nothing, fall back to text-based classification
    if (!relevant.exteriorElevations.length && !relevant.returnElevations.length) {
      jobLog(job, "Index matching found no elevations — using AI text classification as fallback...", "warn");
      const pageIndex = Object.entries(pageSheetMap).map(([p, data]) => "Page " + p + ": " + data.text.replace(/\s+/g, " ").trim());
      const indexSummary = pageIndex.join("\n");

      const classifyResult = parseJSON(await claude(
        [{ type: "text", text: "You are a commercial panel siding estimator. Here is text from every page of a " + total + "-page architectural blueprint set.\n\nIdentify which PDF page numbers contain:\n- EXTERIOR ELEVATIONS: outside building faces (words: elevation, facade, exterior)\n- RETURN ELEVATIONS: corner/return details (words: return, corner return, balcony return)\n- MATERIAL LEGEND or FINISH SCHEDULE\n- FLOOR PLANS\n- 3D VIEWS or RENDERINGS\n\nWe ONLY care about: ACM, fiber cement, Nichiha, aluminum panels, perforated metal, soffits, returns.\nIGNORE: structural, mechanical, electrical, interior, sections, civil, roofing.\n\nPAGE DATA:\n" + indexSummary + "\n\nReturn ONLY JSON: {\"exteriorElevations\":[page numbers],\"returnElevations\":[page numbers],\"materialLegend\":[page numbers],\"floorPlans\":[page numbers],\"views3d\":[page numbers]}" }],
        "Classify architectural blueprint pages for exterior panel siding estimation. Return ONLY valid JSON."
      ));

      if (classifyResult) {
        Object.keys(relevant).forEach(k => { if (classifyResult[k]) relevant[k].push(...classifyResult[k]); });
      }
    }

    // Remove duplicates
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
