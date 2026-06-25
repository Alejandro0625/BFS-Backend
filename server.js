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

// Job store — holds state for each analysis job
const jobs = {};

function createJob(id) {
  jobs[id] = {
    id,
    status: "running", // running | done | error
    log: [],
    progress: { label: "", pct: 0 },
    phase: "filtering",
    takeoffData: [],
    legend: [],
    soffitNotes: [],
    evidencePdfPath: null,
    error: null,
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
  const b64 = canvas.toDataURL("image/jpeg", 0.80).split(",")[1];
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

// ─── Background analysis function ────────────────────────────────────────────
async function runAnalysis(job, pdfPath, originalName) {
  try {
    jobLog(job, "Loading PDF...", "info");
    const pdfDoc = await getDocument({ url: "file://" + pdfPath }).promise;
    const total = pdfDoc.numPages;
    jobLog(job, "Pages loaded: " + total + " — " + originalName, "ok");

    const outputPdf = await PDFDocument.create();
    await outputPdf.embedFont(StandardFonts.HelveticaBold);

    // STEP 1: Free text extraction
    job.phase = "filtering";
    job.progress = { label: "Reading title blocks", pct: 2 };
    jobLog(job, "Reading all page title blocks (free)...", "info");

    const pageIndex = [];
    for (let p = 1; p <= total; p++) {
      job.progress = { label: "Reading title blocks " + p + "/" + total, pct: Math.round((p / total) * 12) };
      const text = await getPageText(pdfDoc, p);
      if (text.trim().length > 10) pageIndex.push({ page: p, text: text.slice(0, 400) });
    }

    jobLog(job, "Classifying " + pageIndex.length + " pages with 1 API call...", "info");

    // STEP 2: One Claude call to classify
    const indexSummary = pageIndex.map(p => "Page " + p.page + ": " + p.text.replace(/\s+/g, " ").trim()).join("\n");
    const classifyResult = parseJSON(await claude(
      [{ type: "text", text: "You are a commercial panel siding estimator. Here is text from every page title block of a " + total + "-page architectural blueprint set.\n\nIdentify pages containing:\n- EXTERIOR ELEVATIONS: outside building faces with panel cladding (words: elevation, facade, exterior, enlarged elevation)\n- RETURN ELEVATIONS: corner/return details (words: return, corner return, balcony return)\n- MATERIAL LEGEND or FINISH SCHEDULE\n- FLOOR PLANS: only needed for soffit/return locations\n- 3D VIEWS: exterior renderings\n\nWe ONLY care about: ACM, fiber cement, Nichiha, aluminum panels, perforated metal, soffits, returns.\nIGNORE: structural, mechanical, plumbing, electrical, interior, sections, civil, landscape, roofing.\n\nPAGE DATA:\n" + indexSummary + "\n\nReturn ONLY JSON: {\"exteriorElevations\":[page numbers],\"returnElevations\":[page numbers],\"materialLegend\":[page numbers],\"floorPlans\":[page numbers],\"views3d\":[page numbers]}" }],
      "Classify architectural blueprint pages for exterior panel siding estimation. Return ONLY valid JSON."
    ));

    const relevant = {
      exteriorElevations: (classifyResult && classifyResult.exteriorElevations) || [],
      returnElevations: (classifyResult && classifyResult.returnElevations) || [],
      materialLegend: (classifyResult && classifyResult.materialLegend) || [],
      floorPlans: (classifyResult && classifyResult.floorPlans) || [],
      views3d: (classifyResult && classifyResult.views3d) || [],
    };

    jobLog(job, "Found: " + relevant.exteriorElevations.length + " elevations | " + relevant.returnElevations.length + " returns | " + relevant.materialLegend.length + " legend | " + relevant.floorPlans.length + " floor plans", "ok");

    if (!relevant.exteriorElevations.length && !relevant.returnElevations.length) {
      throw new Error("No exterior elevation pages found.");
    }

    // STEP 3: Read legend
    job.phase = "legend";
    job.progress = { label: "Reading legend", pct: 15 };
    jobLog(job, "Reading material legend...", "info");

    let legend = [];
    const legendPages = relevant.materialLegend.length ? relevant.materialLegend : relevant.exteriorElevations.slice(0, 2);
    for (const p of legendPages.slice(0, 3)) {
      const { canvas, b64 } = await renderPageOnce(pdfDoc, p, 1.5);
      canvas.width = 0; canvas.height = 0;
      const raw = await claude(
        [
          { type: "image", source: { type: "base64", media_type: "image/jpeg", data: b64 } },
          { type: "text", text: "Find the EXTERIOR BUILDING MATERIALS LEGEND or FINISH SCHEDULE. Extract ONLY panel cladding materials: ACM, MCM, fiber cement panels/planks, Nichiha, aluminum wall panels, perforated metal, soffit panels, returns/trim. IGNORE: brick, masonry, stone, EIFS, stucco, concrete, glass, curtainwall, roofing, vapor barriers. Return ONLY JSON: {\"projectName\":\"if visible\",\"materials\":[{\"id\":\"e.g. 1\",\"name\":\"full name\",\"category\":\"ACM Panel|Fiber Cement Panel|Fiber Cement Plank|Nichiha Panel|Aluminum Wall Panel|Perforated Metal Panel|Soffit Panel|Return/Trim\",\"color\":\"if noted\",\"notes\":\"\"}]}" },
        ],
        "Extract exterior panel material legends. Return ONLY valid JSON."
      );
      const parsed = parseJSON(raw);
      if (parsed && parsed.materials && parsed.materials.length) {
        legend = parsed.materials;
        jobLog(job, "Found " + legend.length + " materials: " + legend.map(m => m.id + " — " + m.name).join(", "), "ok");
        break;
      }
    }

    if (!legend.length) jobLog(job, "No legend found — will identify from callouts", "warn");
    job.legend = legend;

    // STEP 4: Floor plans for soffits
    let soffitNotes = [];
    if (relevant.floorPlans.length) {
      jobLog(job, "Checking floor plans for soffits and returns...", "info");
      for (const p of relevant.floorPlans.slice(0, 2)) {
        const { canvas, b64 } = await renderPageOnce(pdfDoc, p, 1.0);
        canvas.width = 0; canvas.height = 0;
        const raw = await claude(
          [
            { type: "image", source: { type: "base64", media_type: "image/jpeg", data: b64 } },
            { type: "text", text: "Floor plan — identify ALL soffit locations (canopies, overhangs, covered areas) and return locations (corner wraps). Return ONLY JSON: {\"soffits\":[{\"location\":\"desc\",\"width\":\"ft\",\"depth\":\"ft\"}],\"returns\":[{\"location\":\"desc\",\"height\":\"ft\",\"depth\":\"ft\"}]}" },
          ],
          "Identify soffit and return locations from floor plans. Return ONLY valid JSON."
        );
        const parsed = parseJSON(raw);
        if (parsed && ((parsed.soffits && parsed.soffits.length) || (parsed.returns && parsed.returns.length))) {
          soffitNotes.push(parsed);
          jobLog(job, "Floor plan p." + p + ": " + (parsed.soffits || []).length + " soffits, " + (parsed.returns || []).length + " returns", "ok");
        }
      }
    }
    job.soffitNotes = soffitNotes;

    // STEP 5: Analyze elevations
    job.phase = "analyzing";
    const elevPages = [
      ...relevant.exteriorElevations.map(p => ({ p, type: "elevation" })),
      ...relevant.returnElevations.map(p => ({ p, type: "return" })),
    ];

    jobLog(job, "Analyzing " + elevPages.length + " pages...", "info");

    const legendCtx = legend.length ? "PANEL MATERIAL LEGEND: " + JSON.stringify(legend) : "Identify panel materials from callouts.";
    const soffitCtx = soffitNotes.length ? "\nSOFFIT/RETURN NOTES: " + JSON.stringify(soffitNotes) : "";

    for (let i = 0; i < elevPages.length; i++) {
      const { p, type } = elevPages[i];
      job.progress = { label: "Page " + (i + 1) + " of " + elevPages.length, pct: 20 + Math.round((i / elevPages.length) * 72) };

      const { canvas, b64 } = await renderPageOnce(pdfDoc, p, 1.5);

      const prompt = "You are a senior commercial PANEL SIDING estimator.\n\n" + legendCtx + soffitCtx + "\n\nPage " + p + " — " + type + ".\n\nTAKEOFF PROCESS:\n1. Read drawing TITLE and SHEET REF\n2. Read the SCALE\n3. For each panel material zone: identify material, GROSS area using scale, subtract ALL openings (windows/doors/louvers), NET = Gross minus Openings\n4. SOFFITS: underside of overhangs — width x depth = SF\n5. RETURNS: corner wraps — height x depth = SF\n6. BUMP-OUTS: separate zones\n7. IGNORE: brick, masonry, stone, EIFS, glass, curtainwall, roofing, vapor barriers\n8. BUILDING SECTION or WALL SECTION — return 0 zones\n\nReturn ONLY JSON:\n{\"pageNumber\":" + p + ",\"elevations\":[{\"title\":\"\",\"sheetRef\":\"\",\"scale\":\"\",\"building\":\"\",\"direction\":\"\",\"zones\":[{\"materialId\":\"\",\"materialName\":\"\",\"category\":\"\",\"description\":\"\",\"grossArea\":0,\"totalOpeningArea\":0,\"netArea\":0}],\"flags\":[]}]}";

      const raw = await claude(
        [
          { type: "image", source: { type: "base64", media_type: "image/jpeg", data: b64 } },
          { type: "text", text: prompt },
        ],
        "Senior commercial panel siding estimator. Focus ONLY on panel materials. Return ONLY valid JSON."
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
        jobLog(job, "  Page " + p + ": could not read", "warn");
      }

      // Free canvas immediately
      canvas.width = 0; canvas.height = 0;
    }

    // STEP 6: 3D cross-reference
    if (relevant.views3d.length) {
      job.progress = { label: "3D cross-reference", pct: 94 };
      jobLog(job, "Cross-checking 3D views...", "info");
      const { canvas, b64 } = await renderPageOnce(pdfDoc, relevant.views3d[0], 1.0);
      canvas.width = 0; canvas.height = 0;
      const cr = parseJSON(await claude(
        [
          { type: "image", source: { type: "base64", media_type: "image/jpeg", data: b64 } },
          { type: "text", text: legendCtx + "\n\n3D exterior view. Look for SOFFITS, RETURNS, BUMP-OUTS, HIDDEN ELEVATIONS. Return ONLY JSON: {\"warnings\":[\"items with location\"],\"notes\":\"description\"}" },
        ],
        "Review 3D exterior renderings for missed panel areas."
      ));
      if (cr && cr.warnings && cr.warnings.length) {
        cr.warnings.forEach(w => jobLog(job, "3D: " + w, "warn"));
      } else {
        jobLog(job, "3D check complete", "ok");
      }
    }

    // STEP 7: Save evidence PDF
    job.progress = { label: "Saving evidence PDF...", pct: 97 };
    jobLog(job, "Saving evidence PDF...", "info");
    const pdfBytes = await outputPdf.save();
    const evidencePdfPath = "/tmp/evidence_" + job.id + ".pdf";
    fs.writeFileSync(evidencePdfPath, Buffer.from(pdfBytes));
    job.evidencePdfPath = evidencePdfPath;
    jobLog(job, "Evidence PDF ready — " + outputPdf.getPageCount() + " pages", "ok");

    job.status = "done";
    job.progress = { label: "Complete", pct: 100 };
    jobLog(job, "Done — " + job.takeoffData.length + " elevations — Excel + PDF ready", "success");

  } catch (err) {
    job.status = "error";
    job.error = err.message;
    jobLog(job, "Error: " + err.message, "error");
  }
}

// ─── Start analysis endpoint ──────────────────────────────────────────────────
app.post("/analyze", upload.single("pdf"), (req, res) => {
  const jobId = Date.now().toString();
  const job = createJob(jobId);

  // Start analysis in background — don't await
  runAnalysis(job, req.file.path, req.file.originalname);

  // Return job ID immediately
  res.json({ jobId });
});

// ─── Poll for status ──────────────────────────────────────────────────────────
app.get("/status/:jobId", (req, res) => {
  const job = jobs[req.params.jobId];
  if (!job) return res.status(404).json({ error: "Job not found" });

  res.json({
    status: job.status,
    phase: job.phase,
    progress: job.progress,
    log: job.log,
    legend: job.legend,
    takeoffData: job.takeoffData,
    soffitNotes: job.soffitNotes,
    evidenceReady: !!job.evidencePdfPath,
    error: job.error,
  });
});

// ─── Download evidence PDF ────────────────────────────────────────────────────
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
