import express from "express";
import multer from "multer";
import cors from "cors";
import fetch from "node-fetch";
import { createCanvas } from "canvas";
import pkg from "pdfjs-dist/legacy/build/pdf.js"; const { getDocument, GlobalWorkerOptions } = pkg;

GlobalWorkerOptions.workerSrc = "";

const app = express();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 500 * 1024 * 1024 } });

app.use(cors());
app.use(express.json({ limit: "10mb" }));

const API_KEY = process.env.ANTHROPIC_API_KEY;

async function renderPage(pdfDoc, pageNum, scale) {
  const page = await pdfDoc.getPage(pageNum);
  const viewport = page.getViewport({ scale });
  const canvas = createCanvas(Math.floor(viewport.width), Math.floor(viewport.height));
  const ctx = canvas.getContext("2d");
  await page.render({ canvasContext: ctx, viewport }).promise;
  return canvas.toDataURL("image/jpeg", 0.75).split(",")[1];
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
      model: "claude-sonnet-4-20250514",
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
  res.write(`data: ${JSON.stringify(data)}\n\n`);
}

app.post("/analyze", upload.single("pdf"), async (req, res) => {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders();

  try {
    send(res, { type: "log", msg: "Loading PDF...", level: "info" });

    const pdfData = new Uint8Array(req.file.buffer);
    const pdfDoc = await getDocument({ data: pdfData }).promise;
    const total = pdfDoc.numPages;

    send(res, { type: "log", msg: `✓ ${total} pages loaded — ${req.file.originalname}`, level: "ok" });
    send(res, { type: "phase", phase: "filtering" });
    send(res, { type: "log", msg: "Scanning pages to find elevations, floor plans, and legend...", level: "info" });

    const relevant = { floorPlans: [], exteriorElevations: [], returnElevations: [], materialLegend: [], views3d: [], enlargedDetails: [] };

    for (let p = 1; p <= total; p++) {
      send(res, { type: "progress", label: `Scanning page ${p} of ${total}`, pct: Math.round((p / total) * 30) });

      const b64 = await renderPage(pdfDoc, p, 0.12);
      const result = parseJSON(await claude(
        [
          { type: "image", source: { type: "base64", media_type: "image/jpeg", data: b64 } },
          { type: "text", text: `Page ${p} of a commercial architectural blueprint. Classify it. Return ONLY JSON: {"types":["FLOOR_PLAN"|"EXTERIOR_ELEVATION"|"RETURN_ELEVATION"|"MATERIAL_LEGEND"|"3D_VIEW"|"ENLARGED_DETAIL"|"IRRELEVANT"]}` },
        ],
        "Classify architectural blueprint pages. Return ONLY valid JSON."
      ));

      if (result?.types) {
        const t = result.types;
        if (t.includes("FLOOR_PLAN")) relevant.floorPlans.push(p);
        if (t.includes("EXTERIOR_ELEVATION")) relevant.exteriorElevations.push(p);
        if (t.includes("RETURN_ELEVATION")) relevant.returnElevations.push(p);
        if (t.includes("MATERIAL_LEGEND")) relevant.materialLegend.push(p);
        if (t.includes("3D_VIEW")) relevant.views3d.push(p);
        if (t.includes("ENLARGED_DETAIL")) relevant.enlargedDetails.push(p);
        if (!t.includes("IRRELEVANT")) {
          send(res, { type: "log", msg: `Page ${p}: ${t.join(", ")}`, level: "dim" });
        }
      }
    }

    send(res, { type: "log", msg: `✓ Found: ${relevant.materialLegend.length} legend | ${relevant.exteriorElevations.length} elevations | ${relevant.returnElevations.length} returns | ${relevant.views3d.length} 3D views`, level: "ok" });

    if (!relevant.exteriorElevations.length && !relevant.returnElevations.length) {
      throw new Error("No exterior elevation pages found in this PDF.");
    }

    send(res, { type: "phase", phase: "legend" });
    send(res, { type: "log", msg: "Reading material legend...", level: "info" });
    send(res, { type: "progress", label: "Reading legend", pct: 35 });

    let legend = [];
    const legendPages = relevant.materialLegend.length ? relevant.materialLegend : relevant.exteriorElevations.slice(0, 2);

    for (const p of legendPages.slice(0, 3)) {
      const b64 = await renderPage(pdfDoc, p, 1.5);
      const raw = await claude(
        [
          { type: "image", source: { type: "base64", media_type: "image/jpeg", data: b64 } },
          { type: "text", text: `Find the EXTERIOR BUILDING MATERIALS LEGEND. Extract every material. Return ONLY JSON: {"projectName":"if visible","materials":[{"id":"e.g. ACM-1","name":"full name","category":"ACM Panel|Fiber Cement Panel|Fiber Cement Plank|Soffit|Return/Trim|Aluminum Panel|Nichiha|Other","color":"if noted","notes":"spec notes"}]}` },
        ],
        "Extract exterior material legends from architectural drawings. Return ONLY valid JSON."
      );
      const parsed = parseJSON(raw);
      if (parsed?.materials?.length) {
        legend = parsed.materials;
        send(res, { type: "log", msg: `✓ Found ${legend.length} materials: ${legend.map((m) => m.id).join(", ")}`, level: "ok" });
        send(res, { type: "legend", legend });
        break;
      }
    }

    if (!legend.length) {
      send(res, { type: "log", msg: "⚠ No legend found — will identify materials from drawing labels", level: "warn" });
    }

    send(res, { type: "phase", phase: "analyzing" });
    const elevPages = [
      ...relevant.exteriorElevations.map((p) => ({ p, type: "elevation" })),
      ...relevant.returnElevations.map((p) => ({ p, type: "return" })),
      ...relevant.enlargedDetails.map((p) => ({ p, type: "detail" })),
    ];

    send(res, { type: "log", msg: `Analyzing ${elevPages.length} elevation pages...`, level: "info" });
    const legendCtx = legend.length ? `MATERIAL LEGEND: ${JSON.stringify(legend)}` : "Identify materials from callouts on the drawing.";
    const takeoffData = [];

    for (let i = 0; i < elevPages.length; i++) {
      const { p, type } = elevPages[i];
      send(res, { type: "progress", label: `Analyzing elevation ${i + 1} of ${elevPages.length}`, pct: 40 + Math.round((i / elevPages.length) * 50) });

      const b64 = await renderPage(pdfDoc, p, 1.5);
      const prompt = `${legendCtx}

Page ${p} — ${type}. For EVERY elevation drawing on this page:
1. Read the title (e.g. "Building 1 South Elevation")
2. Read the sheet reference (e.g. "1/A-201")
3. Read the SCALE (e.g. 1/8"=1'-0")
4. For each material zone: identify material, calculate gross area using scale, subtract openings, get net SF
5. Flag soffits and returns separately
6. Use scale bar if no dimensions shown

Return ONLY JSON:
{"pageNumber":${p},"elevations":[{"title":"","sheetRef":"","scale":"","building":"","direction":"","zones":[{"materialId":"","materialName":"","category":"","description":"","grossArea":0,"totalOpeningArea":0,"netArea":0}],"flags":[]}]}`;

      const raw = await claude(
        [
          { type: "image", source: { type: "base64", media_type: "image/jpeg", data: b64 } },
          { type: "text", text: prompt },
        ],
        "You are a senior commercial siding estimator doing material takeoffs. Be precise. Return ONLY valid JSON."
      );

      const parsed = parseJSON(raw);
      if (parsed?.elevations?.length) {
        takeoffData.push(...parsed.elevations);
        parsed.elevations.forEach((e) => {
          const sf = e.zones?.reduce((s, z) => s + (z.netArea || 0), 0) || 0;
          send(res, { type: "log", msg: `✓ ${e.title} — ${e.zones?.length || 0} materials, ${Math.round(sf)} SF`, level: "ok" });
          e.flags?.forEach((f) => send(res, { type: "log", msg: `  ⚠ ${f}`, level: "warn" }));
        });
        send(res, { type: "elevation", data: parsed.elevations });
      } else {
        send(res, { type: "log", msg: `⚠ Page ${p}: could not read — manual review needed`, level: "warn" });
      }
    }

    if (relevant.views3d.length) {
      send(res, { type: "progress", label: "Cross-referencing 3D views", pct: 92 });
      send(res, { type: "log", msg: "Cross-checking 3D views for soffits and returns...", level: "info" });
      const b64 = await renderPage(pdfDoc, relevant.views3d[0], 1.0);
      const cr = parseJSON(await claude(
        [
          { type: "image", source: { type: "base64", media_type: "image/jpeg", data: b64 } },
          { type: "text", text: `${legendCtx}\n\nThis is a 3D exterior view. Are there soffits or returns visible that flat elevations might miss? Return ONLY JSON: {"warnings":["items to double check"],"notes":"what you see"}` },
        ],
        "Review 3D renderings to catch missed soffits and returns."
      ));
      if (cr?.warnings?.length) cr.warnings.forEach((w) => send(res, { type: "log", msg: `⚠ 3D: ${w}`, level: "warn" }));
      else send(res, { type: "log", msg: "✓ 3D cross-reference complete", level: "ok" });
    }

    send(res, { type: "done", takeoffData, legend });
    send(res, { type: "log", msg: `✅ Done — ${takeoffData.length} elevations analyzed`, level: "success" });
    res.end();

  } catch (err) {
    send(res, { type: "error", msg: err.message });
    send(res, { type: "log", msg: `❌ ${err.message}`, level: "error" });
    res.end();
  }
});

app.get("/health", (req, res) => res.json({ status: "ok" }));

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`BPS Estimator backend running on port ${PORT}`));
