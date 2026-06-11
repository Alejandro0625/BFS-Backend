import express from "express";
import multer from "multer";
import cors from "cors";
import fetch from "node-fetch";
import { createCanvas } from "canvas";
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

// Materials we care about — panels only
const PANEL_CATEGORIES = [
  "ACM Panel", "MCM Panel", "Fiber Cement Panel", "Fiber Cement Plank",
  "Nichiha Panel", "Aluminum Wall Panel", "Perforated Metal Panel",
  "Metal Panel", "Soffit Panel", "Return/Trim", "Composite Panel"
];

// Materials to ignore
const IGNORE_MATERIALS = [
  "brick", "masonry", "stone", "cast stone", "eifs", "stucco",
  "concrete", "cmu", "glass", "curtainwall", "storefront",
  "roofing", "shingle", "tile", "wood siding", "vinyl"
];

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
  res.write(`data: ${JSON.stringify(data)}\n\n`);
}

app.post("/analyze", upload.single("pdf"), async (req, res) => {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders();

  try {
    send(res, { type: "log", msg: "Loading PDF...", level: "info" });

    const pdfDoc = await getDocument({ url: `file://${req.file.path}` }).promise;
    const total = pdfDoc.numPages;

    send(res, { type: "log", msg: `✓ ${total} pages loaded — ${req.file.originalname}`, level: "ok" });
    send(res, { type: "phase", phase: "filtering" });
    send(res, { type: "log", msg: "Reading sheet index to identify relevant pages...", level: "info" });

    const relevant = { floorPlans: [], exteriorElevations: [], returnElevations: [], materialLegend: [], views3d: [], enlargedDetails: [] };

    // Read first 5 pages to find sheet index
    const indexImages = [];
    for (let p = 1; p <= Math.min(5, total); p++) {
      send(res, { type: "progress", label: `Reading sheet index page ${p}`, pct: p * 4 });
      const b64 = await renderPage(pdfDoc, p, 0.8);
      indexImages.push({ type: "image", source: { type: "base64", media_type: "image/jpeg", data: b64 } });
    }
    indexImages.push({
      type: "text",
      text: `These are the first pages of a commercial architectural blueprint set with ${total} total pages. 

Find the SHEET INDEX or TABLE OF CONTENTS and identify page numbers for:
- EXTERIOR ELEVATIONS (building facades showing outside walls — sheets typically labeled A-2xx)
- RETURN ELEVATIONS (corner/return details — typically A-2xx)  
- ENLARGED ELEVATIONS (zoomed in elevation details)
- MATERIAL LEGEND or FINISH SCHEDULE (exterior material key)
- FLOOR PLANS (needed only to identify soffit and return locations)
- 3D VIEWS or RENDERINGS of the exterior

IMPORTANT: We are a panel siding contractor. We ONLY care about:
ACM panels, MCM panels, fiber cement panels, fiber cement plank, Nichiha panels, aluminum wall panels, perforated metal panels, soffit panels, returns and trim.

We do NOT want: structural drawings, mechanical, plumbing, electrical, interior elevations, interior plans, sections, civil, landscape, roofing, masonry details.

Exterior elevations are typically labeled A-200, A-201, A-202 etc. in the sheet index.
Return elevations are typically A-220, A-221, A-222 etc.
Material legends are often on the same page as the first elevation sheet.
Look carefully at the sheet index table for these sheet numbers and match them to their PDF page numbers.

Return ONLY JSON: {"floorPlans":[page numbers],"exteriorElevations":[page numbers],"returnElevations":[page numbers],"materialLegend":[page numbers],"views3d":[page numbers],"enlargedDetails":[page numbers]}

    const filterResult = parseJSON(await claude(indexImages, "You are a commercial panel siding estimator reading architectural sheet indexes. Identify only pages relevant to exterior panel cladding. Return ONLY valid JSON."));
    if (filterResult) {
      Object.keys(relevant).forEach(k => { if (filterResult[k]) relevant[k].push(...filterResult[k]); });
    }

    send(res, { type: "log", msg: `✓ Sheet index read: ${relevant.exteriorElevations.length} elevations | ${relevant.returnElevations.length} returns | ${relevant.materialLegend.length} legend | ${relevant.floorPlans.length} floor plans | ${relevant.views3d.length} 3D views`, level: "ok" });

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
          { type: "text", text: `Find the EXTERIOR BUILDING MATERIALS LEGEND or FINISH SCHEDULE on this page.

Extract ONLY panel materials — we are a panel siding contractor. Include:
- ACM / MCM panels
- Fiber cement panels and planks  
- Nichiha panels
- Aluminum wall panels
- Perforated metal panels
- Soffit liner / panels
- Returns and trim

IGNORE: brick, masonry, stone, EIFS, stucco, concrete, glass, curtainwall, roofing.

Return ONLY JSON: {"projectName":"if visible","materials":[{"id":"e.g. 1 or ACM-1","name":"full material name","category":"ACM Panel|Fiber Cement Panel|Fiber Cement Plank|Nichiha Panel|Aluminum Wall Panel|Perforated Metal Panel|Soffit Panel|Return/Trim","color":"color or finish if noted","notes":"any spec notes"}]}` },
        ],
        "You are a commercial panel siding estimator extracting material legends. Return ONLY valid JSON."
      );
      const parsed = parseJSON(raw);
      if (parsed?.materials?.length) {
        legend = parsed.materials;
        send(res, { type: "log", msg: `✓ Found ${legend.length} panel materials: ${legend.map((m) => `${m.id} (${m.name})`).join(", ")}`, level: "ok" });
        send(res, { type: "legend", legend });
        break;
      }
    }

    if (!legend.length) {
      send(res, { type: "log", msg: "⚠ No legend found — will identify panel materials from drawing callouts", level: "warn" });
    }

    // Check floor plans for soffits and returns
    let soffitNotes = [];
    if (relevant.floorPlans.length) {
      send(res, { type: "log", msg: "Checking floor plans for soffit and return locations...", level: "info" });
      for (const p of relevant.floorPlans.slice(0, 3)) {
        const b64 = await renderPage(pdfDoc, p, 1.0);
        const raw = await claude(
          [
            { type: "image", source: { type: "base64", media_type: "image/jpeg", data: b64 } },
            { type: "text", text: `This is a floor plan. Identify ALL soffit locations (overhangs, canopies, covered areas) and return locations (where panel wraps around corners). For each one note the approximate dimensions if visible. Return ONLY JSON: {"soffits":[{"location":"description","width":"dimension if visible","depth":"dimension if visible"}],"returns":[{"location":"description","height":"if visible","depth":"if visible"}]}` },
          ],
          "You are identifying soffit and return locations from architectural floor plans for a panel siding takeoff."
        );
        const parsed = parseJSON(raw);
        if (parsed?.soffits?.length || parsed?.returns?.length) {
          soffitNotes.push(parsed);
          send(res, { type: "log", msg: `✓ Floor plan page ${p}: found ${parsed.soffits?.length || 0} soffit locations, ${parsed.returns?.length || 0} return locations`, level: "ok" });
        }
      }
    }

    // Analyze exterior elevations
    send(res, { type: "phase", phase: "analyzing" });
    const elevPages = [
      ...relevant.exteriorElevations.map((p) => ({ p, type: "elevation" })),
      ...relevant.returnElevations.map((p) => ({ p, type: "return" })),
      ...relevant.enlargedDetails.map((p) => ({ p, type: "detail" })),
    ];

    send(res, { type: "log", msg: `Analyzing ${elevPages.length} elevation pages...`, level: "info" });

    const legendCtx = legend.length
      ? `PANEL MATERIAL LEGEND: ${JSON.stringify(legend)}`
      : "Identify panel materials from callouts and labels on the drawing.";

    const soffitCtx = soffitNotes.length
      ? `SOFFIT AND RETURN NOTES FROM FLOOR PLANS: ${JSON.stringify(soffitNotes)}`
      : "";

    const takeoffData = [];

    for (let i = 0; i < elevPages.length; i++) {
      const { p, type } = elevPages[i];
      send(res, { type: "progress", label: `Analyzing elevation ${i + 1} of ${elevPages.length}`, pct: 35 + Math.round((i / elevPages.length) * 55) });

      const b64 = await renderPage(pdfDoc, p, 1.5);

      const prompt = `You are a senior commercial PANEL SIDING estimator doing a material takeoff.

${legendCtx}
${soffitCtx}

This is page ${p} — type: ${type} elevation drawing.

INSTRUCTIONS:
1. Read the drawing title (e.g. "Building 1 - South Elevation", "Return North 01")
2. Read the sheet reference (e.g. "1/A-201")  
3. Read the SCALE printed on the drawing (e.g. 1/8"=1'-0", 1/4"=1'-0")
4. For EACH panel material zone visible:
   - Identify the material using the legend above
   - Calculate GROSS area: measure width × height using the scale
   - List ALL openings (windows, doors, louvers, penetrations) with dimensions
   - NET area = Gross − Total Openings
5. SOFFITS: measure the underside of any overhangs (width × depth = SF)
6. RETURNS: measure any corner wraps (height × return depth = SF)
7. IGNORE completely: brick, masonry, stone, EIFS, stucco, concrete, glass, curtainwall

IMPORTANT: Use the scale to calculate actual dimensions even if not explicitly labeled.

Return ONLY valid JSON:
{
  "pageNumber": ${p},
  "elevations": [
    {
      "title": "Building 1 - South Elevation",
      "sheetRef": "1/A-201",
      "scale": "1/8\\"=1'-0\\"",
      "building": "Building 1",
      "direction": "South",
      "zones": [
        {
          "materialId": "1",
          "materialName": "Architectural Wall Panel - Wood Tones",
          "category": "ACM Panel",
          "description": "Main wall field - upper floors",
          "grossWidth": 120.5,
          "grossHeight": 12.0,
          "grossArea": 1446,
          "openings": [{"label": "Window", "width": 3, "height": 5, "qty": 20, "area": 300}],
          "totalOpeningArea": 300,
          "netArea": 1146
        },
        {
          "materialId": "soffit",
          "materialName": "Soffit Panel",
          "category": "Soffit Panel",
          "description": "Underside of roof overhang at entry",
          "grossWidth": 24,
          "grossHeight": 6,
          "grossArea": 144,
          "openings": [],
          "totalOpeningArea": 0,
          "netArea": 144
        }
      ],
      "flags": ["any unclear dimensions or items needing manual verification"]
    }
  ]
}`;

      const raw = await claude(
        [
          { type: "image", source: { type: "base64", media_type: "image/jpeg", data: b64 } },
          { type: "text", text: prompt },
        ],
        "You are a senior commercial panel siding estimator performing precise material takeoffs from architectural elevation drawings. Focus ONLY on panel materials. Return ONLY valid JSON."
      );

      const parsed = parseJSON(raw);
      if (parsed?.elevations?.length) {
        // Filter out any non-panel materials that slipped through
        parsed.elevations.forEach(e => {
          e.zones = (e.zones || []).filter(z => {
            const nameLower = (z.materialName || "").toLowerCase();
            return !IGNORE_MATERIALS.some(ig => nameLower.includes(ig));
          });
        });
        takeoffData.push(...parsed.elevations);
        parsed.elevations.forEach((e) => {
          const sf = e.zones?.reduce((s, z) => s + (z.netArea || 0), 0) || 0;
          send(res, { type: "log", msg: `✓ ${e.title} (${e.sheetRef}) — ${e.zones?.length || 0} panel zones, ${Math.round(sf)} SF net`, level: "ok" });
          e.flags?.filter(Boolean).forEach((f) => send(res, { type: "log", msg: `  ⚠ ${f}`, level: "warn" }));
        });
        send(res, { type: "elevation", data: parsed.elevations });
      } else {
        send(res, { type: "log", msg: `⚠ Page ${p}: could not read — manual review needed`, level: "warn" });
      }
    }

    // 3D cross reference for soffits/returns
    if (relevant.views3d.length) {
      send(res, { type: "progress", label: "Cross-referencing 3D views", pct: 93 });
      send(res, { type: "log", msg: "Cross-checking 3D views for missed soffits and returns...", level: "info" });
      const b64 = await renderPage(pdfDoc, relevant.views3d[0], 1.0);
      const cr = parseJSON(await claude(
        [
          { type: "image", source: { type: "base64", media_type: "image/jpeg", data: b64 } },
          { type: "text", text: `${legendCtx}\n\nThis is a 3D exterior rendering. Look specifically for:\n1. SOFFITS — underside of overhangs, canopies, covered walkways\n2. RETURNS — where panel material wraps around building corners\n\nAre there any that may have been missed in the flat elevations? Return ONLY JSON: {"warnings":["specific items to double check with location"],"notes":"overall description of what you see"}` },
        ],
        "Review 3D exterior renderings specifically for missed soffit and return conditions."
      ));
      if (cr?.warnings?.length) cr.warnings.forEach((w) => send(res, { type: "log", msg: `⚠ 3D CHECK: ${w}`, level: "warn" }));
      else send(res, { type: "log", msg: "✓ 3D cross-reference complete — no additional items flagged", level: "ok" });
    }

    send(res, { type: "done", takeoffData, legend, soffitNotes });
    send(res, { type: "progress", label: "Complete", pct: 100 });
    send(res, { type: "log", msg: `✅ Analysis complete — ${takeoffData.length} elevations processed`, level: "success" });
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
