// Learning store — Postgres-backed memory so the system gets better every run.
// Degrades gracefully: if DATABASE_URL isn't set, every function is a safe no-op.
import pg from "pg";
const { Pool } = pg;

const url = process.env.DATABASE_URL;
export const dbEnabled = !!url;
const pool = url
  ? new Pool({ connectionString: url, ssl: url.includes("localhost") ? false : { rejectUnauthorized: false } })
  : null;

export async function initDb() {
  if (!pool) { console.log("No DATABASE_URL — learning store disabled (analysis still works)"); return; }
  try {
    await pool.query(`
      CREATE TABLE IF NOT EXISTS runs (
        id SERIAL PRIMARY KEY,
        project_name TEXT, firm TEXT, pages INT, elevations INT,
        total_sf NUMERIC, scale_source TEXT,
        created_at TIMESTAMPTZ DEFAULT now()
      );
      CREATE TABLE IF NOT EXISTS hatch_library (
        id SERIAL PRIMARY KEY,
        firm TEXT NOT NULL DEFAULT 'global',
        signature TEXT NOT NULL,
        category TEXT, material_name TEXT, material_id TEXT,
        hits INT DEFAULT 1,
        updated_at TIMESTAMPTZ DEFAULT now(),
        UNIQUE(firm, signature)
      );
      CREATE TABLE IF NOT EXISTS firm_profiles (
        firm TEXT PRIMARY KEY,
        data JSONB DEFAULT '{}'::jsonb,
        runs INT DEFAULT 0,
        updated_at TIMESTAMPTZ DEFAULT now()
      );
    `);
    console.log("Learning store ready (Postgres connected) ✓");
  } catch (e) { console.log("DB init failed:", e.message); }
}

export async function recordRun(r) {
  if (!pool) return;
  try {
    await pool.query(
      `INSERT INTO runs(project_name,firm,pages,elevations,total_sf,scale_source) VALUES($1,$2,$3,$4,$5,$6)`,
      [r.projectName||null, r.firm||"global", r.pages||null, r.elevations||null, r.totalSf||null, r.scaleSource||null]
    );
  } catch (e) { console.log("recordRun:", e.message); }
}

export async function learnHatch(h) {
  if (!pool || !h?.signature) return;
  try {
    await pool.query(
      `INSERT INTO hatch_library(firm,signature,category,material_name,material_id)
       VALUES($1,$2,$3,$4,$5)
       ON CONFLICT(firm,signature) DO UPDATE SET
         category=EXCLUDED.category, material_name=EXCLUDED.material_name,
         material_id=EXCLUDED.material_id, hits=hatch_library.hits+1, updated_at=now()`,
      [h.firm||"global", h.signature, h.category||null, h.materialName||null, h.materialId||null]
    );
  } catch (e) { console.log("learnHatch:", e.message); }
}

export async function recallHatches(firm) {
  if (!pool) return {};
  try {
    const { rows } = await pool.query(
      `SELECT signature,category,material_name,material_id,hits
       FROM hatch_library WHERE firm=$1 OR firm='global' ORDER BY hits DESC`,
      [firm||"global"]
    );
    const out = {};
    for (const r of rows) {
      if (!out[r.signature]) out[r.signature] = { category:r.category, materialName:r.material_name, id:r.material_id, hits:r.hits };
    }
    return out;
  } catch (e) { console.log("recallHatches:", e.message); return {}; }
}

export async function stats() {
  if (!pool) return { enabled:false };
  try {
    const a = await pool.query(`SELECT count(*)::int n FROM runs`);
    const b = await pool.query(`SELECT count(*)::int n FROM hatch_library`);
    return { enabled:true, runs:a.rows[0].n, learned_hatches:b.rows[0].n };
  } catch (e) { return { enabled:true, error:e.message }; }
}
