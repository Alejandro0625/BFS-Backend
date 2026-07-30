# -*- coding: utf-8 -*-
"""coverage.py -- BFS COVERAGE LAYER (v1, build spec 2026-07-28).

Read-only over job + pdf. Composes the existing string flags into a typed layer,
reconciles declared-vs-found materials, re-runs the sentinel scans, cross-checks
plan-to-elevation geometry, and writes job["coverageReport"]. Flags and report
ONLY -- no SF computation, reader, emitter, regex, or gate is ever touched here
(two-tier rule: SF is never touched; coverage only reads, composes, and flags).

NAMING HAZARD: this module's filename collides with the PyPI `coverage` test
package. The backend runs from its own directory so this local file shadows
site-packages; app.py imports it explicitly at the call site. Recorded risk,
per spec section 0 -- do NOT pip-install `coverage` into the backend venv.

DoD rule (hard): verdict "verified" is NEVER set while ANY closure input was
unavailable. Check 5 (envelope closure) is not buildable in v1, so the v1
verdict is ALWAYS "unverified" -- by design; the deliverable value is the
reasons list and checks_passing shrinking toward verified as stages land.
"""
import os
import re
import json
import datetime
from collections import defaultdict

import fitz
import callouts
import ocr_text
import titleblock  # label-anchored text-layer sheet ID (+ OCR reader, unused here)
import vector_hatch

SCHEMA_VERSION = 1
NORMS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sanity_norms.json")

# Every flag coverage emits carries this prefix (spec: warning-triangle +
# " COVERAGE: "); sanity-ratio flags use the section 5 variant "COVERAGE RATIO:".
# The non-ASCII warning triangle is required in the FLAG STRINGS themselves to
# match the existing emitter style (evidence PDF filters key on prefixes).
_COV = "⚠ COVERAGE: "
_COV_RATIO = "⚠ COVERAGE RATIO: "

# -- FLAG TAXONOMY (spec section 3) ------------------------------------------
# Ordered substring match, first match wins. This is the parallel typed layer --
# emit sites in app.py are untouched. DECISION (recon OPEN #1): the two
# unprefixed scale flags (app.py:711/713) are classified critical/warn by
# SUBSTRING, compensating for the missing warn-prefix without touching emit
# sites; the evidence PDF's startswith(check-mark) filter (app.py:1454) is
# unaffected. Entry: (code, substrings-tuple, severity, scope)
FLAG_TAXONOMY = [
    ("SCALE_UNREAD",      ("Scale could not be read",),                    "critical", "page"),  # app.py:711 -- no warn prefix
    ("SCALE_ELEV_MISM",   ("Elevation markers imply",),                    "critical", "page"),  # app.py:747
    ("SCALE_DIM_MISM",    ("dimension strings imply",),                    "critical", "page"),  # app.py:765
    ("SCALE_VERIFY",      ("Scale read as",),                              "warn",     "page"),  # app.py:713 -- no warn prefix
    ("SCALE_ELEV_OK",     ("Elevation markers confirm the scale",),        "ok",       "page"),
    ("SCALE_DIM_OK",      ("dimension strings confirm the scale",),        "ok",       "page"),
    ("VIEW_SCALE_OK",     ("at their view's own printed scale",),          "ok",       "page"),
    ("VIEW_SCALE_RISK",   ("Multiple view scales",),                       "warn",     "page"),
    ("SOFFIT_JOB",        ("JOB CHECK:",),                                 "critical", "job"),   # app.py:954 (lives on takeoffData[-1])
    ("ANNOT_INTEGRITY",   ("MARKUP INTEGRITY:",),                          "critical", "job"),   # app.py:901 (ditto)
    ("SOFFIT_PAGE",       ("often drawn in SECTION views",),               "warn",     "page"),
    ("REWORK",            ("REWORK job",),                                 "warn",     "page"),
    ("TYP_OF",            ("REPETITION:",),                                "warn",     "page"),
    ("MATCH_LINE",        ("MATCH LINE on this sheet",),                   "warn",     "page"),
    ("SANITY",            ("SANITY:",),                                    "warn",     "page"),
    ("FAMILY_NONBFS",     ("NON-BFS MATERIAL",),                           "warn",     "page"),
    ("FAMILY_UNASSIGNED", ("FAMILY UNASSIGNED:",),                         "warn",     "page"),
    ("SF_LABEL_OUTLIER",  ("this sheet's scale implies",),                 "warn",     "page"),  # markup path, app.py:296
    ("AI_SUGGESTIONS",    ("AI wall suggestion",),                         "info",     "page"),
    # em-dash below is functional: must match the emitter string app.py:690
    ("ENGINE_PROV",       ("pattern vectors", "AI suggestion — verify"),   "info",     "page"),
    ("AUTO_TRIM",         ("Suggested trim (auto)",),                      "info",     "page"),
]

# -- SANITY RATIO THRESHOLDS (spec section 5 -- FROZEN in code) ---------------
# DECISION (recon OPEN): thresholds are frozen constants; sanity_norms.json is
# loaded as evidence/context only and its drift is a report note -- norms never
# float thresholds (prevents absorbing a systematic engine error).
FROZEN = {
    "R1_job_total_sf":        (150, 100_000),  # corpus gold/job p5 627 / p95 52,942 / max 93,580
    "R2_page_sf_max":         40_000,          # per-page gold max 37,995 -- 2x scale error quadruples SF past this
    "R2_page_sf_min_nonzero": 50,              # fragment-only read (page p5 = 106)
    "R3_single_poly_sf_max":  20_000,          # wall max 18,108; GFA/roof/merged-flood signature
    "R3_page_mean_poly_max":  5_000,           # per-job mean SF/wall p95 ~1,200
    "R4_dominance_share":     0.85,            # largest-wall share p95 0.72
    "R4_dominance_min_walls": 5,               # keeps legit 1-2 wall jobs quiet
    "R5_soffit_share_max":    0.35,            # ledger soffit share p95 0.18 (n=50 -- recompute norms as ledger grows)
    "R5_unassigned_nonbfs_share_max": 0.25,    # ratio upgrade of app.py:650-655 fixed-SF flags
}

# -- SYNC'd regex copies (money-path regexes are NEVER modified; these are
#    coverage's own read-only re-runs of the same patterns) -------------------
# SYNC WITH app.py:591 -- the codemap key grammar
_CODE_RE = re.compile(r"\b([A-Z]{1,4}-?\d{1,2}[A-Z]?)\b")
# SYNC WITH app.py:577-580 -- the two per-job codemap regexes (the en-dash in
# the character classes is functional: byte-for-byte parity with app.py)
_CM_RE1 = re.compile(r"([A-Z][A-Z\s/&]{6,40})\s*[-–:]\s*\(?([A-Z]{1,4}-?\d{1,2}[A-Z]?)\)?")
_CM_RE2 = re.compile(r"\b([A-Z]{1,4}-?\d{1,2}[A-Z]?)\s*[:=–-]\s*([A-Z][A-Z\s/&]{6,40})")
# Coverage's OWN loose compound-code grammar (spec section 4 -- lives ONLY here;
# callouts._KEY_RE is NOT modified). Covers E3A-FC1 / E2B-TC1 (the documented
# 26-020 gap) PLUS digit-first legend keys (2A / 12C -- 26-133-class elevation
# material legends key materials by number+letter; without this alternative the
# declared side misses them entirely).
_LOOSE_KEY_RE = re.compile(r"^(?:[A-Z]{1,3}\d{0,2}[A-Z]?(?:-[A-Z]{1,4}\d{0,2}[A-Z]?)?|\d{1,2}[A-Z]?)$")
# SYNC WITH app.py:914 -- the TYP-OF emitter regex
_TYP_RE = re.compile(r"TYP(?:ICAL)?\.?\s*(?:OF|X)\s*\(?(\d+)\)?")
# SYNC WITH app.py:682-683 -- REWORK sentinel terms
_REWORK_TERMS = ("REFABRICATE", "REMOVE AND REINSTALL", "REWORK", "LIMITS OF WORK")
# Owner refinement 2026-07-28 -- courtyard/separate-structure vocabulary. Each
# hit is a STRUCTURE_TERM convention: always unresolved in v1 (same semantics
# as TYP -- no auto-resolution signal exists; SF never auto-extended).
_STRUCT_TERMS = ("COURTYARD", "LIGHT WELL", "LIGHTWELL", "AREAWAY", "BREEZEWAY",
                 "POOL HOUSE", "CANOPY", "OVERHANG", "PORTE COCHERE")
_BLDG_RE = re.compile(r"\bBUILDING\s+([B-F])\b")
# GARAGE counts only as a whole-word title-cased/upper-case HEADING (short line)
# -- the word appears in notes on nearly every sheet and would drown the report.
_GARAGE_RE = re.compile(r"\b(GARAGE|Garage)\b")
# Sheet-number grammar for the drawing index / title blocks (A-series + general)
_SHEETNUM_RE = re.compile(r"^[A-Z]{1,2}\d{1,2}\.\d{1,2}[A-Z]?$")
_A_SHEET_RE = re.compile(r"^A[A-Z]?\d{1,2}\.\d{1,2}[A-Z]?$")
# Positive floor/roof-plan title classifier (spec section 8 -- today only a
# negative signal exists at app.py:315)
_PLAN_TITLE_RE = re.compile(r"\b((FIRST|SECOND|THIRD|GROUND|\d+(ST|ND|RD|TH))\s+)?(FLOOR|ROOF)\s+PLAN\b")
# Sheet-index headings (owner refinement 2026-07-28)
_INDEX_HEAD_RE = re.compile(r"(SHEET\s+INDEX|DRAWING\s+INDEX|DRAWING\s+LIST|SHEET\s+LIST|LIST\s+OF\s+DRAWINGS)")
# SYNC WITH app.py:312 -- elevation title regex (compass extraction reuses it)
_ELEV_TITLE_RE = re.compile(r"\b(north|south|east|west|front|rear|left|right|side|partial|overall|exterior|building)\s+[\w\s]{0,18}elevation")
_COMPASS_WORDS = ("north", "south", "east", "west")


# -- family normalizer --------------------------------------------------------
# SYNC WITH app.py:611-655 -- verbatim copies of the nested _fam9/_fam9_outer.
# ASSUMPTION (spec register): duplication acceptable until a shared families.py
# refactor (out of scope v1). Rationale: they are nested inside process() and
# only bound if an auto page ran -- importing app from coverage would be
# circular, and injection can NameError on markup-only jobs.
def _fam9(m):
    u = (m or "").upper()
    if re.match(r"^[AS]-?\d{2,4}\b", u) or u.startswith("GROUP ") or len(u) <= 2:
        return "junk-name"
    if any(k in u for k in ("BRICK", "MASON", "STONE", "EIFS", "STUCCO", "CONCRETE",
                            "GLAZ", "GLASS", "CURTAIN", "STOREFRONT", "GWB", "ROOF MEMBRANE")):
        return "non-bfs"
    if "SOFFIT" in u:
        return "soffit"
    if "ACM" in u or "MCM" in u or "COMPOSITE" in u or "ALUCOBOND" in u:
        return "acm"
    if "NICHIHA" in u or "NCH" in u:
        return "nichiha"
    if "PERF" in u:
        return "perforated"
    if re.match(r"^(FCP|FCL|FCS|FCBB|FSV|FSH|FC)[-\s:]", u):
        return "fc"
    if re.match(r"^(VCS|EL0?\d)[-\s:]", u) or "VINYL" in u:
        return "vinyl"
    if re.match(r"^AP[-\s:]", u):
        return "metal"
    if any(k in u for k in ("SHINGLE", "SHAKE", "CEDAR")):
        return "shingle"
    if any(k in u for k in ("STANDING", "SEAM", "CORRUGAT", "METAL PANEL", "MTL", "ALUM")):
        return "metal"
    if any(k in u for k in ("FC", "FIBER", "HARDIE", "LAP", "SIDING", "CEMENT", "CLAPBOARD", "PLANK")):
        return "fc"
    return "unassigned"


def _fam9_outer(m):
    # READER-generic vocabulary is not a drawing identity -- always unassigned
    u = (m or "").upper()
    if any(k in u for k in ("WALL AREA", "(CONFIRM)", "WALL BAND", "PANEL WALL",
                            "VERTICAL RIB", "RENDERED SIDING", "HATCHED AREA",
                            "LAP / HORIZONTAL", "COLOR FILL", "AI SUGGESTION")):
        return "unassigned"
    return _fam9(m)


# -- small helpers ------------------------------------------------------------
def _now_iso():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pbp(job):
    """polygons_by_page with int keys (JSON round-trips stringify them)."""
    out = {}
    for k, v in (job.get("polygons_by_page") or {}).items():
        try:
            out[int(k)] = v or []
        except Exception:
            continue
    return out


def _counted(polys):
    """Polys that enter zones/totals -- suggestions NEVER count (app.py:604-605)."""
    return [p for p in polys if not p.get("suggest_only") and (p.get("area_sf") or 0) > 0]


def _page_entry(job, page):
    for e in (job.get("takeoffData") or []):
        if e.get("pageNumber") == page:
            return e
    return None


def _add_page_flag(job, report, page, text):
    """Append a coverage flag to a page's takeoffData entry (dedup; page entry
    may be absent -- then the flag is job-scoped instead)."""
    e = _page_entry(job, page)
    if e is None:
        _add_job_flag(job, report, text)
        return
    fl = e.setdefault("flags", [])
    if text not in fl:
        fl.append(text)


def _add_job_flag(job, report, text):
    """Job-scoped flag: into report job_flags AND mirrored onto takeoffData[-1]
    (matching current frontend behavior until React renders coverageReport)."""
    if text not in report["job_flags"]:
        report["job_flags"].append(text)
    td = job.get("takeoffData") or []
    if td:
        fl = td[-1].setdefault("flags", [])
        if text not in fl:
            fl.append(text)


def _texts(doc):
    """One cached text pass over ALL pages (raw + upper) -- every check reads
    from this instead of re-extracting (one extra full pass total, spec s11)."""
    raw, up = {}, {}
    for pi in range(doc.page_count):
        try:
            t = doc[pi].get_text() or ""
        except Exception:
            t = ""
        raw[pi] = t
        up[pi] = t.upper()
    return raw, up


# -- typed flag layer ---------------------------------------------------------
def classify_flags(takeoff_data):
    """[{code, severity, scope, page, text}] via FLAG_TAXONOMY ordered substring
    match (first match wins). Parallel typed layer -- emit sites untouched.
    DECISION (recon OPEN #2): job-level flags (SOFFIT_JOB, ANNOT_INTEGRITY) are
    RE-HOMED to scope "job" (page None) in the typed layer; the original strings
    stay on takeoffData[-1] untouched (known frontend mis-attribution, healed at
    the report layer, not the emitter)."""
    out = []
    for e in (takeoff_data or []):
        page = e.get("pageNumber")
        for f in (e.get("flags") or []):
            code, sev, scope = "UNCLASSIFIED", "info", "page"
            for c, subs, s, sc in FLAG_TAXONOMY:
                if any(sub in f for sub in subs):
                    code, sev, scope = c, s, sc
                    break
            out.append({"code": code, "severity": sev, "scope": scope,
                        "page": None if scope == "job" else page, "text": f})
    return out


# -- check 1: legend reconciliation -------------------------------------------
def _loose_legend_pairs(pg, blocks, allow_digit_first):
    """Coverage's own key->material pairing: callouts._legend_pairs convention
    (key just left of a material block at the same height) but with the loose
    compound grammar, nearest-block-by-gap wins. Digit-first keys are harvested
    only on pages whose text says LEGEND (they are too common as plain numbers
    elsewhere). callouts._KEY_RE / money path untouched."""
    pairs = {}
    try:
        for w in pg.get_text("words"):
            key = (w[4] or "").strip()
            if not _LOOSE_KEY_RE.match(key):
                continue
            # noise guard: require a digit or hyphen (pure-alpha tokens like
            # THE/OF pass the spec grammar but are never legend keys)
            if not (any(ch.isdigit() for ch in key) or "-" in key):
                continue
            if key[0].isdigit() and not allow_digit_first:
                continue
            wy = (w[1] + w[3]) / 2
            best = None
            for b in blocks:
                r = b["rect"]
                if r[0] >= w[2] - 4 and r[0] - w[2] < 180 and r[1] - 8 <= wy <= r[3] + 8:
                    gap = r[0] - w[2]
                    if best is None or gap < best[0]:
                        best = (gap, b)
            if best is not None and key not in pairs:
                pairs[key] = callouts._clean(best[1]["text"])
    except Exception:
        pass
    return pairs


def _loose_schedule_scan(doc, raw, up):
    """Coverage's OWN loose schedule re-scan (spec section 4 source 5): same
    row-clustering idea as callouts.read_schedule_pg but accepts compound codes
    via the loose grammar and accepts >=1 surviving row. The money-path reader
    and its >=2-row gates (callouts.py:235 / app.py:821) are NOT changed -- this
    only makes NONE-by-gap distinguishable from NONE-by-absence in the report."""
    rows_out = {}
    for pi in range(doc.page_count):
        if len(raw.get(pi, "")) < 60:      # mirror app.py:814 textless skip
            continue
        try:
            pg = doc[pi]
            blocks = callouts._blocks(pg)
            legend = _loose_legend_pairs(pg, blocks, allow_digit_first=("LEGEND" in up.get(pi, "")))
            rows = defaultdict(list)
            for w in pg.get_text("words"):
                rows[round(w[1] / 4)].append(w)
            for _, ws in rows.items():
                ws = sorted(ws, key=lambda w: w[0])
                toks = [w[4] for w in ws]
                for i in range(len(toks) - 1):
                    k = toks[i]
                    if not _LOOSE_KEY_RE.match(k):
                        continue
                    if not (any(ch.isdigit() for ch in k) or "-" in k):
                        continue
                    for j in range(i + 1, min(i + 3, len(toks))):
                        if callouts._SCHED_NUM.match(toks[j]):
                            val = float(toks[j].replace(",", ""))
                            unit_ok = (j + 1 < len(toks) and callouts._SCHED_UNIT.match(toks[j + 1]))
                            if val >= 50 and unit_ok:
                                mat = legend.get(k, "")
                                if mat and callouts._LINEARISH_RE.search(mat):
                                    break
                                cur = rows_out.get(k)
                                if cur is None or val > cur["sf"]:
                                    rows_out[k] = {"key": k, "sf": round(val), "material": mat}
                            break
        except Exception:
            continue
    return list(rows_out.values())


def _declared_materials(job, doc, raw, up):
    """(declared, notes). Union of five sources per spec section 4. DECISION:
    _code_map9 is NOT added to /status; the report carries the reconciliation."""
    declared = {}   # dedup key -> entry
    notes = []

    def _put(name, code, source, sf_stated=None):
        key = (code or "").upper() or (name or "").upper()
        if not key:
            return
        cur = declared.get(key)
        if cur is None:
            declared[key] = {"name": (name or "").strip()[:60] or None, "code": code,
                             "source": source, "sf_stated": sf_stated}
        else:
            if cur.get("sf_stated") is None and sf_stated is not None:
                cur["sf_stated"] = sf_stated
            if not cur.get("name") and name:
                cur["name"] = name.strip()[:60]

    # 1) the money-path drawing schedule (exists only when >=2 rows survived)
    try:
        for it in ((job.get("drawingSchedule") or {}).get("items") or []):
            _put(it.get("material"), it.get("key"), "schedule", it.get("sf"))
    except Exception:
        pass
    # 2) job["_code_map9"] -- server-side only, only built if an auto page ran;
    #    when absent, re-run the two codemap regexes over full-doc upper text
    #    ourselves (cheap, deterministic -- SYNC WITH app.py:577-580)
    try:
        cm = job.get("_code_map9")
        if not isinstance(cm, dict) or not cm:
            cm = {}
            if doc is not None:
                for pi in range(doc.page_count):
                    t = up.get(pi, "")
                    for m in _CM_RE1.finditer(t):
                        cm.setdefault(m.group(2), m.group(1).strip())
                    for m in _CM_RE2.finditer(t):
                        cm.setdefault(m.group(1), m.group(2).strip())
        for code, name in cm.items():
            _put(name, code, "codemap")
    except Exception:
        pass
    # 3) per-page legend-pair recompute (never persisted) on text-bearing pages
    #    -- callouts' own strict pairing plus coverage's loose pairing
    try:
        if doc is not None:
            for pi in range(doc.page_count):
                if len(raw.get(pi, "")) < 60:   # mirror app.py:814
                    continue
                pg = doc[pi]
                blocks = callouts._blocks(pg)
                if not blocks:
                    continue
                for k, v in (callouts._legend_pairs(pg, blocks) or {}).items():
                    _put(v, k, "legend_pairs")
                for k, v in _loose_legend_pairs(pg, blocks,
                                                allow_digit_first=("LEGEND" in up.get(pi, ""))).items():
                    _put(v, k, "legend_pairs")
    except Exception:
        pass
    # 4) OCR vocabulary for flattened sets (app.py:839-849 output)
    try:
        for m in (job.get("ocrMaterials") or []):
            _put(m.get("text"), None, "ocr")
    except Exception:
        pass
    # 5) coverage's own loose schedule re-scan
    try:
        if doc is not None:
            loose = _loose_schedule_scan(doc, raw, up)
            if loose and not job.get("drawingSchedule"):
                # rows exist that the money-path reader dropped (its >=2-row
                # gate, or its stricter key grammar) -- NONE-by-gap, not absence
                notes.append("schedule_below_gate")
            for r in loose:
                _put(r.get("material"), r.get("key"), "loose_scan", r.get("sf"))
    except Exception:
        pass
    return list(declared.values()), notes


def _found_materials(job):
    """Found side per-poly via polygons_by_page (DECISION, recon OPEN #4) --
    honoring suggest_only (excluded) and named_by provenance exactly.
    job["legend"] is NOT used as a source (it conflates provenance and carries
    reader-generic names) -- cross-check only."""
    agg = {}
    for page, polys in _pbp(job).items():
        for p in _counted(polys):
            name = str(p.get("material") or p.get("category") or "Unlabeled")
            a = agg.setdefault(name, {"name": name, "family": _fam9_outer(name),
                                      "sf": 0.0, "provenance": None})
            a["sf"] += float(p.get("area_sf") or 0)
            prov = p.get("named_by") or p.get("named_by_tag")
            if prov and not a["provenance"]:
                a["provenance"] = str(prov)
    out = []
    for a in agg.values():
        a["sf"] = round(a["sf"], 1)
        out.append(a)
    out.sort(key=lambda x: -x["sf"])
    return out


def _tokens_of(text, code=None):
    """Code tokens from both key grammars (app.py:591 + loose), with compound
    split on '-' (E3A-FC1 -> {E3A-FC1, E3A, FC1})."""
    toks = set()
    u = (text or "").upper()
    for c in _CODE_RE.findall(u):
        toks.add(c)
    if code:
        toks.add(str(code).upper())
    for c in list(toks):
        if "-" in c:
            for part in c.split("-"):
                if part:
                    toks.add(part)
    return toks


def _reconcile(declared, found):
    """Code-token match first, then family-normalized name match (reader-generic
    found names dropped via the _fam9_outer stop list)."""
    found_tokens = set()
    found_fams = set()
    for f in found:
        found_tokens |= _tokens_of(f.get("name"))
        prov = f.get("provenance") or ""
        if ":" in prov:                       # codemap:FC1 / legend:M5
            found_tokens |= _tokens_of(prov.split(":", 1)[1])
        fam = _fam9_outer(f.get("name"))      # stop list drops reader-generic
        if fam not in ("junk-name", "unassigned"):
            found_fams.add(fam)
    matched, missing = [], []
    for d in declared:
        toks = _tokens_of(d.get("name"), d.get("code"))
        if toks & found_tokens:
            matched.append(d)
            continue
        fam = _fam9(d.get("name") or "")
        if fam not in ("junk-name", "unassigned") and fam in found_fams:
            matched.append(d)
            continue
        missing.append(d)
    declared_tokens = set()
    for d in declared:
        declared_tokens |= _tokens_of(d.get("name"), d.get("code"))
    declared_fams = {_fam9(d.get("name") or "") for d in declared}
    undeclared = []
    for f in found:
        ftoks = _tokens_of(f.get("name"))
        prov = f.get("provenance") or ""
        if ":" in prov:
            ftoks |= _tokens_of(prov.split(":", 1)[1])
        fam = _fam9_outer(f.get("name"))
        if ftoks & declared_tokens:
            continue
        if fam not in ("junk-name", "unassigned") and fam in declared_fams:
            continue
        undeclared.append({"name": f.get("name"), "family": fam, "sf": f.get("sf")})
    return {"matched": matched, "missing": missing, "undeclared": undeclared}


def _check_legend(job, doc, raw, up, report):
    declared, notes = _declared_materials(job, doc, raw, up)
    found = _found_materials(job)
    rec = _reconcile(declared, found)
    report["materials_declared"] = declared
    report["materials_found"] = found
    report["materials_missing"] = [{"name": d.get("name"), "code": d.get("code"),
                                    "sf_stated": d.get("sf_stated")} for d in rec["missing"]]
    report["materials_undeclared"] = rec["undeclared"]
    for d in rec["missing"]:
        sf = d.get("sf_stated")
        # ASSUMPTION (spec register): 100 SF floor for missing-material flags to
        # keep noise down; below that (or no stated SF), report-only.
        if sf is None or sf < 100:
            continue
        # non-BFS families (brick/masonry/glazing) are never BFS scope -- their
        # absence from the takeoff is not a coverage gap (report-only).
        if _fam9(d.get("name") or "") == "non-bfs":
            continue
        nm = d.get("name") or d.get("code") or "?"
        code = (" (%s, %s SF stated)" % (d.get("code"), format(int(sf), ","))) if d.get("code") \
            else (" (%s SF stated)" % format(int(sf), ","))
        _add_job_flag(job, report, _COV + "drawings declare '%s'%s but no matching zone exists"
                      " in this takeoff — confirm measured or excluded before bidding." % (nm, code))
    if not declared:
        # Unavailability: textless set, no ocrMaterials, no pairs, no codemap
        # hits -- absence of evidence never passes.
        return {"status": "unavailable", "notes": notes + ["no declared-side sources readable"]}
    return {"status": "fail" if rec["missing"] else "pass", "notes": notes}


# -- check 2: sanity ratios ---------------------------------------------------
def load_norms(path=NORMS_PATH):
    """Evidence/context only -- thresholds stay FROZEN regardless of norms."""
    try:
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
    except Exception:
        pass
    return None


def _sanity_ratios(job, norms):
    """failed_ratios per FROZEN. Denominators: poly counts from polygons_by_page
    excluding suggest_only (ledger-style); job SF = sum of zone netArea
    (matches app.py:960). Opening ratio and read/sub ratio EXCLUDED by recon
    finding (grossArea==netArea hard-coded, totalOpeningArea always 0 --
    app.py:659-660/1562-1563; new jobs have no sub_total)."""
    failed = []
    td = job.get("takeoffData") or []

    def _norm(key):
        try:
            n = (norms or {}).get(key)
            return n if isinstance(n, dict) else None
        except Exception:
            return None

    total = sum(float(z.get("netArea") or 0) for e in td for z in (e.get("zones") or []))
    lo, hi = FROZEN["R1_job_total_sf"]
    if total > 0 and (total < lo or total > hi):
        failed.append({"ratio": "R1_job_total_sf", "page": None, "value": round(total, 1),
                       "threshold": lo if total < lo else hi, "norm": _norm("R1_job_total_sf")})
    pbp = _pbp(job)
    for e in td:
        page = e.get("pageNumber")
        psf = sum(float(z.get("netArea") or 0) for z in (e.get("zones") or []))
        if psf > FROZEN["R2_page_sf_max"]:
            failed.append({"ratio": "R2_page_sf_max", "page": page, "value": round(psf, 1),
                           "threshold": FROZEN["R2_page_sf_max"], "norm": _norm("R2_page_sf_max")})
        elif 0 < psf < FROZEN["R2_page_sf_min_nonzero"]:
            failed.append({"ratio": "R2_page_sf_min_nonzero", "page": page, "value": round(psf, 1),
                           "threshold": FROZEN["R2_page_sf_min_nonzero"],
                           "norm": _norm("R2_page_sf_min_nonzero")})
        polys = _counted(pbp.get(page) or [])
        if polys:
            sfs = [float(p.get("area_sf") or 0) for p in polys]
            if max(sfs) > FROZEN["R3_single_poly_sf_max"]:
                failed.append({"ratio": "R3_single_poly_sf_max", "page": page,
                               "value": round(max(sfs), 1),
                               "threshold": FROZEN["R3_single_poly_sf_max"],
                               "norm": _norm("R3_single_poly_sf_max")})
            mean = sum(sfs) / len(sfs)
            if mean > FROZEN["R3_page_mean_poly_max"]:
                failed.append({"ratio": "R3_page_mean_poly_max", "page": page,
                               "value": round(mean, 1),
                               "threshold": FROZEN["R3_page_mean_poly_max"],
                               "norm": _norm("R3_page_mean_poly_max")})
    all_sfs = [float(p.get("area_sf") or 0) for polys in pbp.values() for p in _counted(polys)]
    if len(all_sfs) >= FROZEN["R4_dominance_min_walls"] and sum(all_sfs) > 0:
        share = max(all_sfs) / sum(all_sfs)
        if share > FROZEN["R4_dominance_share"]:
            failed.append({"ratio": "R4_dominance_share", "page": None, "value": round(share, 3),
                           "threshold": FROZEN["R4_dominance_share"], "norm": _norm("R4_dominance_share")})
    if total > 0:
        soffit = sum(float(z.get("netArea") or 0) for e in td for z in (e.get("zones") or [])
                     if "soffit" in str(z.get("materialName") or z.get("category") or "").lower()
                     or (z.get("family") == "soffit"))
        if soffit / total > FROZEN["R5_soffit_share_max"]:
            failed.append({"ratio": "R5_soffit_share_max", "page": None,
                           "value": round(soffit / total, 3),
                           "threshold": FROZEN["R5_soffit_share_max"], "norm": _norm("R5_soffit_share_max")})
        una = sum(float(z.get("netArea") or 0) for e in td for z in (e.get("zones") or [])
                  if (z.get("family") or _fam9_outer(str(z.get("materialName") or ""))) in
                  ("unassigned", "non-bfs"))
        if una / total > FROZEN["R5_unassigned_nonbfs_share_max"]:
            failed.append({"ratio": "R5_unassigned_nonbfs_share_max", "page": None,
                           "value": round(una / total, 3),
                           "threshold": FROZEN["R5_unassigned_nonbfs_share_max"],
                           "norm": _norm("R5_unassigned_nonbfs_share_max")})
    return failed


# plain-English ratio flag texts (em-dashes match existing flag style)
_RATIO_TEXT = {
    "R1_job_total_sf": "job total %s SF is outside the plausible range %s-%s SF for one job",
    "R2_page_sf_max": "page total %s SF is above the %s SF single-sheet maximum ever seen in the archive",
    "R2_page_sf_min_nonzero": "page total %s SF reads like a fragment-only page (< %s SF)",
    "R3_single_poly_sf_max": "a single region of %s SF exceeds the %s SF largest-wall ceiling (merged/leaked region signature)",
    "R3_page_mean_poly_max": "mean region size %s SF on this page exceeds %s SF (scale or merge error signature)",
    "R4_dominance_share": "one region carries %s of the whole job's SF (threshold %s) — check for a merged flood",
    "R5_soffit_share_max": "soffit share %s of job SF exceeds %s — verify soffit zones",
    "R5_unassigned_nonbfs_share_max": "unassigned/non-BFS share %s of job SF exceeds %s — name the walls before pricing",
}


def _check_ratios(job, report):
    td = job.get("takeoffData") or []
    if not td:
        return {"status": "unavailable", "notes": ["no takeoffData"]}
    norms = load_norms()
    report["norms_source"] = (norms or {}).get("source") if isinstance(norms, dict) and norms.get("source") \
        else ("sanity_norms.json" if norms else "frozen-defaults")
    failed = _sanity_ratios(job, norms)
    report["failed_ratios"] = failed
    for f in failed:
        tmpl = _RATIO_TEXT.get(f["ratio"], "%s over %s")
        try:
            if f["ratio"] == "R1_job_total_sf":
                txt = tmpl % (format(f["value"], ","),
                              format(FROZEN["R1_job_total_sf"][0], ","),
                              format(FROZEN["R1_job_total_sf"][1], ","))
            else:
                txt = tmpl % (format(f["value"], ","), format(f["threshold"], ","))
        except Exception:
            txt = "%s failed (value %s, threshold %s)" % (f["ratio"], f["value"], f["threshold"])
        flag = _COV_RATIO + txt
        # page-scoped R2/R3 to the offending page; job-scoped R1/R4/R5 to
        # job_flags + takeoffData[-1] (spec section 5)
        if f.get("page"):
            _add_page_flag(job, report, f["page"], flag)
        else:
            _add_job_flag(job, report, flag)
    return {"status": "fail" if failed else "pass", "notes": []}


# -- check 3: sentinel rollup -------------------------------------------------
def _sentinel_scan(doc, raw, up):
    """One text pass over ALL pages re-running the emitters' own patterns.
    {page: [{type, text}]}. Includes the owner-refinement STRUCTURE_TERM
    vocabulary (courtyard / separate-building terms)."""
    hits = defaultdict(list)
    for pi in range(doc.page_count):
        t = up.get(pi, "")
        if not t:
            continue
        page = pi + 1
        m = _TYP_RE.search(t)             # SYNC app.py:914 (one flag per page)
        if m:
            try:
                n = int(m.group(1))
                if 2 <= n <= 30:          # SYNC app.py:916 gate
                    hits[page].append({"type": "TYP_OF", "text": m.group(0).strip()})
            except Exception:
                pass
        if "MATCH LINE" in t:             # SYNC app.py:925
            hits[page].append({"type": "MATCH_LINE", "text": "MATCH LINE"})
        # SYNC app.py:945 -- the job-level soffit/return sentinel condition
        if "SOFFIT" in t or ("RETURN" in t and ("PANEL" in t or "CANOPY" in t or "ACM" in t)):
            hits[page].append({"type": "SOFFIT", "text": "SOFFIT/RETURN"})
        if any(k in t for k in _REWORK_TERMS):   # SYNC app.py:682-683
            hits[page].append({"type": "REWORK", "text": "REWORK terms"})
        # owner refinement: courtyard / separate-structure vocabulary
        for term in _STRUCT_TERMS:
            if term in t:
                hits[page].append({"type": "STRUCTURE_TERM", "text": term})
        for m2 in set(_BLDG_RE.findall(t)):
            hits[page].append({"type": "STRUCTURE_TERM", "text": "BUILDING " + m2})
        # GARAGE: whole-word heading only (short all-caps or Title Case line)
        for line in raw.get(pi, "").splitlines():
            s = line.strip()
            if len(s) <= 40 and _GARAGE_RE.search(s) and (s == s.upper() or s.istitle()):
                hits[page].append({"type": "STRUCTURE_TERM", "text": "GARAGE"})
                break
    return hits


def _check_sentinels(job, doc, raw, up, typed, report):
    if doc is None:
        return {"status": "unavailable", "notes": ["pdf unreadable"]}
    hits = _sentinel_scan(doc, raw, up)
    emitted = {(t["code"], t["page"]) for t in typed}
    soffit_job_emitted = any(t["code"] == "SOFFIT_JOB" for t in typed)
    # SOFFIT resolution -- SYNC app.py:949-951: resolved iff a soffit/return
    # zone exists anywhere in the takeoff
    soffit_zone = any(("soffit" in str(z.get("materialName") or z.get("category") or "").lower())
                      or ("return" in str(z.get("materialName") or z.get("category") or "").lower())
                      for e in (job.get("takeoffData") or []) for z in (e.get("zones") or []))
    unresolved = []
    notes = []
    struct_pages = defaultdict(list)      # term -> pages
    soffit_resolved_hits = 0
    for page in sorted(hits.keys()):
        for h in hits[page]:
            typ, text = h["type"], h["text"]
            if typ == "STRUCTURE_TERM":
                struct_pages[text].append(page)
                continue
            if typ == "TYP_OF":
                was = ("TYP_OF", page) in emitted
            elif typ == "MATCH_LINE":
                was = ("MATCH_LINE", page) in emitted
            elif typ == "SOFFIT":
                was = ("SOFFIT_PAGE", page) in emitted or soffit_job_emitted
            else:
                was = ("REWORK", page) in emitted
            if typ == "SOFFIT" and soffit_zone:
                # resolved -- a soffit/return zone exists; NOT an unresolved
                # convention (verdict rule wants the list to reach [])
                soffit_resolved_hits += 1
                continue
            # TYP_OF / MATCH_LINE / REWORK (and unresolved SOFFIT): NO
            # auto-resolution signal exists in v1 (SF never auto-multiplied --
            # two-tier rule) so they are ALWAYS unresolved -> check fail.
            # Stage 3b (spec'd, not built): POST /coverage-ack flips resolved.
            unresolved.append({"type": typ, "page": page, "text": text,
                               "emitted": bool(was), "resolved": False})
            if not was:
                # RECOVERED drop (recon gap #2: pages skipped at app.py:600-601
                # never reach the 917/926 lookups) -- re-scan recovers them; the
                # emitter is NOT fixed in v1 (money-path flow, separate lane).
                if _page_entry(job, page) is None:
                    _add_job_flag(job, report, _COV + "'%s' on page %d which has no measurements"
                                  " — the repetition/continuation is not represented in this"
                                  " takeoff." % (text, page))
                else:
                    _add_page_flag(job, report, page, _COV + "'%s' on this sheet has no matching"
                                   " flag from the reader — confirm it is represented in this"
                                   " takeoff." % text)
    # STRUCTURE_TERM rollup: entries per (term, page) capped at 10 pages/term;
    # ONE flag per term (page list) to keep 100-page sets readable. Always
    # unresolved in v1 -- same resolution semantics as TYP.
    for term, pages in sorted(struct_pages.items()):
        for page in pages[:10]:
            unresolved.append({"type": "STRUCTURE_TERM", "page": page, "text": term,
                               "emitted": False, "resolved": False,
                               "n_pages": len(pages)})
        if len(pages) > 10:
            notes.append("structure_term '%s' capped at 10 of %d pages" % (term, len(pages)))
        plist = ", ".join(str(p) for p in pages[:6]) + ("…" if len(pages) > 6 else "")
        _add_job_flag(job, report, _COV + "'%s' appears on page(s) %s — %s faces/scope must be"
                      " accounted or excluded explicitly." % (term, plist, term.lower()))
    if soffit_resolved_hits:
        notes.append("soffit/return hits resolved by existing zone(s): %d page(s)" % soffit_resolved_hits)
    # per-page rollup of per-poly signals /status does not expose (recon OPEN
    # #5 answer: YES -- only /polygons/{jid}/{page} exposes them today)
    per_page = {}
    pbp = _pbp(job)
    for e in (job.get("takeoffData") or []):
        page = e.get("pageNumber")
        polys = pbp.get(page) or []
        n_sug = sum(1 for p in polys if p.get("suggest_only"))
        per_page[str(page)] = {
            "sf_warn": sum(1 for p in polys if p.get("sf_warn")),
            "scale_risk": sum(1 for p in polys if p.get("scale_risk")),
            "suggest_only": n_sug,
            "view_scale": sum(1 for p in polys if p.get("view_scale")),
            "flags": len(e.get("flags") or []),
        }
        # suggestions_stale (recon gap #4: stale robot-count after accepts;
        # healed at report level on re-run -- patch P4 is stage 3b, not v1)
        for f in (e.get("flags") or []):
            m = re.search(r"(\d+) AI wall suggestion", f)
            if m and int(m.group(1)) != n_sug:
                notes.append("suggestions_stale: p%s flag says %s, live suggest_only=%d"
                             % (page, m.group(1), n_sug))
                break
    report["per_page"] = per_page
    report["unresolved_conventions"] = unresolved
    return {"status": "fail" if unresolved else "pass", "notes": notes}


# -- check 4: plan-to-elevation MVP -------------------------------------------
def _struct_hlines(pg):
    """SYNC WITH app.py:338-361 (_page_struct_lines, horizontal part only) --
    local copy for the textless-elevation rescue; importing app is circular."""
    W, H = pg.rect.width, pg.rect.height
    lmin = 0.06 * max(W, H)
    hlines = []
    try:
        for d in pg.get_drawings():
            for it in d["items"]:
                segs = []
                if it[0] == "l":
                    segs = [(it[1].x, it[1].y, it[2].x, it[2].y)]
                elif it[0] == "re":
                    r = it[1]
                    segs = [(r.x0, r.y0, r.x1, r.y0), (r.x0, r.y1, r.x1, r.y1)]
                for (x1, y1, x2, y2) in segs:
                    if (x2 - x1) ** 2 + (y2 - y1) ** 2 < lmin ** 2:
                        continue
                    if abs(y2 - y1) < abs(x2 - x1) * 0.02:
                        hlines.append((min(x1, x2), max(x1, x2), (y1 + y2) / 2))
    except Exception:
        pass
    return hlines[:600]


def _is_elevation_page_cov(pg, text):
    """SYNC WITH app.py:300-336 (is_elevation_page v2) -- local copy; coverage
    cannot import app (circular / FastAPI side effects)."""
    t = (text or "").lower()
    if _ELEV_TITLE_RE.search(t):
        return True
    if ("roof plan" in t) or ("floor plan" in t):
        return False
    if ("elevation" in t) or ("return" in t) or ("soffit" in t):
        return True
    if len(t) < 60:
        try:
            words = pg.get_text("words") or []
            lv = sum(1 for w in words if (w[4] or "").strip().upper() in ("LEVEL", "ROOF")
                     or (w[4] or "").strip().upper().startswith("T.O"))
            if lv >= 3:
                hl = _struct_hlines(pg)
                W9 = pg.rect.width
                long_h = sum(1 for (x0, x1, _y) in hl if (x1 - x0) > 0.30 * W9)
                if long_h >= 3:
                    return True
        except Exception:
            pass
    return False


def _story_table(doc, elev_pages, pdf_bytes, raw):
    """Story table (spec 7a): re-read level markers per elevation page at
    coverage time -- takeoffData 'levels' keeps top-8 and drops y (app.py:736),
    DECISION: that stored surface is left alone, coverage re-reads instead.
    Returns {levels, story_heights_ft, building_height_ft, n_markers}."""
    named = {}
    total_markers = 0
    for page in elev_pages:
        pi = page - 1
        try:
            lv = callouts.read_levels_pg(doc[pi])
            if len(lv) < 2 and len(raw.get(pi, "")) < 250 and ocr_text.available():
                lv = ocr_text.read_levels(pdf_bytes, pi)   # flattened-page fallback
        except Exception:
            lv = []
        if not lv:
            continue
        total_markers += len(lv)
        for l in lv:
            lab = re.sub(r"\s+", " ", str(l.get("label") or "")).strip().upper()[:40]
            try:
                ft = float(l.get("ft"))
            except Exception:
                continue
            key = (lab, round(ft, 2))
            named.setdefault(key, {"label": lab, "ft": round(ft, 2)})
    levels = sorted(named.values(), key=lambda x: x["ft"])
    fts = sorted({l["ft"] for l in levels})
    story = []
    for i in range(1, len(fts)):
        d = round(fts[i] - fts[i - 1], 2)
        if 6.0 <= d <= 20.0:    # plausible story heights only
            story.append(d)
    bh = None
    grade = [l["ft"] for l in levels if "GRADE" in l["label"]]
    roof = [l["ft"] for l in levels if ("ROOF" in l["label"] or "PARAPET" in l["label"])]
    if grade and roof:
        bh = round(max(roof) - min(grade), 2)
    return {"levels": levels[:40], "story_heights_ft": story[:20],
            "building_height_ft": bh, "n_markers": total_markers}


def _page_scale_ftpi(entry):
    """ft-per-inch from a takeoffData entry's verified scale string."""
    try:
        if entry and entry.get("verifiedScale"):
            m = re.search(r"1\"\s*=\s*([\d.]+)'", str(entry.get("scale") or ""))
            if m:
                return float(m.group(1))
    except Exception:
        pass
    return None


def _faces(job, doc, elev_pages, raw, report):
    """Faces inventory (spec 7b): views via vector_hatch.view_boxes over
    page_geometry on elevation pages; compass name attached ONLY when the
    page-global title regex yields exactly one compass word (view-level title
    localization = RESEARCH; no view-to-name association exists).
    RUNTIME CONTAINMENT (documented deviation): the ink pass runs only on
    elevation pages that HAVE a takeoffData entry (measured), capped at 12
    pages by page SF -- get_drawings on unmeasured sheets of a 100+ page volume
    costs minutes; unmeasured elevation sheets are caught by the sheet-index
    reconciliation lane instead. ASSUMPTION (spec register): the <5 s runtime
    budget does not hold on 300MB+ sets; the cap contains the cost."""
    pbp = _pbp(job)
    td_pages = {e.get("pageNumber"): e for e in (job.get("takeoffData") or [])}
    cand = [p for p in elev_pages if p in td_pages]
    cand.sort(key=lambda p: -sum(float(z.get("netArea") or 0)
                                 for z in (td_pages[p].get("zones") or [])))
    capped = len(cand) > 12
    cand = cand[:12]
    expected, found, missing = [], [], []
    notes = ["faces_pass_capped_at_12_pages"] if capped else []
    skipped = [p for p in elev_pages if p not in td_pages]
    if skipped:
        notes.append("unmeasured elevation-classified pages not ink-scanned: %d"
                     " (sheet-index lane covers them)" % len(skipped))
    for page in sorted(cand):
        pi = page - 1
        try:
            pg = doc[pi]
            W, H = pg.rect.width, pg.rect.height
            geo = vector_hatch.page_geometry(pg)
            boxes = vector_hatch.view_boxes(geo, W, H)
        except Exception:
            continue
        if not boxes:
            continue
        # compass name: page-global title regex -- exactly one compass word or null
        low = (raw.get(pi, "") or "").lower()
        compass = {w for w in _COMPASS_WORDS
                   if re.search(r"\b%s\s+[\w\s]{0,18}elevation" % w, low)}
        name = list(compass)[0].upper() if len(compass) == 1 else None
        ftpi = _page_scale_ftpi(td_pages.get(page))
        try:
            marks = callouts.read_levels_pg(pg)   # display-coord level markers
        except Exception:
            marks = []
        dims = (job.get("dims_by_page") or {}).get(page) \
            or (job.get("dims_by_page") or {}).get(str(page)) or {}
        pw, ph = float(dims.get("width") or W), float(dims.get("height") or H)
        polys = _counted(pbp.get(page) or [])
        risky = [p for p in polys if p.get("scale_risk")]
        for vi, (bx0, by0, bx1, by1) in enumerate(boxes):
            wft = round((bx1 - bx0) * (ftpi / 72.0), 1) if ftpi else None
            vm = [m for m in marks if by0 <= m.get("y", -1) <= by1]
            hft = round(max(m["ft"] for m in vm) - min(m["ft"] for m in vm), 1) if len(vm) >= 2 else None
            expected.append({"page": page, "view": vi, "name": name,
                             "width_ft": wft, "height_ft": hft})
            vpolys = []
            for p in polys:
                pts = p.get("points") or []
                if not pts:
                    continue
                cx = sum(x for x, _ in pts) / len(pts) * pw
                cy = sum(y for _, y in pts) / len(pts) * ph
                if bx0 <= cx <= bx1 and by0 <= cy <= by1:
                    vpolys.append(p)
            if not vpolys:
                missing.append({"page": page, "view": vi, "name": name})
                _add_page_flag(job, report, page, _COV + "an elevation view on page %d has no"
                               " measured walls — confirm it is out of scope or measure it." % page)
                continue
            vsf = sum(float(p.get("area_sf") or 0) for p in vpolys)
            occ = None
            # views containing scale_risk polys are excluded from occupancy
            # (vector_hatch.py:1968/1978 signals honored)
            if not any(p.get("scale_risk") for p in vpolys):
                vft = None
                vs = {p.get("view_scale") for p in vpolys if p.get("view_scale")}
                if len(vs) == 1:
                    try:
                        vft = float(list(vs)[0])
                    except Exception:
                        vft = None
                use_ftpi = vft or ftpi
                if use_ftpi:
                    w_ft = (bx1 - bx0) * use_ftpi / 72.0   # ft conversion per gold_bench.py:190
                    h_ft = hft if hft else (by1 - by0) * use_ftpi / 72.0
                    if w_ft > 0 and h_ft > 0:
                        occ = round(vsf / (w_ft * h_ft), 3)
            # ASSUMPTION (spec register): the 1.10 box-occupancy band needs
            # corpus calibration before enabling the flag -- occupancy is
            # computed but REPORT-ONLY in v1 (no flag emitted, per the owner's
            # calibration note). Low occupancy is NOT flagged either (partial
            # cladding is legitimate; poly extent is a lower bound).
            found.append({"page": page, "view": vi, "sf": round(vsf, 1), "occupancy": occ})
        if risky:
            notes.append("p%d: %d scale_risk poly(s) — occupancy excluded for their views"
                         % (page, len(risky)))
    return {"expected": expected, "found": found, "missing": missing, "notes": notes}


# -- sheet-index reconciliation (owner refinement 2026-07-28) -----------------
def _title_block_words(pg):
    """Words in the title-block region (bottom-right ~18% x 20% of the DISPLAY
    page), rotation-corrected, in reading order."""
    W, H = pg.rect.width, pg.rect.height
    rot = pg.rotation_matrix
    out = []
    try:
        for w in pg.get_text("words"):
            q = fitz.Point((w[0] + w[2]) / 2, (w[1] + w[3]) / 2) * rot
            if q.x >= W * 0.82 and q.y >= H * 0.80:
                out.append((q.y, q.x, w[4]))
    except Exception:
        pass
    out.sort()
    return [t for _, _, t in out]


def _sheet_of_page(pg):
    """(sheet_number|None, title|None) from the title block. v2 (2026-07-29): the
    sheet number comes from titleblock.text_sheet_of_page — LABEL-ANCHORED (token
    nearest a DRAWING/SHEET NUMBER label) with wider sheet shapes; the legacy
    bottom-most-token rule misread spec tokens (real 26-169 p65 trap: 'DEARBORN
    BC-45 CLEANER' bound as sheet BC-45 while the DRAWING NUMBER box says P-003).
    Legacy rule kept as fallback when titleblock reads nothing. The title is
    still the text after the sheet token."""
    toks = _title_block_words(pg)
    num, title = None, None
    idx = None
    try:
        num = titleblock.text_sheet_of_page(pg)
    except Exception:
        num = None
    if num:
        for i, t in enumerate(toks):
            if (t or "").strip().upper() == num:
                idx = i                       # last occurrence = the ID box token
    else:
        for i, t in enumerate(toks):          # legacy fallback: bottom-most token
            if _SHEETNUM_RE.match((t or "").strip()):
                num, idx = t.strip(), i
    if idx is not None:
        tail = " ".join(toks[idx + 1:]).strip()
        title = re.sub(r"\s+", " ", tail).upper()[:80] or None
    return num, title


def _parse_sheet_index(doc, raw, up):
    """Parse a SHEET INDEX / DRAWING INDEX region on the first ~5 pages for
    A-series ELEVATION sheet rows. Returns list of {sheet, title} or None when
    no index heading is found."""
    for pi in range(min(5, doc.page_count)):
        if not _INDEX_HEAD_RE.search(up.get(pi, "")):
            continue
        rows = defaultdict(list)
        try:
            for w in doc[pi].get_text("words"):
                rows[round(w[1] / 4)].append(w)     # ~4pt y-band = one index row
        except Exception:
            continue
        out = []
        for _, ws in sorted(rows.items()):
            ws = sorted(ws, key=lambda w: w[0])
            toks = [(w[4] or "").strip() for w in ws]
            for i, t in enumerate(toks):
                if _A_SHEET_RE.match(t):
                    title = re.sub(r"\s+", " ", " ".join(toks[i + 1:])).upper()
                    if "ELEVATION" in title and "INTERIOR" not in title:
                        out.append({"sheet": t, "title": title[:80]})
                    break
        if out:
            return out
    return None


def _sheet_index(job, doc, raw, up, report):
    """Sheet-index reconciliation. Declared elevation sheets come from, in
    order: (1) the drawing index on the first ~5 pages (owner refinement);
    when no index exists (frequent -- many volumes carry the index in another
    volume), fall back to (2) the set's OWN title blocks (elevation-titled
    sheets present in the set but not measured) and (3) suffix-series gap
    inference on elevation sheet numbers (A4.6A/B/C/E present -> D missing:
    the 'A4.6D-class' gap). (2)+(3) are a documented EXTENSION of the owner
    refinement -- without them a volume with no index page could never surface
    a missing elevation sheet. No index found still records
    sheet_index_unavailable (counts toward unverified, never pass)."""
    td_pages = {e.get("pageNumber") for e in (job.get("takeoffData") or [])}
    measured_sheets = {}
    page_sheets = {}    # page -> (num, title) for every page (used by fallbacks)
    for pi in range(doc.page_count):
        try:
            num, title = _sheet_of_page(doc[pi])
        except Exception:
            num, title = None, None
        page_sheets[pi + 1] = (num, title)
        if (pi + 1) in td_pages and num:
            measured_sheets[num.upper()] = pi + 1
    notes = []
    missing = []
    idx = _parse_sheet_index(doc, raw, up)
    if idx is None:
        notes.append("sheet_index_unavailable")
    else:
        for row in idx:
            if row["sheet"].upper() not in measured_sheets:
                missing.append({"sheet": row["sheet"], "title": row["title"], "source": "index"})
                _add_job_flag(job, report, _COV + "sheet %s '%s' listed in the drawing index but"
                              " not measured — confirm out of scope or measure it."
                              % (row["sheet"], row["title"]))
    # fallback (2): elevation-titled sheets present in the set but unmeasured
    elev_sheet_nums = {}
    for page, (num, title) in page_sheets.items():
        if not num or not title:
            continue
        if "ELEVATION" in title and "INTERIOR" not in title and num.upper().startswith("A"):
            elev_sheet_nums[num.upper()] = (page, title)
            if page not in td_pages and num.upper() not in measured_sheets \
                    and not any(m["sheet"].upper() == num.upper() for m in missing):
                missing.append({"sheet": num, "title": title, "page": page, "source": "title_blocks"})
                _add_job_flag(job, report, _COV + "sheet %s '%s' (page %d) is an elevation sheet"
                              " in this set but not measured — confirm out of scope or measure"
                              " it." % (num, title, page))
    # fallback (3): suffix-series gap on elevation sheet numbers (A4.6D-class)
    fams = defaultdict(set)
    for num in elev_sheet_nums:
        m = re.match(r"^([A-Z]{1,2}\d{1,2}\.\d{1,2})([A-Z])$", num)
        if m:
            fams[m.group(1)].add(m.group(2))
    for base, letters in sorted(fams.items()):
        if len(letters) < 2:
            continue
        run = [chr(c) for c in range(ord("A"), ord(max(letters)) + 1)]
        for letter in run:
            cand = base + letter
            if letter in letters:
                continue
            # absent from the ENTIRE volume (not merely unmeasured)?
            if any((num or "").upper() == cand for num, _ in page_sheets.values()):
                continue
            if any(m["sheet"].upper() == cand for m in missing):
                continue
            missing.append({"sheet": cand, "title": None, "source": "series_gap"})
            _add_job_flag(job, report, _COV + "sheet %s — the elevation series %s%s-%s%s in this"
                          " set skips '%s'; the sheet may live in another volume — confirm out"
                          " of scope or measure it." % (cand, base, min(letters), base,
                                                        max(letters), letter))
    report["sheets_missing"] = missing
    status = "fail" if missing else ("unavailable" if idx is None else "pass")
    return {"status": status, "notes": notes, "indexed_rows": len(idx) if idx else 0}


def _check_plan_elev(job, doc, pdf_bytes, raw, up, report):
    if doc is None:
        return {"status": "unavailable", "notes": ["pdf unreadable"]}
    elev_pages = []
    for pi in range(doc.page_count):
        try:
            if _is_elevation_page_cov(doc[pi], raw.get(pi, "")):
                elev_pages.append(pi + 1)
        except Exception:
            continue
    story = _story_table(doc, elev_pages, pdf_bytes, raw)
    faces = _faces(job, doc, elev_pages, raw, report)
    report["faces_expected"] = faces["expected"]
    report["faces_found"] = faces["found"]
    report["faces_missing"] = faces["missing"]
    sheet = _sheet_index(job, doc, raw, up, report)
    notes = faces["notes"] + sheet["notes"]
    # Unavailability (spec section 7): scale/marker/view inputs absent
    unavailable = (not elev_pages) or (story.get("n_markers", 0) == 0) or (not faces["expected"])
    if faces["missing"] or sheet["status"] == "fail":
        status = "fail"
    elif unavailable or sheet["status"] == "unavailable":
        status = "unavailable"
    else:
        status = "pass"
    return {"status": status, "notes": notes,
            "story_table": {"levels": story["levels"],
                            "story_heights_ft": story["story_heights_ft"],
                            "building_height_ft": story["building_height_ft"]},
            "elevation_pages": elev_pages,
            "sheet_index": {"status": sheet["status"], "indexed_rows": sheet["indexed_rows"]}}


# -- check 5: envelope closure ------------------------------------------------
def _plan_pages(up, page_count):
    """NEW positive floor/roof-plan classifier (spec section 8) -- today only a
    negative plan signal exists (app.py:315). Feeds the future footprint tracer."""
    out = []
    for pi in range(page_count):
        if _PLAN_TITLE_RE.search(up.get(pi, "")):
            out.append(pi + 1)
    return out


def _envelope_closure(job, doc, up):
    """v1: the exterior-footprint closed-loop tracer does not exist anywhere in
    the codebase (recon verdict) -- always unavailable. Stage-5 comparator spec
    lives in the build spec section 8 (perimeter from heavy-line loop or roof
    outline; ASSUMPTION (spec register): ~1-2 ft/side overhang bias on the
    roof-outline path; +/-15% closure band, calibrate before enforcement)."""
    pages = _plan_pages(up, doc.page_count) if doc is not None else []
    return {"status": "unavailable", "plan_pages": pages,
            "reason": "exterior footprint tracer not built"}


# -- verdict ------------------------------------------------------------------
def _verdict(checks, report):
    """DoD rule (spec section 10), enforced here and ONLY here: 'verified' iff
    ALL checks pass AND materials_missing==[] AND failed_ratios==[] AND
    unresolved_conventions==[] AND faces_missing==[] AND sheets_missing==[]
    AND zero critical typed flags. ANY status in {unavailable, error} forces
    'unverified' with reason 'input unavailable: <check>'. No third verdict
    value. In v1 envelope_closure is always unavailable => always unverified --
    intended: absence of evidence flags, never passes."""
    reasons = []
    for name, c in checks.items():
        if c.get("status") in ("unavailable", "error"):
            reasons.append("input unavailable: " + name)
    sheet = (checks.get("plan_elevation") or {}).get("sheet_index") or {}
    if sheet.get("status") == "unavailable":
        reasons.append("input unavailable: sheet_index")
    for d in report.get("materials_missing", [])[:20]:
        nm = d.get("name") or d.get("code") or "?"
        code = (" (%s)" % d["code"]) if d.get("code") else ""
        reasons.append("material declared but not found: %s%s" % (nm, code))
    for u in report.get("unresolved_conventions", [])[:20]:
        reasons.append("unresolved convention: %s '%s' on p%s" % (u["type"], u["text"], u["page"]))
    for f in report.get("failed_ratios", [])[:20]:
        reasons.append("sanity ratio failed: %s%s" % (f["ratio"],
                       (" on p%s" % f["page"]) if f.get("page") else ""))
    for fm in report.get("faces_missing", [])[:20]:
        reasons.append("elevation view without measurements: p%s view %s" % (fm["page"], fm["view"]))
    for sm in report.get("sheets_missing", [])[:20]:
        reasons.append("sheet not measured: %s" % sm["sheet"])
    crit = [t for t in report.get("flags_typed", []) if t.get("severity") == "critical"]
    for t in crit[:20]:
        reasons.append("critical flag%s: %s" % ((" on p%s" % t["page"]) if t.get("page") else "",
                                                t["code"]))
    all_pass = all(c.get("status") == "pass" for c in checks.values())
    clean = (not report.get("materials_missing") and not report.get("failed_ratios")
             and not report.get("unresolved_conventions") and not report.get("faces_missing")
             and not report.get("sheets_missing") and not crit)
    verdict = "verified" if (all_pass and clean and not reasons) else "unverified"
    seen, out = set(), []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return verdict, out[:60]


# -- entry point --------------------------------------------------------------
def run(job, pdf_bytes, *, jlog=None):
    """Entry point. Mutates job: writes job['coverageReport'], appends flags.
    Returns the report dict. Never raises (per-check containment inside; the
    app.py call site adds its own try/except on top -- coverage can NEVER fail
    a job). Crashed check => status 'error' (counts as unavailable for the
    verdict) -- never the silent bare-except-returns-empty pattern."""
    log = jlog if callable(jlog) else (lambda m, lv="info": None)
    report = {
        "version": SCHEMA_VERSION,
        "generatedAt": _now_iso(),
        "verdict": "unverified",
        "reasons": [],
        "checks_passing": 0,
        "checks_total": 5,
        "checks": {},
        "faces_expected": [], "faces_found": [], "faces_missing": [],
        "materials_declared": [], "materials_found": [],
        "materials_missing": [], "materials_undeclared": [],
        "sheets_missing": [],
        "unresolved_conventions": [],
        "failed_ratios": [],
        "flags_typed": [],
        "job_flags": [],
        "per_page": {},
        "norms_source": "frozen-defaults",
    }
    try:
        try:
            typed = classify_flags(job.get("takeoffData") or [])
        except Exception:
            typed = []
        report["flags_typed"] = typed
        # re-home existing job-level strings into job_flags (originals stay on
        # takeoffData[-1] untouched -- DECISION recon OPEN #2)
        for t in typed:
            if t["scope"] == "job" and t["text"] not in report["job_flags"]:
                report["job_flags"].append(t["text"])
        doc, raw, up = None, {}, {}
        try:
            if pdf_bytes:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                raw, up = _texts(doc)
        except Exception:
            doc = None
        checks = {}
        try:
            checks["legend_reconciliation"] = _check_legend(job, doc, raw, up, report) \
                if doc is not None else {"status": "unavailable", "notes": ["pdf unreadable"]}
        except Exception as e:
            checks["legend_reconciliation"] = {"status": "error", "error": str(e)[:200]}
        try:
            checks["sanity_ratios"] = _check_ratios(job, report)
        except Exception as e:
            checks["sanity_ratios"] = {"status": "error", "error": str(e)[:200]}
        try:
            checks["sentinel_rollup"] = _check_sentinels(job, doc, raw, up, typed, report)
        except Exception as e:
            checks["sentinel_rollup"] = {"status": "error", "error": str(e)[:200]}
        try:
            checks["plan_elevation"] = _check_plan_elev(job, doc, pdf_bytes, raw, up, report)
        except Exception as e:
            checks["plan_elevation"] = {"status": "error", "error": str(e)[:200]}
        try:
            checks["envelope_closure"] = _envelope_closure(job, doc, up)
        except Exception as e:
            checks["envelope_closure"] = {"status": "error", "error": str(e)[:200]}
        report["checks"] = checks
        report["checks_passing"] = sum(1 for c in checks.values() if c.get("status") == "pass")
        try:
            verdict, reasons = _verdict(checks, report)
        except Exception:
            verdict, reasons = "unverified", ["verdict computation error"]
        report["verdict"] = verdict
        report["reasons"] = reasons
        # refresh per_page flag counts (later checks appended page flags after
        # the check-3 rollup snapshotted them)
        try:
            for e in (job.get("takeoffData") or []):
                k = str(e.get("pageNumber"))
                if k in report["per_page"]:
                    report["per_page"][k]["flags"] = len(e.get("flags") or [])
        except Exception:
            pass
        try:
            if doc is not None:
                doc.close()
        except Exception:
            pass
        try:
            log("Coverage: %s — %d/%d checks passing, %d job flag(s)"
                % (verdict, report["checks_passing"], report["checks_total"],
                   len(report["job_flags"])), "ok" if verdict == "verified" else "warn")
        except Exception:
            pass
    except Exception as e:
        report["reasons"] = ["coverage internal error: %s" % str(e)[:200]]
    try:
        job["coverageReport"] = report
    except Exception:
        pass
    return report
