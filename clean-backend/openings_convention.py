"""GROSS-vs-NET CONVENTION FLAG  (Lane 6 confidence layer, 2026-08-05).

WHAT THIS IS
------------
The engine measures the wall the architect drew: a GROSS facade field, openings not
deducted (`vector_hatch.py:596`, engine_net_factor 1.018).  The BFS archive books NET on
walls of any size: measured over 1,872 archive walls, 18.9% of the gross SF is deducted.
That is a CONVENTION DISAGREEMENT, not a measurement error -- the polygon is right and the
number is right for what it measures.

Two opening detectors were built and REFUTED against pre-registered bars on 2026-08-04
(`FIX_OPENINGS.md`): congruence (B4 1.2297x vs bar >=10x) and ink-statistics netting
(all three of P1/P2/P3 fail held-out, every predictor NEGATIVELY correlated).  So the
deduction is not readable from the drawing by classical means, and the product may not
guess it: a guessed deduction that fires where the estimator did not deduct manufactures
exactly the confidently-wrong class the owner's bar forbids.

So this module does the one honest thing: it NAMES the convention, on the walls where it
matters, and hands the estimator the archive's own measured deduction range as
INFORMATION.  It never adds to or subtracts from any total.

HARD INVARIANT
--------------
SF IS NEVER TOUCHED.  `annotate()` fingerprints every SF number in the job before it runs
and again after; if a single one moved, it removes everything it wrote and reports the
breach.  The flag is metadata and flag text only.

WHO GETS FLAGGED  (predicate, engine-side only -- no gold, no oracle at runtime)
--------------------------------------------------------------------------------
  * the page was read by the AI (`source == "texture-auto"`).  On a marked-up page the
    estimator's OWN polygon is the measurement and already carries his convention --
    flagging it would be telling him about his own habit.
  * the piece is counted (not `suggest_only`).
  * the engine deducted NOTHING on it (`holes` empty).  A piece the engine already cut
    openings out of is not being reported gross.
  * the piece measures >= FLOOR_SF (400 SF).

WHY 400 SF -- measured, not chosen
----------------------------------
Full-archive convention census (`openflag_census.py`): every gold wall in
`rb_gold_geom.jsonl` whose page carries a FIX_SCALE oracle-v2 scale, 1,872 walls / 50 jobs
/ 121 pages, truth = the archive booked a deduction of >=5%:

    gross SF band        n     P(archive booked NET)
    0 -  100 SF        729            0.071
    100 -  400 SF        659            0.390
    400 - 1200 SF        345            0.719
    1200 +   SF        139            0.748

Below 400 SF the archive mostly does NOT deduct, so a flag there is noise.  400 SF is the
smallest round floor whose HELD-OUT precision clears 0.70:

    held-out precision, 5-fold BY JOB, gold-side  : 0.7273 (flag-all baseline 0.3531)
    held-out precision, 5-fold BY JOB, engine-side: 0.7179 (engine's own SF, live engine)

and it still covers 86.9% of the archive's entire gross-vs-net gap (118,395 of 136,204 SF).

HONESTY THE FLAG MUST CARRY
---------------------------
The convention is a PER-JOB habit, not a constant: across 32 jobs the realized deduction on
this population runs p25 7.2% / median 16.5% / p75 24.6% of gross, and 8 of 32 jobs (1 in 4)
deducted essentially nothing.  The flag says so.  It is a convention estimate; it is never
applied.
"""

FLOOR_SF = 400.0

# (lo, hi, P(archive booked net), deduction p25/p50/p75 among the walls that WERE deducted)
BANDS = [
    (400.0, 1200.0, 0.719, 17, 30, 37),
    (1200.0, float("inf"), 0.748, 20, 30, 37),
]

# measured but deliberately SILENT -- kept here so the floor is auditable in code
SILENT_BANDS = [(0.0, 100.0, 0.071), (100.0, 400.0, 0.390)]

# per-JOB realized deduction on the flagged population, % of flagged gross (32 jobs)
JOB_P25, JOB_MED, JOB_P75 = 7, 17, 25
JOBS_THAT_BOOK_GROSS_TOO = "8 of 32"

CENSUS = "1,872 archive walls / 50 jobs / 121 pages"
FLAG_NAME = "GROSS_NOT_NET"


def _band(sf):
    for lo, hi, rate, p25, med, p75 in BANDS:
        if lo <= sf < hi:
            return lo, hi, rate, p25, med, p75
    return None


def _fingerprint(job):
    """Every SF number the job publishes, in order. Used to PROVE nothing moved."""
    fp = []
    for e in job.get("takeoffData") or []:
        fp.append(("pg", e.get("pageNumber")))
        for z in e.get("zones") or []:
            fp.append((z.get("materialName"), z.get("netArea"), z.get("grossArea"),
                       z.get("totalOpeningArea")))
        for li in e.get("linearItems") or []:
            fp.append((li.get("material"), li.get("lf")))
        for at in e.get("autoTrim") or []:
            fp.append((at.get("material"), at.get("lf")))
        fp.append(("openings", e.get("openingsCount")))
    for pg, polys in sorted((job.get("polygons_by_page") or {}).items(), key=lambda kv: str(kv[0])):
        for p in polys or []:
            fp.append((str(pg), p.get("area_sf"), p.get("suggest_only"),
                       len(p.get("holes") or [])))
    return tuple(fp)


def annotate(job, jlog=None):
    """Attach the named convention flag. Returns the summary dict (also stored on the job).

    SF-INVARIANT BY CONSTRUCTION AND BY CHECK: writes only new keys, then verifies the SF
    fingerprint is byte-identical and self-reverts if it is not.
    """
    def _log(m, lv="info"):
        if jlog:
            try:
                jlog(m, lv)
            except Exception:
                pass

    before = _fingerprint(job)

    auto_pages = {e.get("pageNumber") for e in (job.get("takeoffData") or [])
                  if e.get("source") == "texture-auto"}
    polys_by_page = job.get("polygons_by_page") or {}

    flagged = []          # (page, piece) for rollup
    touched = []          # pieces we wrote to, so we can undo on a breach
    silent_n = silent_sf = 0
    for pg in sorted(auto_pages):
        polys = polys_by_page.get(pg) or polys_by_page.get(str(pg)) or []
        for p in polys:
            if p.get("suggest_only"):
                continue
            sf = float(p.get("area_sf") or 0.0)
            if sf <= 0:
                continue
            if p.get("holes"):
                continue                      # engine already cut openings out of this one
            if sf < FLOOR_SF:
                silent_n += 1
                silent_sf += sf
                continue
            b = _band(sf)
            if not b:
                continue
            lo, hi, rate, p25, med, p75 = b
            band_label = (f"{int(lo):,}-{int(hi):,} SF" if hi != float("inf")
                          else f"{int(lo):,}+ SF")
            p["conventionFlag"] = {
                "name": FLAG_NAME,
                "title": "Measured GROSS - your archive books NET",
                "sizeBand": band_label,
                "archiveBooksNetPct": round(100 * rate),
                "typicalDeductionPct": [p25, p75],
                "medianDeductionPct": med,
                "appliedToSF": 0,
                "isConventionEstimate": True,
                "basis": f"convention estimate from {CENSUS} - NEVER applied to any total",
                "text": (f"Measured GROSS ({sf:,.0f} SF, no openings deducted). In your "
                         f"archive {round(100*rate)}% of walls this size ({band_label}) are "
                         f"booked NET; when they are, the deduction is typically "
                         f"{p25}-{p75}% (median {med}%). CONVENTION ESTIMATE - not applied "
                         f"to this or any total."),
            }
            touched.append(p)
            flagged.append((pg, sf))

    n = len(flagged)
    gross = sum(sf for _, sf in flagged)
    summary = {
        "flag": FLAG_NAME,
        "walls": n,
        "grossSF": round(gross, 1),
        "pages": sorted({pg for pg, _ in flagged}),
        "floorSF": FLOOR_SF,
        "belowFloorWalls": silent_n,
        "belowFloorSF": round(silent_sf, 1),
        "jobTypicalDeductionPct": [JOB_P25, JOB_P75],
        "jobMedianDeductionPct": JOB_MED,
        "estimatedDeductionSF": [round(gross * JOB_P25 / 100.0), round(gross * JOB_P75 / 100.0)],
        "appliedToSF": 0,
        "isConventionEstimate": True,
        "census": CENSUS,
        "heldOutPrecision": 0.727,
    }

    if n and job.get("takeoffData"):
        lo_sf, hi_sf = summary["estimatedDeductionSF"]
        job["takeoffData"][-1].setdefault("flags", []).append(
            f"⚠ GROSS-vs-NET CONVENTION — {n} wall(s) totalling {gross:,.0f} SF are "
            f"reported GROSS (openings not deducted; that is what the drawing shows). Your "
            f"archive books NET on walls this size: typically {JOB_P25}-{JOB_P75}% of gross "
            f"(median {JOB_MED}%), which on this job would be about {lo_sf:,}-{hi_sf:,} SF. "
            f"On {JOBS_THAT_BOOK_GROSS_TOO} archive jobs nothing was deducted at all, so "
            f"CHECK the elevations — this is a CONVENTION ESTIMATE from {CENSUS} and has "
            f"NOT been applied to any total. The SF above is gross, as measured."
        )
    job["openingsConvention"] = summary

    after = _fingerprint(job)
    if after != before:
        for p in touched:
            p.pop("conventionFlag", None)
        job.pop("openingsConvention", None)
        _log("Gross-vs-net convention flag SELF-REVERTED — it moved an SF number "
             "(this must never happen); takeoff unaffected.", "warn")
        return {"flag": FLAG_NAME, "walls": 0, "sf_invariant_breach": True}

    if n:
        _log(f"Gross-vs-net convention: {n} wall(s) / {gross:,.0f} SF reported GROSS "
             f"— flagged with the archive's own deduction range (SF untouched)", "warn")
    return summary
