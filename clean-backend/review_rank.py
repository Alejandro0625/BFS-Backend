# -*- coding: utf-8 -*-
r"""review_rank.py -- ONE owner of "which regions should the estimator look at first?".

WHY THIS EXISTS (FIX_TWO_TIER_GATE.md, the replacement for DO-item 2)
---------------------------------------------------------------------
The honest two-tier ASSERT/FLAG product was measured and cannot be built: its ceiling is
15.9% confidently-wrong on 3.81% of corpus SF, and even that operating point does not
survive a fold swap. What DOES reproduce is the ordering -- the engine's own reader class
is real self-knowledge:

    multi 61% > v13 51% > train 38% > flood 24% > fill 21% > color 21% > rc 13%
    (share of that class's walls where the engine booked >1.15x the wall's true SF)

So: nothing is auto-booked, nothing is asserted, no SF moves. Every region is still
reviewed and confirmed by the estimator exactly as today. This module only says which
ones to put in front of her FIRST. That claim cannot violate the zero-confidently-wrong
promise, because it books nothing.

THE ORDERING KEY IS A BOUND, NOT A RATE
---------------------------------------
Ranking by the measured rate would put `rc` (13.0% on 223 walls) AFTER `band` (10.0% on
**20** walls) on a difference no amount of data supports. The key is the 95% Wilson UPPER
bound of the rate, which charges every class for its own uncertainty: same rate + fewer
walls => ranked EARLIER. Thin support sorting earlier is the honest direction when the
only consequence is that a human looks at it sooner. There is no threshold anywhere in
this module -- that is deliberate, and it is why `FIX_TWO_TIER_GATE`'s unstable operating
point cannot repeat here.

THE TABLE IS A MEASUREMENT, NOT A CONSTANT SOMEBODY TYPED
---------------------------------------------------------
Every number below is DERIVED by `bfs_overnight\probe_review_rank.py` (30/30) from the
certified 122-job corpus and pasted here because production cannot read a corpus.
`bfs_overnight\test_review_rank_stamp.py` re-derives it from that corpus on every run and
FAILS if this table has drifted, so a stale table is a red test, not a silent wrong order.
The ordering was re-derived independently in two disjoint job-level halves and agrees
(Spearman rho 0.83); a truth-shuffled twin of the corpus collapses the spread (53.9 ->
24.6), so the order is reading risk, not class size.

⚠ THE CLASS MUST BE STAMPED AT detect()-RETURN OR THIS TABLE DOES NOT APPLY.
`material` is overwritten twice downstream (callouts.name_regions, and app.py's per-job
code map), so by the time a piece reaches the response its provenance is gone and every
region would collapse into one bucket. `app.py` stamps `p["reader_class"]` on the pieces
the moment `vector_hatch.detect` returns -- the same strings `gold_bench.py` reads at the
same moment, which is the only reason the measured risks above apply verbatim.
"""

# ---------------------------------------------------------------- the class of a piece
# Prefix -> class, VERBATIM from `gold_bench.py:370-378` (`_rdr`). Order is load-bearing:
# "Wall area (AI boundary" must be tested before "Wall area".
_PREFIXES = (("Hatched area", "hatch"), ("Color fill", "color"),
             ("Rendered", "rc"), ("Wall area (AI boundary", "v13"),
             ("Wall band", "band"), ("Wall area", "flood"),
             ("Panel wall", "fill"))
FALLBACK_CLASS = "train"     # no prefix matched -- named inside detect() before we saw it
MULTI_CLASS = "multi"        # a region several reader classes contributed to

# Display provenance for the evidence receipt (unchanged strings -- these already shipped)
LABELS = {"hatch": "Hatch reader", "color": "Drawn color", "rc": "Rendered reader",
          "v13": "AI boundary model", "band": "Story-band", "flood": "Structural flood",
          "fill": "Drawn fill", "train": "Drawing geometry", "multi": "Several readers",
          "model": "Trained model", "texture": "Texture reader", "density": "Density reader",
          "markup": "Your markup (exact)", "suggestion": "AI suggestion (not counted)"}

# ------------------------------------------------------------------- the measured risk
# walls / wrong_high / rate% / wilson-upper%  -- REVIEW_RANK.json, corpus md5
# EE743811EF435752F05B50D34431C673 (corpus_certified_122jobs_2026-08-11.json), 2026-08-11.
# That md5 is the artifact three independent runs agreed on byte-for-byte (master's
# unsharded 05:59 run and both sharded runs), so the table's source is not one process.
RISK = {
    "multi": {"walls": 70, "wrong_high": 43, "rate": 61.4, "upper": 72.0},
    "v13":   {"walls": 878, "wrong_high": 450, "rate": 51.3, "upper": 54.5},
    "train": {"walls": 248, "wrong_high": 94, "rate": 37.9, "upper": 44.1},
    "fill":  {"walls": 38, "wrong_high": 8, "rate": 21.1, "upper": 36.3},
    "flood": {"walls": 102, "wrong_high": 24, "rate": 23.5, "upper": 32.6},
    "band":  {"walls": 20, "wrong_high": 2, "rate": 10.0, "upper": 30.1},
    "color": {"walls": 136, "wrong_high": 29, "rate": 21.3, "upper": 28.9},
    "hatch": {"walls": 23, "wrong_high": 2, "rate": 8.7, "upper": 26.8},
    "rc":    {"walls": 223, "wrong_high": 29, "rate": 13.0, "upper": 18.1},
}
TABLE = {"source": "corpus_certified_122jobs_2026-08-11.json",
         "source_md5": "EE743811EF435752F05B50D34431C673",
         "derived": "2026-08-11", "drawn_walls": 1738, "fold_rho": 0.83,
         "probe": "bfs_overnight/probe_review_rank.py"}

# riskiest first; a class with no measurement is not "safe", it is UNKNOWN -- see rank()
ORDER = [c for c, _ in sorted(RISK.items(), key=lambda kv: (-kv[1]["upper"], kv[0]))]
UNMEASURED_RANK = 0          # sorts before every measured class
MARKUP_CLASS = "markup"      # the estimator's own marked set
MARKUP_RANK = len(ORDER) + 1  # ...sorts LAST: exact by geometry, nothing to re-measure


def classify(material):
    """The reader class of a piece, from the raw name its reader gave it.

    Must be called on the name the piece has AT detect()-RETURN. Downstream renames
    (callouts, code map, material grouping) destroy the prefix, and this function would
    then answer FALLBACK_CLASS for everything -- silently, and with the wrong risk."""
    m = str(material or "")
    for pre, nm in _PREFIXES:
        if m.startswith(pre):
            return nm
    return FALLBACK_CLASS


def combine(classes):
    """The class of a REGION built from pieces of the given classes.

    Several classes -> `multi`, which is measured in its own right (70 walls, 61.4%
    confidently-wrong -- the worst class in the corpus). It is deliberately NOT the max or
    the average of its members: a wall two readers disagree about is a different animal
    from either of them, and the corpus says so."""
    seen = [c for c in dict.fromkeys(c for c in classes if c)]
    if not seen:
        return None
    return seen[0] if len(seen) == 1 else MULTI_CLASS


def risk(cls):
    """Wilson-upper wrong-HIGH % for a class, or None if it was never measured.

    None is not zero. `model`/`texture`/density suggestions have no corpus measurement at
    all, so they cannot be given a number -- they are ranked FIRST instead (see rank())."""
    r = RISK.get(cls)
    return r["upper"] if r else None


def rate(cls):
    """The measured point rate (what actually happened), beside the bound used to rank."""
    r = RISK.get(cls)
    return r["rate"] if r else None


def rank(cls):
    """1 = look at this first. 0 = unmeasured class, look at it before anything measured.

    An unmeasured reader ranking LAST would be the `bfs-self-selected-denominator` shape:
    a class escapes the queue's attention precisely because nobody has graded it.

    `markup` is the ONE class that ranks last, and it is not an exception to that rule: a
    marked region's SF comes from the estimator's own polygon and is exact by geometry
    (the owner's constitution, tier 1). It is not unmeasured, it is not measured by this
    corpus at all — it is not a reader."""
    if cls == MARKUP_CLASS:
        return MARKUP_RANK
    if cls in ORDER:
        return ORDER.index(cls) + 1
    return UNMEASURED_RANK


def label(cls, src=""):
    """Human-readable provenance for the evidence receipt."""
    if "bluebeam" in str(src or ""):
        return "Your markup (exact)"
    return LABELS.get(cls, "Drawing geometry")


def why(cls):
    """One sentence the UI can show beside a queue position. Never a number without its
    support -- a 10.0% rate on 20 walls must not read like a 13.0% on 223."""
    if cls == MARKUP_CLASS:
        return "your own marked polygon — exact by geometry, nothing to re-measure"
    r = RISK.get(cls)
    if not r:
        return ("this reader has no corpus measurement — review it before the graded ones")
    return ("this reader booked more SF than the wall on %d of %d benchmark walls (%.0f%%)"
            % (r["wrong_high"], r["walls"], r["rate"]))


def sort_key(cls, sf=0.0):
    """Queue order: riskiest class first, biggest SF first inside a class.

    SF is the tie-break, not the primary key: a 4,000 SF region an `rc` reader traced is
    still a better bet than a 900 SF region the boundary model inferred, and the corpus
    says the second one is wrong 4x as often."""
    return (rank(cls) if rank(cls) else -1, -float(sf or 0.0))


def order_zones(zones):
    """Zones (dicts carrying `readerClass` and `netArea`), riskiest first. Pure -- returns
    a new list and never mutates or drops a zone: a queue that loses a region is worse
    than an unordered one."""
    return sorted(list(zones or []),
                  key=lambda z: sort_key(z.get("readerClass"), z.get("netArea", 0)))


# --------------------------------------------------------------------------- selftest
def _selftest():
    ok = [0]
    bad = []

    def ck(name, cond):
        ok[0] += 1
        if not cond:
            bad.append(name)
        print("  %-56s %s" % (name, "ok" if cond else "FAIL"))

    # classify: identical to gold_bench._rdr, including the order-sensitive pair
    ck("T_CL_hatch", classify("Hatched area 8\" course") == "hatch")
    ck("T_CL_color", classify("Color fill rgb(1,2,3)") == "color")
    ck("T_CL_rc", classify("Rendered siding") == "rc")
    ck("T_CL_v13_before_flood", classify("Wall area (AI boundary model)") == "v13")
    ck("T_CL_flood", classify("Wall area 3") == "flood")
    ck("T_CL_band", classify("Wall band L2") == "band")
    ck("T_CL_fill", classify("Panel wall A") == "fill")
    ck("T_CL_named_is_train", classify("MP-1 METAL PANEL") == "train")
    ck("T_CL_empty_is_train", classify("") == "train" and classify(None) == "train")

    # combine
    ck("T_CB_single", combine(["v13"]) == "v13")
    ck("T_CB_repeat_is_single", combine(["v13", "v13"]) == "v13")
    ck("T_CB_two_is_multi", combine(["v13", "rc"]) == "multi")
    ck("T_CB_none", combine([]) is None and combine([None]) is None)

    # the ordering the corpus produced
    ck("T_OR_multi_first", ORDER[0] == "multi")
    ck("T_OR_rc_last", ORDER[-1] == "rc")
    ck("T_OR_v13_before_rc", rank("v13") < rank("rc"))
    ck("T_OR_band_before_rc_despite_lower_rate",     # the whole reason the key is a bound
       rate("band") < rate("rc") and rank("band") < rank("rc"))
    ck("T_OR_total_order", len(set(rank(c) for c in ORDER)) == len(ORDER))

    # unmeasured is FIRST, and is not zero risk
    ck("T_UN_rank_first", rank("model") == UNMEASURED_RANK == 0)
    ck("T_UN_risk_is_None", risk("model") is None and risk("texture") is None)
    ck("T_UN_sorts_before_worst_measured",
       sort_key("model", 1) < sort_key("multi", 1e9))
    ck("T_UN_why_says_unmeasured", "no corpus measurement" in why("model"))

    # ordering zones
    zs = [{"readerClass": "rc", "netArea": 5000}, {"readerClass": "v13", "netArea": 10},
          {"readerClass": "model", "netArea": 1}, {"readerClass": "multi", "netArea": 20}]
    o = order_zones(zs)
    ck("T_Z_order", [z["readerClass"] for z in o] == ["model", "multi", "v13", "rc"])
    ck("T_Z_no_zone_lost", len(o) == len(zs))
    ck("T_Z_pure", zs[0]["readerClass"] == "rc" and zs is not o)
    big = [{"readerClass": "v13", "netArea": 10}, {"readerClass": "v13", "netArea": 900}]
    ck("T_Z_sf_breaks_ties_within_class",
       [z["netArea"] for z in order_zones(big)] == [900, 10])
    ck("T_Z_missing_class_is_unmeasured",
       order_zones([{"netArea": 5}, {"readerClass": "rc", "netArea": 5}])[0].get("readerClass")
       is None)

    # labels: the strings the evidence receipt already shipped
    ck("T_LB_v13", label("v13") == "AI boundary model")
    ck("T_LB_markup_wins", label("v13", "bluebeam") == "Your markup (exact)")
    ck("T_LB_unknown_default", label("nope") == "Drawing geometry")

    # the table itself
    ck("T_TB_upper_above_rate_everywhere",
       all(v["upper"] >= v["rate"] for v in RISK.values()))
    ck("T_TB_counts_consistent",
       all(0 <= v["wrong_high"] <= v["walls"] for v in RISK.values()))
    ck("T_TB_rate_matches_counts",
       all(abs(100.0 * v["wrong_high"] / v["walls"] - v["rate"]) < 0.1
           for v in RISK.values()))
    ck("T_TB_provenance_present",
       bool(TABLE["source_md5"]) and bool(TABLE["probe"]) and TABLE["fold_rho"] > 0.5)

    print("\nSELFTEST %d/%d" % (ok[0] - len(bad), ok[0]))
    for b in bad:
        print("  FAILED: %s" % b)
    return 0 if not bad else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
