# BFS AI Estimator — Master Blueprint (A to Z)
*2026-07-06. How a master estimator actually works, mapped to what's built, what's missing, and the sequence to a sellable product. Companion to BFS_SYSTEM_PLAYBOOK.md.*

---

## PART 1 — HOW A MASTER ESTIMATOR READS A JOB (and what the system does at each step)

### A. Intake
The bid arrives as an ITB email with links (Procore, BuildingConnected, Pipeline): drawing set, specs, addenda, bid form, due date.
- **Built**: manual PDF upload, Queue tab for batches.
- **Missing**: email/link ingestion, multi-document jobs (drawings + specs + addenda as ONE job), due-date tracking. *Sellable products meet the customer at their inbox.*

### B. Sheet triage
A pro flips to A2xx elevations instantly, notes each sheet's scale, spots enlarged/partial elevations, multi-building numbering.
- **Built**: elevation-page filter (text-based; falls back to all pages on flattened sets), biggest-first sorting.
- **Missing**: sheet-index parsing ("A2.1 EXTERIOR ELEVATIONS" from the cover index tells you the pages even when sheets are textless), enlarged-elevation linkage (measure the 1/4" enlarged version, not the 1/16" overall), multi-building grouping.

### C. Scope determination — WHAT is ours
Specs sections 07 42 xx (metal panels), 07 46 xx (siding) + finish legend + alternates decide which materials the siding sub bids. Brick usually = mason's. This is why the estimator DELETES the brick face.
- **Built**: Scope tab (document analysis), delete-with-learning (NOT-CLADDING negatives), material selection flow.
- **Missing**: closing the loop — Scope tab's conclusions should PRE-SELECT which detected materials are in scope ("specs say Metl-Span + fiber cement: auto-select those faces, leave brick unselected"). This single connection turns two tools into one brain.

### D. Measure — WHERE and HOW MUCH (the core, mostly built)
- **Built and strong**: exact digitize of marked sets; vector engine (pattern faces, welded one-piece, openings netted, ~95% of the gold job total); bucket = click a pattern → every face of it; model fallback for scans; deep zoom; calibrate; corner tool for gables.
- **The two biggest accuracy levers still on the table:**
  1. **Read the ELEVATION MARKERS.** Every elevation carries "T.O. STEEL 139'-0"", "FINISHED FLOOR 100'-0"", "CEILING 122'-4"". Markers give EXACT face heights — no scale needed. Cross-multiplying with grid dimensions gives expected face areas to validate every region. This is how a human knows a wall is 39 ft tall without measuring.
  2. **Self-calibration from dimension strings.** A dimension "24'-0"" drawn over a line whose vector length we can measure = exact scale, per sheet, even on mixed-scale sheets. Stronger than title-block text.
- **Missing/partial**: small faces under 100 SF (needs the junk discriminator — keep small faces only when pattern-matched to a big confirmed face); mixed-style faces (slab+ribbed = one material) need learned equivalence.

### E. Openings
Windows/doors subtracted; schedule cross-check; shop rules (some don't deduct under 16 SF — we already support min_opening_sf).
- **Built**: evidence-based opening detection (nested frame/mullions), schedule reader (window/door schedule → exact SF), visible cutouts, veto per opening.
- **Missing**: reconcile detected openings against the schedule automatically ("schedule says 14 type-A windows @ 12x52; I found 12 — two missing on the North face"). That's how a human catches their own misses.

### F. Linear items — where siding bids are won/lost
Trim, corners, J-channel, flashing, coping, watertable, soffit/fascia, transitions. Many jobs are MORE linear than area.
- **Built**: reads the estimator's polyline markups exactly (LF with heights); Budget prices LF separately.
- **Missing (very doable, pure geometry)**:
  - **Outside/inside corner LF** = the vertical edges of the faces we ALREADY have, times height. Free from existing data.
  - **Opening trim LF** = perimeter of every detected opening. Also free.
  - **Base/top trim LF** = bottom/top edges of faces. Free.
  A "Trim (auto)" panel from data we already compute would be a killer differentiator — nobody automates trim.

### G. Counts & conditions
Window wrap counts, penetrations, louvers; height tiers (staging/lifts), existing-vs-new, color count.
- **Built**: openings are already counted per face (just not surfaced as counts).
- **Missing**: a counts panel; height-tier flag (faces above 30 ft → lift note in Budget as equipment line).

### H. Sanity checks — the money rules
- **Built**: scale-unread blocks export, label-vs-geometry typo detection, under-marked warnings, deduction caps, honest flags everywhere, autonomy meter vs answer keys.
- **Missing**: elevation-marker cross-check (from D1) would be the strongest possible sanity: "face measures 4,630 SF; markers×grid say the face is 4,700 gross − 320 openings = 4,380–4,700 plausible ✓".

### I. Pricing
- **Built**: Budget tab (per-material rates, per-material waste, LF pricing, custom lines, margin, rate cards, BFS Excel export).
- **Missing for pro-grade**: supplier quote ingestion (PQ PDFs → panel $/SF, trim $/LF — the Quote folder already sits next to every bid!), labor production rates (SF/crew-day by system), equipment module (lift weeks), tax/bond/GC toggles. Post-bid win/loss tracking feeding the rates back.

### J. Proposal
The deliverable a GC sees: scope letter with inclusions/exclusions, alternates, unit prices, quals.
- **Built**: priced Excel + evidence PDF (strong proof docs).
- **Missing**: **scope-letter generator** — from selected materials + Scope tab exclusions + alternates, generate the proposal text. High value, low risk, and it's a language task (a future session does this well in a day).

---

## PART 2 — SELLING IT (what "perfect or nobody buys it" actually requires)

1. **Staging environment** (highest priority after today: a second Railway service, deploys tested there first — prod NEVER breaks during development again).
2. **Security before ANY outside user**: today the backend is open (anyone with the URL can upload/learn; admin key is a default string in code). Needs: API auth, per-company tenancy + data isolation, CORS lock to the app domain, rate limits, volume backups, token rotation (current tokens also live in memory files — rotate before any external exposure).
3. **Accounts & tenancy**: login, company workspaces, roles (estimator/admin). Each company's corrections train THEIR instance's suggestions (the data moat is per-customer AND collective).
4. **Reliability & observability**: error tracking (Sentry), uptime alerts, job queue for big sets, structured logs. The battery test as CI on every push.
5. **Model ops**: versioned models with eval-before-deploy against a growing gold library (the estimator-gold examples are the seed), scheduled retrains, rollback.
6. **Onboarding**: a demo sandbox with a sample drawing pre-loaded (10-second wow), 3-minute tutorial, tooltips pass.
7. **Billing**: Stripe subscriptions, per-seat or per-bid metering.
8. **Legal/positioning**: "assists estimates — professional review required" disclaimer; sell as the estimator's power tool, not their replacement (also true, also what buyers want to hear).
9. **Performance**: 77-page set ≈ 4 min → background + notify; page-crop caching for zoom.

---

## PART 3 — THE SEQUENCE (each phase demoable, ordered by value ÷ risk)

**Phase 1 — Accuracy levers (1–2 sessions each):**
a) Elevation-marker reading → exact heights + the strongest sanity check
b) Dimension-string self-calibration → exact per-sheet scale
c) Auto-trim LF (corners, opening perimeters, base/top) from existing geometry
d) Opening-vs-schedule reconciliation
e) Scope→Takeoff connection (specs pre-select in-scope materials)

**Phase 2 — Deploy v12 when training completes** (steps in PLAYBOOK; beat 0.741).

**Phase 3 — Productization foundation:** staging env → security/auth → CI battery → Sentry.

**Phase 4 — Money features:** quote ingestion, labor rates, scope-letter generator, win/loss tracking.

**Phase 5 — Go-to-market:** sandbox demo, onboarding, billing, 2–3 friendly siding subs as pilots (the estimator's network), iterate on their jobs, THEN wider sales.

**North star metric: the autonomy meter.** When it holds ≥95% across 20 real bids, the pitch writes itself: "upload the set, price the bid, done."

---

## PART 4 — THE HARD-WON RULES (paid for in outages and regressions)
1. Deploy-sim in a fresh venv before ANY dependency change.
2. Never full opencv on the backend; vendored zips must use forward slashes.
3. Never PowerShell string-edits on App.jsx (UTF-8 mojibake).
4. Coverage changes interact with the welder — anti-sprawl guard + eyeball the viz before deploy.
5. The marked-path exactness (1364/2209/4630/1653) is the canary — run it after EVERY backend deploy.
6. Precision beats recall everywhere money flows: a missing face costs a click; a wrong number costs a bid.
7. The user's testing is the QA department. Ship, let him break it, fix the real break — faster than guessing at perfection.
