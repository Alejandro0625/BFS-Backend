"""BFS bid workbook generator (template v2): clone bid_template_v2.xlsx and fill
job data the way the estimator does — per-page SF as an addition formula in the
Quantity cell, Amount = Qty*Conv*Rate, proposal TOTAL linked to the estimate.

jobdata = {
  "date": "7/1/2026", "job_name": "...", "job_number": "26-262",
  "gc": {"contact": "...", "title": "...", "company": "...", "city": "...",
          "email": "...", "phone": "..."},
  "estimator": "...",
  "materials": [   # up to 6 priced lines, in bid order
     {"code": "M1", "desc": "JAMES HARDIE ...", "per_page": {7: 2049, 8: 321},
      "rate": 30.0, "unit": "sf", "sub": "1x3 pt furrning"},
  ],
  "lumps": [{"desc": "", "amount": 35000}],           # optional lump-sum rows
}
"""
import io, os, datetime
import openpyxl


def _txt(v):
    """Sanitize user-supplied text destined for a cell: openpyxl stores any string
    starting with '=' as a FORMULA (injection risk); +/-/@ trigger Excel's legacy
    formula parsing too. Prefix with a space — visually invisible, formula-dead."""
    s = str(v if v is not None else "")
    return (" " + s) if s[:1] in "=+-@" else s

_TPL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bid_template_v2.xlsx")
# material blocks live at these anchor rows in the Estimate sheet (desc row; +1 = sub row)
_MAT_ROWS = [7, 10, 13, 16, 19, 23]
_LUMP_ROW = 25


def fill_bid(jobdata):
    wb = openpyxl.load_workbook(_TPL)
    s1, est, ck = wb["Sheet1"], wb["Estimate"], wb["Siding Estimate Checklist"]
    gc = jobdata.get("gc") or {}

    # ---- proposal header
    d = jobdata.get("date")
    s1["G2"] = d if d else datetime.date.today().strftime("%m/%d/%Y")
    s1["B7"] = _txt(gc.get("contact") or "")
    if gc.get("title"):
        s1["B8"] = _txt(gc["title"])
        s1["B9"] = _txt(gc.get("company") or "")
    else:
        s1["B8"] = _txt(gc.get("company") or "")
        s1["B9"] = _txt(gc.get("city") or "")
    s1["F5"] = _txt(gc.get("email") or "")
    s1["F7"] = _txt(gc.get("phone") or "")
    s1["F9"] = _txt(jobdata.get("job_name") or "")
    s1["F11"] = _txt(jobdata.get("job_number") or "")

    mats = (jobdata.get("materials") or [])[:len(_MAT_ROWS)]

    # ---- proposal summary lines (pair materials per line, estimator style)
    def _short(m):
        return " ".join((m.get("desc") or "material").replace("\n", " ").split())[:48]
    sums = [f'{int(round(sum(m["per_page"].values()))):,} sf of {_short(m)}' for m in mats if m.get("per_page")]
    lines = []
    for i in range(0, len(sums), 2):
        pair = " & ".join(sums[i:i + 2])
        lines.append(f"Install {pair}.")
    for i, coord in enumerate(["B13", "B14", "B15"]):
        s1[coord] = _txt(lines[i]) if i < len(lines) else None

    # ---- inclusions: fixed openers + one F&I per material + fixed closers
    incl = ["Include all OSHA and fall protection compliance for the installation of panels",
            "Include all staging and lifts for the performance of work."]
    incl += [f"F&I all metal trim and accessories with {_short(m)} as specified." for m in mats]
    incl += ["Remove and dispose of all job related debris to the general contractor's dumpster.",
             "MA Sales Tax Included on all materials if applicable"]
    for i, r in enumerate(range(17, 27)):
        s1[f"A{r}"] = (i + 1) if i < len(incl) else None
        s1[f"B{r}"] = _txt(incl[i]) if i < len(incl) else None

    # ---- estimate rows: Quantity = per-page addition formula (the house style)
    from openpyxl.worksheet.datavalidation import DataValidation
    from openpyxl.comments import Comment
    for i, m in enumerate(mats):
        r = _MAT_ROWS[i]
        pages = sorted((m.get("per_page") or {}).items(), key=lambda kv: int(kv[0]))  # JSON keys are strings
        terms = "+".join(str(int(round(sf))) for _, sf in pages if sf > 0) or "0"
        est[f"A{r}"] = _txt(m.get("code") or f"M{i+1}")
        est[f"B{r}"] = _txt(m.get("desc") or "")
        est[f"C{r}"] = f"={terms}"     # always formula-style — the estimator's own habit (=85, =2049+321)
        est[f"D{r}"] = m.get("unit") or "sf"
        est[f"E{r}"] = m.get("conv", 1)
        est[f"F{r}"] = m.get("rate", 0)
        est[f"G{r}"] = f"=C{r}*E{r}*F{r}"
        if m.get("sub"):
            est[f"B{r+1}"] = _txt(m["sub"])
        # winning-rate DROPDOWN on the Rate cell: pick low/suggested/high and the
        # =C*E*F amount updates instantly; typing a custom rate stays allowed.
        sug = m.get("suggest") or {}
        vals = []
        for k in ("lo", "med", "hi"):
            v = sug.get(k)
            if isinstance(v, (int, float)) and v > 0 and v not in vals:
                vals.append(v)
        if vals:
            fm = ",".join(str(int(v)) if v == int(v) else str(v) for v in vals)
            dv = DataValidation(type="list", formula1=f'"{fm}"', allow_blank=True,
                                showErrorMessage=False, showDropDown=False)
            est.add_data_validation(dv)
            dv.add(f"F{r}")
            names = {}
            if isinstance(sug.get("med"), (int, float)): names[sug["med"]] = "Suggested"
            if isinstance(sug.get("hi"), (int, float)): names.setdefault(sug["hi"], "Higher")
            if isinstance(sug.get("lo"), (int, float)): names.setdefault(sug["lo"], "Lower")
            lab = " / ".join(f"{names.get(v, '')} ${v:g}".strip() for v in vals)
            est[f"F{r}"].comment = Comment(
                f"BFS winning rates: {lab}\nSuggested = similar winning jobs. Higher = GCs we rarely work with (room to charge). "
                f"Lower = partner GCs we protect. Custom = type any rate.\nSF quantity is locked from the takeoff — pricing never changes it.",
                "BFS Price Engine")
    for j, l in enumerate((jobdata.get("lumps") or [])[:1]):
        est[f"B{_LUMP_ROW}"] = _txt(l.get("desc") or "")
        est[f"G{_LUMP_ROW}"] = l.get("amount") or 0

    # ---- specifications block: GC + building height (labels live in col A)
    for r in range(30, 50):
        lab = str(est[f"A{r}"].value or "").lower()
        if "height" in lab and jobdata.get("height"):
            est[f"C{r}"] = _txt(f"{jobdata['height']}'")
        elif lab.strip() == "gc" and gc.get("company"):
            est[f"C{r}"] = _txt(gc["company"])
        elif lab.startswith("location") and jobdata.get("job_name"):
            est[f"C{r}"] = _txt(jobdata["job_name"])

    # ---- checklist header
    ck["C5"] = _txt(jobdata.get("estimator") or "")
    ck["C6"] = s1["G2"].value

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


if __name__ == "__main__":
    import sys, json
    out = sys.argv[1] if len(sys.argv) > 1 else "bid_test.xlsx"
    jd = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
    open(out, "wb").write(fill_bid(jd))
    print("wrote", out)
