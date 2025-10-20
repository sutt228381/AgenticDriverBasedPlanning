
import streamlit as st
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List

st.set_page_config(page_title="Agentic Driver-Based Planning — v12.1 (Months Across, Jan–Mar Locked)", layout="wide")

APP_TITLE = "Agentic Driver-Based Planning — Hierarchical P&L v12.1 (Months Across)"
MONTHS: List[str] = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
LOCKED = {"Jan","Feb","Mar"}  # actualized (read-only)

SECTIONS = [
    ("Revenue", ["Sales","Returns & Allowances","Royalty Income"]),
    ("COGS",    ["COGS","Freight","Fulfillment"]),
    ("Opex",    ["Marketing","Payroll","G&A","Depreciation"]),
    ("Other",   ["Other Income","Interest Expense"]),
    ("Taxes",   ["Taxes"]),
]

@dataclass
class LineMeta:
    driver: str
    param: str

DEFAULT_DRIVERS: Dict[str, LineMeta] = {
    "Sales": LineMeta("MANUAL","Amount"),
    "Returns & Allowances": LineMeta("PCT_OF_SALES","% of Sales"),
    "Royalty Income": LineMeta("PY_RATIO_SALES",""),
    "COGS": LineMeta("PCT_GROWTH","Growth%"),
    "Freight": LineMeta("OIL_LINKED_FREIGHT","% of Sales"),
    "Fulfillment": LineMeta("PCT_OF_SALES","% of Sales"),
    "Marketing": LineMeta("PCT_OF_SALES","% of Sales"),
    "Payroll": LineMeta("CPI_INDEXED",""),
    "G&A": LineMeta("CPI_INDEXED",""),
    "Depreciation": LineMeta("MANUAL","Amount"),
    "Other Income": LineMeta("MANUAL","Amount"),
    "Interest Expense": LineMeta("MANUAL","Amount"),
    "Taxes": LineMeta("MANUAL","Amount"),
}

COMPUTED_ORDER = ["Gross Profit","Operating Income","Pre-Tax Income","Net Income"]

# ---- Seed actuals for Jan–Mar across multiple accounts (demo) ----
ACTUALS = {
    "Sales":               {"Jan": 120000.0, "Feb": 118000.0, "Mar": 119500.0},
    "Returns & Allowances":{"Jan":  -2400.0, "Feb":  -2360.0, "Mar":  -2390.0},
    "Royalty Income":      {"Jan":   3100.0, "Feb":   3200.0, "Mar":   3150.0},
    "COGS":                {"Jan": -55000.0, "Feb": -60500.0, "Mar": -58000.0},
    "Freight":             {"Jan":  -8000.0, "Feb":  -8300.0, "Mar":  -8200.0},
    "Fulfillment":         {"Jan":  -5000.0, "Feb":  -5200.0, "Mar":  -5100.0},
    "Marketing":           {"Jan":  -7000.0, "Feb":  -6800.0, "Mar":  -6900.0},
    "Payroll":             {"Jan": -12000.0, "Feb": -12150.0, "Mar": -12200.0},
    "G&A":                 {"Jan":  -6000.0, "Feb":  -6100.0, "Mar":  -6050.0},
    "Depreciation":        {"Jan":  -1500.0, "Feb":  -1500.0, "Mar":  -1500.0},
    "Other Income":        {"Jan":   1000.0, "Feb":   1000.0, "Mar":   1000.0},
    "Interest Expense":    {"Jan":  -4200.0, "Feb":  -4200.0, "Mar":  -4200.0},
    "Taxes":               {"Jan":  -3500.0, "Feb":  -3600.0, "Mar":  -3550.0},
}

def seed_cube()->pd.DataFrame:
    rows = []
    order = 0
    for sec, accounts in SECTIONS:
        for acc in accounts:
            meta = DEFAULT_DRIVERS[acc]
            for comp in ["CALC","MANUAL_ADJ","TOTAL"]:
                row = {
                    "Order": order, "Section": sec, "Account": acc, "Component": comp,
                    "Driver": meta.driver if comp=="CALC" else "MANUAL",
                    "Param":  meta.param if comp=="CALC" else "Adj",
                }
                for m in MONTHS:
                    row[m] = 0.0
                rows.append(row)
            order += 1
    df = pd.DataFrame(rows)
    # Seed Jan–Mar CALC from ACTUALS for all leaf accounts
    for acc, months in ACTUALS.items():
        for m, val in months.items():
            df.loc[(df.Account==acc)&(df.Component=="CALC"), m] = float(val)
    return df

def recalc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # TOTAL = CALC + MANUAL_ADJ (for all months)
    for m in MONTHS:
        out.loc[out.Component=="TOTAL", m] = (
            out.loc[out.Component=="CALC", m].values + out.loc[out.Component=="MANUAL_ADJ", m].values
        )
    return out

def section_totals(df_total: pd.DataFrame):
    # returns {month: {section: value}}
    res = {m:{} for m in MONTHS}
    for sec, _ in SECTIONS:
        mask = (df_total["Component"]=="TOTAL") & (df_total["Section"]==sec)
        for m in MONTHS:
            res[m][sec] = float(df_total.loc[mask, m].sum())
    return res

def computed_lines(sec_map):
    out = {}
    out["Gross Profit"]     = sec_map.get("Revenue",0.0) - sec_map.get("COGS",0.0)
    out["Operating Income"] = out["Gross Profit"] - sec_map.get("Opex",0.0)
    out["Pre-Tax Income"]   = out["Operating Income"] + sec_map.get("Other",0.0)
    out["Net Income"]       = out["Pre-Tax Income"] - sec_map.get("Taxes",0.0)
    return out

def build_display(df: pd.DataFrame, expanded: Dict[str,bool]) -> pd.DataFrame:
    df = recalc(df)
    rows = []
    # per-section subtotals
    sec_sub = section_totals(df[df.Component=="TOTAL"])
    for sec, accounts in SECTIONS:
        parent = {"Indent":0, "RowType":"PARENT", "Line": sec, "Driver":"", "Param":""}
        for m in MONTHS:
            parent[m] = float(sec_sub[m].get(sec,0.0))
        rows.append(parent)

        if expanded.get(sec, True):
            for acc in accounts:
                # TOTAL row for account
                total_row = {"Indent":1, "RowType":"LEAF_TOTAL", "Line": acc, "Driver":"", "Param":""}
                mask_tot = (df.Account==acc)&(df.Component=="TOTAL")
                for m in MONTHS:
                    total_row[m] = float(df.loc[mask_tot, m].sum())
                rows.append(total_row)

                # CALC row
                calc_row = {"Indent":2, "RowType":"LEAF_CALC", "Line": f"{acc} · CALC", "Driver":"", "Param":""}
                mask_calc = (df.Account==acc)&(df.Component=="CALC")
                for m in MONTHS:
                    calc_row[m] = float(df.loc[mask_calc, m].sum())
                rows.append(calc_row)

                # ADJ row
                adj_row = {"Indent":2, "RowType":"LEAF_ADJ", "Line": f"{acc} · ADJ", "Driver":"", "Param":""}
                mask_adj = (df.Account==acc)&(df.Component=="MANUAL_ADJ")
                for m in MONTHS:
                    adj_row[m] = float(df.loc[mask_adj, m].sum())
                rows.append(adj_row)

    # computed lines at bottom
    for name in ["Gross Profit","Operating Income","Pre-Tax Income","Net Income"]:
        row = {"Indent":0, "RowType":"COMPUTED", "Line": name, "Driver":"", "Param":""}
        for m in MONTHS:
            sec_map = { s: float(df[(df.Component=="TOTAL")&(df.Section==s)][m].sum()) for s,_ in SECTIONS }
            row[m] = computed_lines(sec_map)[name]
        rows.append(row)

    disp = pd.DataFrame(rows)
    # render label with indentation
    def label(r):
        bullet = ">> " if r["RowType"]=="PARENT" else ("  " * r["Indent"] + "- ")
        return ("  " * r["Indent"]) + bullet + r["Line"]
    disp["Account"] = disp.apply(label, axis=1)
    cols = ["Account"] + MONTHS + ["RowType","Line","Indent"]
    return disp[cols]

st.title(APP_TITLE)
st.caption("Rows = Accounts (hierarchical). Columns = Months. Parents sum children. CALC+ADJ -> TOTAL. Jan–Mar are seeded as ACTUALS and hard-locked (non-editable).")

# init session
if "cube" not in st.session_state:
    st.session_state["cube"] = seed_cube()
if "expanded" not in st.session_state:
    st.session_state["expanded"] = {sec: True for sec,_ in SECTIONS}

cube = st.session_state["cube"]
expanded = st.session_state["expanded"]

# Sidebar controls
with st.sidebar:
    st.header("View")
    colA, colB = st.columns(2)
    if colA.button("Expand all"):
        for sec,_ in SECTIONS: expanded[sec] = True
    if colB.button("Collapse all"):
        for sec,_ in SECTIONS: expanded[sec] = False
    for sec,_ in SECTIONS:
        expanded[sec] = st.toggle(sec, value=expanded[sec], key=f"exp_{sec}")

# Build table
display_df = build_display(cube, expanded)

# Column config: explicitly disable Jan–Mar across all rows (visual + interaction lock)
col_cfg = {}
for m in MONTHS:
    if m in LOCKED:
        col_cfg[m] = st.column_config.NumberColumn(disabled=True)
    else:
        col_cfg[m] = st.column_config.NumberColumn()

edited = st.data_editor(
    display_df,
    use_container_width=True,
    num_rows="fixed",
    hide_index=True,
    column_config=col_cfg
)

# Apply edits back to the cube with rules:
# - Only LEAF_CALC and LEAF_ADJ rows are sources
# - Months in LOCKED are ignored (kept as ACTUALS)
def apply_edits(original: pd.DataFrame, before_disp: pd.DataFrame, after_disp: pd.DataFrame) -> pd.DataFrame:
    df = original.copy()
    for idx in range(len(after_disp)):
        rt = after_disp.loc[idx, "RowType"]
        label = after_disp.loc[idx, "Line"]
        if rt not in ("LEAF_CALC","LEAF_ADJ"):
            continue
        if rt == "LEAF_CALC":
            acc = label.replace(" · CALC","")
            comp = "CALC"
        else:
            acc = label.replace(" · ADJ","")
            comp = "MANUAL_ADJ"
        for m in MONTHS:
            if m in LOCKED:
                continue  # keep actuals
            val = after_disp.loc[idx, m]
            if pd.isna(val):
                val = before_disp.loc[idx, m]
            df.loc[(df.Account==acc)&(df.Component==comp), m] = float(val)
    return df

if st.button("Recalculate & Save", type="primary"):
    updated = apply_edits(cube, display_df, edited)
    updated = recalc(updated)
    st.session_state["cube"] = updated
    st.success("Recalculated. Parents & computed updated. Jan–Mar remained locked as actuals.")
