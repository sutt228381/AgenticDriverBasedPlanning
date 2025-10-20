
import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

st.set_page_config(page_title="Agentic Driver-Based Planning — v13 (Dimensional CSV → P&L)", layout="wide")

APP_TITLE = "Agentic Driver-Based Planning — v13 (Upload CSV, Slice by Dimensions, AI Driver Hints)"
MONTHS: List[str] = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
MONTH_ALIASES = {
    "jan":"Jan","january":"Jan","01":"Jan","1":"Jan",
    "feb":"Feb","february":"Feb","02":"Feb","2":"Feb",
    "mar":"Mar","march":"Mar","03":"Mar","3":"Mar",
    "apr":"Apr","april":"Apr","04":"Apr","4":"Apr",
    "may":"May","05":"May","5":"May",
    "jun":"Jun","june":"Jun","06":"Jun","6":"Jun",
    "jul":"Jul","july":"Jul","07":"Jul","7":"Jul",
    "aug":"Aug","august":"Aug","08":"Aug","8":"Aug",
    "sep":"Sep","sept":"Sep","september":"Sep","09":"Sep","9":"Sep",
    "oct":"Oct","october":"Oct","10":"Oct",
    "nov":"Nov","november":"Nov","11":"Nov",
    "dec":"Dec","december":"Dec","12":"Dec",
}
def norm_period(x:str)->str:
    if x is None: return ""
    s=str(x).strip().lower()
    if s in MONTH_ALIASES: return MONTH_ALIASES[s]
    if s[:3] in MONTH_ALIASES: return MONTH_ALIASES[s[:3]]
    return s[:3].title()

SECTIONS = [
    ("Revenue", ["Sales","Returns & Allowances","Royalty Income"]),
    ("COGS",    ["COGS","Freight","Fulfillment"]),
    ("Opex",    ["Marketing","Payroll","G&A","Depreciation"]),
    ("Other",   ["Other Income","Interest Expense"]),
    ("Taxes",   ["Taxes"]),
]
ACCOUNT_TO_SECTION = {acc:sec for sec, accs in SECTIONS for acc in accs}
COMPUTED_ORDER = ["Gross Profit","Operating Income","Pre-Tax Income","Net Income"]

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

# ---------------- UI Header ----------------
st.title(APP_TITLE)
st.caption("Upload a comma‑delimited P&L with extra dimensions (Product, Entity, Channel, Currency). We'll ingest, total at the top, and let you slice by dimensions. Then we suggest drivers per account/dimension combo.")

# ---------------- Upload & Schema Mapping ----------------
with st.expander("1) Upload CSV & map columns", expanded=True):
    f = st.file_uploader("CSV file", type=["csv"])
    if f is not None:
        raw = pd.read_csv(f)
        st.write("Preview:", raw.head())

        cols = list(raw.columns)
        col_account = st.selectbox("Account column", cols)
        col_period  = st.selectbox("Period column (month name/number)", cols)
        col_value   = st.selectbox("Value/Amount column", cols)
        col_entity  = st.selectbox("Entity column (optional)", ["<none>"] + cols)
        col_product = st.selectbox("Product column (optional)", ["<none>"] + cols)
        col_channel = st.selectbox("Channel column (optional)", ["<none>"] + cols)
        col_currency= st.selectbox("Currency column (optional)", ["<none>"] + cols)

        # normalize and store
        df = raw.copy()
        df["Account"] = df[col_account].astype(str).str.strip()
        df["Period"]  = df[col_period].apply(norm_period)
        df["Value"]   = pd.to_numeric(df[col_value], errors="coerce").fillna(0.0)
        if col_entity != "<none>":  df["Entity"]= raw[col_entity].astype(str)
        else:                        df["Entity"]= "All"
        if col_product != "<none>": df["Product"]= raw[col_product].astype(str)
        else:                        df["Product"]= "All"
        if col_channel != "<none>": df["Channel"]= raw[col_channel].astype(str)
        else:                        df["Channel"]= "All"
        if col_currency != "<none>":df["Currency"]= raw[col_currency].astype(str)
        else:                        df["Currency"]= "USD"

        # Map to standard accounts (if novel accounts appear, keep them in Opex as default)
        df["Section"] = df["Account"].map(lambda a: ACCOUNT_TO_SECTION.get(a, "Opex"))

        # Keep only known months
        df = df[df["Period"].isin(MONTHS)].copy()

        st.session_state["uploaded_df"] = df

# ---------------- Filters & Actuals Cutoff ----------------
if "uploaded_df" in st.session_state:
    src = st.session_state["uploaded_df"]
    with st.expander("2) Filters & settings", expanded=True):
        ent = st.multiselect("Entity", sorted(src["Entity"].unique().tolist()), default=sorted(src["Entity"].unique().tolist()))
        prod= st.multiselect("Product", sorted(src["Product"].unique().tolist()), default=sorted(src["Product"].unique().tolist()))
        chan= st.multiselect("Channel", sorted(src["Channel"].unique().tolist()), default=sorted(src["Channel"].unique().tolist()))
        tgt_ccy = st.selectbox("Target currency", ["USD","EUR","GBP"], index=0, help="Simple 1:1 unless you enter an override below")
        fx_override = st.number_input("FX override (multiply values)", value=1.0, step=0.01, help="Set e.g. 1.08 for EUR→USD")

        cutoff = st.selectbox("Actuals cutoff month (inclusive — locked)", MONTHS, index=2)  # default Mar
        locked_set = set(MONTHS[:MONTHS.index(cutoff)+1])

    # slice
    sliced = src[src["Entity"].isin(ent) & src["Product"].isin(prod) & src["Channel"].isin(chan)].copy()
    # currency handling (simple, user-controlled)
    sliced["Value"] = sliced["Value"] * fx_override

    # pivot to month-wide by Account
    by_acc = sliced.pivot_table(index=["Section","Account"], columns="Period", values="Value", aggfunc="sum").reindex(columns=MONTHS, fill_value=0.0)

    # seed cube structure for CALC/ADJ/TOTAL by leaf
    def seed_from_actuals(by_acc: pd.DataFrame)->pd.DataFrame:
        rows=[]
        order=0
        for sec, accs in SECTIONS:
            for acc in accs:
                base = by_acc.loc[(sec, acc)] if (sec,acc) in by_acc.index else pd.Series(0.0,index=MONTHS)
                for comp in ["CALC","MANUAL_ADJ","TOTAL"]:
                    row = {"Order":order,"Section":sec,"Account":acc,"Component":comp}
                    for m in MONTHS:
                        if comp=="CALC":  row[m]= float(base.get(m,0.0))
                        else:             row[m]= 0.0
                    rows.append(row)
                order+=1
        return pd.DataFrame(rows)

    if "cube" not in st.session_state:
        st.session_state["cube"] = seed_from_actuals(by_acc)

    # Show totals top-line for context
    st.subheader("Top-line totals (current slice)")
    totals = by_acc.groupby(level=0).sum()
    st.dataframe(totals.style.format("{:,.0f}"), use_container_width=True)

    # ---------------- Insights: heuristic driver suggestions ----------------
    with st.expander("3) Driver suggestions (heuristic)", expanded=False):
        st.write("We infer drivers per Account × Channel/Product combination based on names and patterns (no external data).")
        # simple naming heuristics
        def suggest(acc:str, channel:str, product:str)->str:
            a=acc.lower(); ch=str(channel).lower(); pr=str(product).lower()
            if "freight" in a or "shipping" in a or "postage" in a or "magazine" in ch:
                return "OIL_LINKED_FREIGHT"
            if a in ["payroll","g&a","g&a","general & administrative","general and administrative"] or "salary" in a:
                return "CPI_INDEXED"
            if "marketing" in a or "ad" in a:
                return "PCT_OF_SALES"
            if "return" in a:
                return "PCT_OF_SALES (negative)"
            if "royalty" in a:
                return "PY_RATIO_SALES"
            if acc=="COGS":
                return "PCT_GROWTH"
            if acc=="Sales":
                return "MANUAL"
            return "MANUAL"
        # build a small table by channel/product showing suggestions
        dim = sliced.groupby(["Account","Channel","Product"])["Value"].sum().reset_index()
        dim["Suggested Driver"] = dim.apply(lambda r: suggest(r["Account"], r["Channel"], r["Product"]), axis=1)
        st.dataframe(dim.sort_values(["Account","Channel","Product"]).reset_index(drop=True), use_container_width=True)
        st.caption("Example: Revenue in **Magazine** channel flagged for **OIL_LINKED_FREIGHT** due to postage/shipping sensitivity.")

    # ---------------- Hierarchical months-across grid ----------------
    st.subheader("4) P&L Grid — Months Across (CALC + ADJ → TOTAL)")

    cube = st.session_state["cube"]
    # recalc TOTAL
    def recalc(df: pd.DataFrame)->pd.DataFrame:
        out = df.copy()
        for m in MONTHS:
            out.loc[out["Component"]=="TOTAL", m] = (
                out.loc[out["Component"]=="CALC", m].values + out.loc[out["Component"]=="MANUAL_ADJ", m].values
            )
        return out
    cube = recalc(cube)

    # build display
    def section_totals(df_total: pd.DataFrame):
        res = {m:{} for m in MONTHS}
        for sec,_ in SECTIONS:
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

    if "expanded" not in st.session_state:
        st.session_state["expanded"] = {sec: True for sec,_ in SECTIONS}
    expanded = st.session_state["expanded"]

    with st.sidebar:
        st.header("Expand/Collapse")
        c1,c2 = st.columns(2)
        if c1.button("Expand all"):  expanded = {sec:True for sec,_ in SECTIONS}
        if c2.button("Collapse all"):expanded = {sec:False for sec,_ in SECTIONS}
        for sec,_ in SECTIONS:
            expanded[sec] = st.toggle(sec, value=expanded[sec], key=f"exp_{sec}")

    def build_display(df: pd.DataFrame, expanded_map: Dict[str,bool]) -> pd.DataFrame:
        df = recalc(df)
        rows=[]
        sec_sub = section_totals(df[df.Component=="TOTAL"])
        for sec, accounts in SECTIONS:
            parent = {"Indent":0, "RowType":"PARENT", "Line": sec}
            for m in MONTHS:
                parent[m] = float(sec_sub[m].get(sec,0.0))
            rows.append(parent)
            if expanded_map.get(sec, True):
                for acc in accounts:
                    # TOTAL
                    total_row = {"Indent":1, "RowType":"LEAF_TOTAL", "Line": acc}
                    mask_tot = (df.Account==acc)&(df.Component=="TOTAL")
                    for m in MONTHS: total_row[m] = float(df.loc[mask_tot, m].sum())
                    rows.append(total_row)
                    # CALC
                    calc_row  = {"Indent":2, "RowType":"LEAF_CALC", "Line": f"{acc} · CALC"}
                    mask_calc = (df.Account==acc)&(df.Component=="CALC")
                    for m in MONTHS: calc_row[m] = float(df.loc[mask_calc, m].sum())
                    rows.append(calc_row)
                    # ADJ
                    adj_row   = {"Indent":2, "RowType":"LEAF_ADJ", "Line": f"{acc} · ADJ"}
                    mask_adj  = (df.Account==acc)&(df.Component=="MANUAL_ADJ")
                    for m in MONTHS: adj_row[m] = float(df.loc[mask_adj, m].sum())
                    rows.append(adj_row)
        # computed bottom
        for name in ["Gross Profit","Operating Income","Pre-Tax Income","Net Income"]:
            row = {"Indent":0, "RowType":"COMPUTED", "Line": name}
            for m in MONTHS:
                sec_map = { s: float(df[(df.Component=="TOTAL")&(df.Section==s)][m].sum()) for s,_ in SECTIONS }
                row[m] = computed_lines(sec_map)[name]
            rows.append(row)
        disp = pd.DataFrame(rows)
        def label(r):
            bullet = ">> " if r["RowType"]=="PARENT" else ("  " * r["Indent"] + "- ")
            return ("  " * r["Indent"]) + bullet + r["Line"]
        disp["Account"] = disp.apply(label, axis=1)
        return disp[["Account"] + MONTHS + ["RowType","Line","Indent"]]

    display_df = build_display(cube, expanded)

    # lock months up to cutoff
    col_cfg = {}
    for m in MONTHS:
        if m in locked_set:
            col_cfg[m] = st.column_config.NumberColumn(disabled=True)
        else:
            col_cfg[m] = st.column_config.NumberColumn()

    edited = st.data_editor(
        display_df, use_container_width=True, num_rows="fixed", hide_index=True, column_config=col_cfg
    )

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
                if m in locked_set:  # keep actuals
                    continue
                val = after_disp.loc[idx, m]
                if pd.isna(val): val = before_disp.loc[idx, m]
                df.loc[(df.Account==acc)&(df.Component==comp), m] = float(val)
        return df

    if st.button("Recalculate & Save", type="primary"):
        updated = apply_edits(cube, display_df, edited)
        updated = recalc(updated)
        st.session_state["cube"] = updated
        st.success("Recalculated. Parents & computed updated.")

else:
    st.info("Upload a CSV to begin. Expect columns for Account, Period (month), Value, and optional Entity/Product/Channel/Currency.")
