
import io
import math
import streamlit as st
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List

st.set_page_config(page_title="Agentic Driver-Based Planning — v13.2 (Drivers per Combo)", layout="wide")

APP_TITLE = "Agentic Driver-Based Planning — v13.2 (Upload → Slice → Choose Drivers per Combo → Apply)"
MONTHS: List[str] = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
Q1 = {"Jan","Feb","Mar"}

# ---------------- Hierarchy ----------------
SECTIONS = [
    ("Revenue", ["Sales","Returns & Allowances","Royalty Income"]),
    ("COGS",    ["COGS","Freight","Fulfillment"]),
    ("Opex",    ["Marketing","Payroll","G&A","Depreciation"]),
    ("Other",   ["Other Income","Interest Expense"]),
    ("Taxes",   ["Taxes"]),
]
ACCOUNT_TO_SECTION = {acc:sec for sec, accs in SECTIONS for acc in accs}

@dataclass
class LineMeta:
    driver: str
    param: float

DEFAULT_DRIVERS: Dict[str, LineMeta] = {
    "Sales": LineMeta("MANUAL", 0.0),
    "Returns & Allowances": LineMeta("PCT_OF_SALES", -0.02),
    "Royalty Income": LineMeta("PY_RATIO_SALES", 0.0),
    "COGS": LineMeta("PCT_OF_SALES", -0.45),
    "Freight": LineMeta("OIL_LINKED_FREIGHT", -0.02),
    "Fulfillment": LineMeta("PCT_OF_SALES", -0.015),
    "Marketing": LineMeta("PCT_OF_SALES", -0.07),
    "Payroll": LineMeta("CPI_INDEXED", 0.02),
    "G&A": LineMeta("CPI_INDEXED", 0.02),
    "Depreciation": LineMeta("MANUAL", 0.0),
    "Other Income": LineMeta("MANUAL", 0.0),
    "Interest Expense": LineMeta("MANUAL", 0.0),
    "Taxes": LineMeta("PCT_OF_SALES", -0.025),
}

# ---------------- Utils ----------------
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

def computed_lines(sec_map):
    out = {}
    out["Gross Profit"]     = sec_map.get("Revenue",0.0) - sec_map.get("COGS",0.0)
    out["Operating Income"] = out["Gross Profit"] - sec_map.get("Opex",0.0)
    out["Pre-Tax Income"]   = out["Operating Income"] + sec_map.get("Other",0.0)
    out["Net Income"]       = out["Pre-Tax Income"] - sec_map.get("Taxes",0.0)
    return out

# ---------------- Header ----------------
st.title(APP_TITLE)
st.caption("Upload a dimensional CSV → Map → Slice → Choose per-combination drivers → Apply to CALC for forecast months (post actuals cutoff).")

# ---------------- Upload & Map ----------------
with st.expander("1) Upload CSV & map columns", expanded=True):
    uf = st.file_uploader("CSV file", type=["csv"])
    # Optional autoload sample
    if uf is None and st.button("Load built-in sample"):
        sample = st.session_state.get("_SAMPLE_EMBED", "")
        uf = io.StringIO(sample) if sample else None
    if uf is not None:
        raw = pd.read_csv(uf)
        st.write("Preview", raw.head())
        cols = list(raw.columns)
        col_account = st.selectbox("Account column", cols, index=cols.index("Account") if "Account" in cols else 0)
        col_period  = st.selectbox("Period column", cols, index=cols.index("Period") if "Period" in cols else 0)
        col_value   = st.selectbox("Value/Amount column", cols, index=cols.index("Value") if "Value" in cols else 0)
        col_entity  = st.selectbox("Entity column", ["<none>"]+cols, index=(cols.index("Entity")+1) if "Entity" in cols else 0)
        col_product = st.selectbox("Product column", ["<none>"]+cols, index=(cols.index("Product")+1) if "Product" in cols else 0)
        col_channel = st.selectbox("Channel column", ["<none>"]+cols, index=(cols.index("Channel")+1) if "Channel" in cols else 0)
        col_currency= st.selectbox("Currency column", ["<none>"]+cols, index=(cols.index("Currency")+1) if "Currency" in cols else 0)

        df = raw.copy()
        df["Account"] = df[col_account].astype(str).str.strip()
        df["Period"]  = df[col_period].apply(norm_period)
        df["Value"]   = pd.to_numeric(df[col_value], errors="coerce").fillna(0.0)
        df["Entity"]  = raw[col_entity].astype(str) if col_entity!="<none>" else "All"
        df["Product"] = raw[col_product].astype(str) if col_product!="<none>" else "All"
        df["Channel"] = raw[col_channel].astype(str) if col_channel!="<none>" else "All"
        df["Currency"]= raw[col_currency].astype(str) if col_currency!="<none>" else "USD"
        df["Section"] = df["Account"].map(lambda a: ACCOUNT_TO_SECTION.get(a, "Opex"))
        df = df[df["Period"].isin(MONTHS)].copy()

        st.session_state["uploaded_df"] = df

if "uploaded_df" not in st.session_state:
    st.info("Upload a CSV to begin.")
    st.stop()

src = st.session_state["uploaded_df"]

# ---------------- Filters & Actuals Cutoff ----------------
with st.expander("2) Filters & actuals settings", expanded=True):
    ent = st.multiselect("Entity", sorted(src["Entity"].unique().tolist()), default=sorted(src["Entity"].unique().tolist()))
    prod= st.multiselect("Product", sorted(src["Product"].unique().tolist()), default=sorted(src["Product"].unique().tolist()))
    chan= st.multiselect("Channel", sorted(src["Channel"].unique().tolist()), default=sorted(src["Channel"].unique().tolist()))
    cutoff = st.selectbox("Actuals cutoff (locked inclusive)", MONTHS, index=2)  # default Mar
    locked_set = set(MONTHS[:MONTHS.index(cutoff)+1])

# Slice
sliced = src[src["Entity"].isin(ent) & src["Product"].isin(prod) & src["Channel"].isin(chan)].copy()

# Pivot totals by account
by_acc = sliced.pivot_table(index=["Section","Account"], columns="Period", values="Value", aggfunc="sum").reindex(columns=MONTHS, fill_value=0.0)

# ---------------- Seed cube CALC/ADJ/TOTAL from actuals ----------------
def seed_from_actuals(by_acc: pd.DataFrame)->pd.DataFrame:
    rows=[]
    order=0
    for sec, accs in SECTIONS:
        for acc in accs:
            base = by_acc.loc[(sec, acc)] if (sec,acc) in by_acc.index else pd.Series(0.0,index=MONTHS)
            for comp in ["CALC","MANUAL_ADJ","TOTAL"]:
                row = {"Order":order,"Section":sec,"Account":acc,"Component":comp}
                for m in MONTHS:
                    row[m] = float(base.get(m,0.0)) if comp=="CALC" else 0.0
                rows.append(row)
            order+=1
    return pd.DataFrame(rows)

if "cube" not in st.session_state:
    st.session_state["cube"] = seed_from_actuals(by_acc)
cube = st.session_state["cube"]

# ---------------- Driver suggestions & selection per combo ----------------
st.subheader("3) Driver suggestions & selection (per Account × Channel × Product)")

# Build base table of combos present in the current slice for known accounts
dim = sliced[sliced["Account"].isin(ACCOUNT_TO_SECTION.keys())]
combos = dim.groupby(["Account","Channel","Product"])["Value"].sum().reset_index()

# heuristic suggestions
def suggest(acc:str, channel:str, product:str)->str:
    a=acc.lower(); ch=str(channel).lower(); pr=str(product).lower()
    if "freight" in a or "postage" in a or "magazine" in ch:
        return "OIL_LINKED_FREIGHT"
    if a in ["payroll","g&a","general & administrative"] or "salary" in a:
        return "CPI_INDEXED"
    if "marketing" in a or "ad" in a:
        return "PCT_OF_SALES"
    if "return" in a:
        return "PCT_OF_SALES"
    if "royalty" in a:
        return "PY_RATIO_SALES"
    if acc=="COGS":
        return "PCT_OF_SALES"
    if acc=="Sales":
        return "PCT_GROWTH"
    return "MANUAL"

# Build/editable config table
choices = ["MANUAL","PCT_GROWTH","PCT_OF_SALES","PY_RATIO_SALES","CPI_INDEXED","OIL_LINKED_FREIGHT","FX_CONVERTED_SALES"]
default_params = {
    "MANUAL":0.0,
    "PCT_GROWTH":0.03,
    "PCT_OF_SALES":0.02,
    "PY_RATIO_SALES":0.00,
    "CPI_INDEXED":0.02,
    "OIL_LINKED_FREIGHT":0.015,
    "FX_CONVERTED_SALES":1.05,
}
config_df = combos.copy()
config_df["Suggested"] = config_df.apply(lambda r: suggest(r["Account"], r["Channel"], r["Product"]), axis=1)
config_df["Driver"] = config_df["Account"].map(lambda a: DEFAULT_DRIVERS.get(a, LineMeta("MANUAL",0.0)).driver)
config_df["Param"]  = config_df["Account"].map(lambda a: DEFAULT_DRIVERS.get(a, LineMeta("MANUAL",0.0)).param)

# Session persistence for config
if "driver_cfg" not in st.session_state:
    st.session_state["driver_cfg"] = config_df
else:
    # Merge to preserve user edits when slice changes
    prev = st.session_state["driver_cfg"]
    # left join on combo keys
    merged = pd.merge(config_df, prev[["Account","Channel","Product","Driver","Param"]],
                      on=["Account","Channel","Product"], how="left", suffixes=("","_old"))
    merged["Driver"] = merged["Driver_old"].fillna(merged["Driver"])
    merged["Param"]  = merged["Param_old"].fillna(merged["Param"])
    config_df = merged.drop(columns=["Driver_old","Param_old"])
    st.session_state["driver_cfg"] = config_df

# Editable grid (driver + param)
edited_cfg = st.data_editor(
    st.session_state["driver_cfg"],
    use_container_width=True,
    num_rows="fixed",
    hide_index=True,
    column_config={
        "Driver": st.column_config.SelectboxColumn(options=choices),
        "Param": st.column_config.NumberColumn(help="Meaning depends on driver: e.g., 0.02=2% for PCT_OF_SALES; 1.05=+5% for FX; 0.02=2% CPI YoY"),
    }
)
st.session_state["driver_cfg"] = edited_cfg

# ---------------- Global knobs for driver application ----------------
with st.expander("Global assumptions for drivers", expanded=False):
    c1,c2,c3 = st.columns(3)
    with c1: cpi_yoy = st.number_input("CPI YoY (for CPI_INDEXED)", value=0.02, step=0.005, format="%.3f")
    with c2: oil_ratio = st.number_input("Oil/Postage index ratio (for OIL_LINKED_FREIGHT)", value=1.00, step=0.05, format="%.2f")
    with c3: fx_rate = st.number_input("FX rate (for FX_CONVERTED_SALES)", value=1.05, step=0.01, format="%.2f")

# ---------------- Build hierarchical months-across display ----------------
def recalc(df: pd.DataFrame)->pd.DataFrame:
    out = df.copy()
    for m in MONTHS:
        out.loc[out["Component"]=="TOTAL", m] = (
            out.loc[out["Component"]=="CALC", m].values + out.loc[out["Component"]=="MANUAL_ADJ", m].values
        )
    return out

def section_totals(df_total: pd.DataFrame):
    res = {m:{} for m in MONTHS}
    for sec,_ in SECTIONS:
        mask = (df_total["Component"]=="TOTAL") & (df_total["Section"]==sec)
        for m in MONTHS:
            res[m][sec] = float(df_total.loc[mask, m].sum())
    return res

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
                total_row={"Indent":1,"RowType":"LEAF_TOTAL","Line":acc}
                mask_tot=(df.Account==acc)&(df.Component=="TOTAL")
                for m in MONTHS: total_row[m]=float(df.loc[mask_tot, m].sum())
                rows.append(total_row)

                calc_row={"Indent":2,"RowType":"LEAF_CALC","Line":f"{acc} · CALC"}
                mask_calc=(df.Account==acc)&(df.Component=="CALC")
                for m in MONTHS: calc_row[m]=float(df.loc[mask_calc, m].sum())
                rows.append(calc_row)

                adj_row={"Indent":2,"RowType":"LEAF_ADJ","Line":f"{acc} · ADJ"}
                mask_adj=(df.Account==acc)&(df.Component=="MANUAL_ADJ")
                for m in MONTHS: adj_row[m]=float(df.loc[mask_adj, m].sum())
                rows.append(adj_row)
    # computed bottom
    for name in ["Gross Profit","Operating Income","Pre-Tax Income","Net Income"]:
        row={"Indent":0,"RowType":"COMPUTED","Line":name}
        for m in MONTHS:
            sec_map = { s: float(df[(df.Component=="TOTAL")&(df.Section==s)][m].sum()) for s,_ in SECTIONS }
            row[m]=computed_lines(sec_map)[name]
        rows.append(row)
    disp = pd.DataFrame(rows)
    def label(r):
        bullet = ">> " if r["RowType"]=="PARENT" else ("  " * r["Indent"] + "- ")
        return ("  " * r["Indent"]) + bullet + r["Line"]
    disp["Account"] = disp.apply(label, axis=1)
    return disp[["Account"] + MONTHS + ["RowType","Line","Indent"]]

# Expand/collapse controls
if "expanded" not in st.session_state:
    st.session_state["expanded"] = {sec: True for sec,_ in SECTIONS}
expanded = st.session_state["expanded"]
with st.sidebar:
    st.header("Sections")
    c1,c2 = st.columns(2)
    if c1.button("Expand all"):
        for sec,_ in SECTIONS: expanded[sec]=True
    if c2.button("Collapse all"):
        for sec,_ in SECTIONS: expanded[sec]=False
    for sec,_ in SECTIONS:
        expanded[sec] = st.toggle(sec, value=expanded[sec], key=f"exp_{sec}")

display_df = build_display(cube, expanded)

# Lock up to cutoff
col_cfg = {m: st.column_config.NumberColumn(disabled=(m in locked_set)) for m in MONTHS}

edited = st.data_editor(
    display_df, use_container_width=True, num_rows="fixed", hide_index=True, column_config=col_cfg
)

# ---------------- Apply user-selected drivers to CALC (Apr–Dec) ----------------
def apply_driver_calc(cube_df: pd.DataFrame, cfg: pd.DataFrame) -> pd.DataFrame:
    df = cube_df.copy()
    # Helper to get Sales TOTAL for a month (per current slice aggregate)
    sales_tot = {m: float(df[(df["Account"]=="Sales") & (df["Component"]=="TOTAL")][m].sum()) for m in MONTHS}
    # Prior-year ratio baseline: avg Q1 ratio of account to Sales (avoid div/0)
    def py_ratio(acc: str) -> float:
        acc_q1 = sum(float(df[(df["Account"]==acc) & (df["Component"]=="CALC")][m].sum()) for m in ["Jan","Feb","Mar"])
        sales_q1 = sum(float(df[(df["Account"]=="Sales") & (df["Component"]=="CALC")][m].sum()) for m in ["Jan","Feb","Mar"])
        if sales_q1 == 0.0: return 0.0
        return acc_q1 / sales_q1

    ratio_cache = {}
    for _, row in cfg.iterrows():
        acc = row["Account"]; channel = row["Channel"]; product = row["Product"]
        drv = str(row.get("Driver","MANUAL")).upper()
        prm = float(row.get("Param", 0.0))

        # For simplicity in this MVP, we apply drivers at the aggregate (current slice) level per account,
        # not per specific channel/product split. Extension: scope df updates with filters.
        for m in MONTHS:
            if m in locked_set:
                continue  # keep actuals
            mask_calc = (df["Account"]==acc) & (df["Component"]=="CALC")
            base_prev = float(df.loc[mask_calc, m].sum())  # current value (will be overwritten)
            # Fetch previous month for growth/indexing
            prev_idx = MONTHS.index(m) - 1
            prev_month = MONTHS[prev_idx] if prev_idx >= 0 else None
            prev_val = float(df.loc[mask_calc, prev_month].sum()) if prev_month else base_prev

            val = base_prev
            if drv == "MANUAL":
                val = base_prev
            elif drv == "PCT_GROWTH":
                val = prev_val * (1.0 + prm)
            elif drv == "PCT_OF_SALES":
                val = sales_tot[m] * prm
            elif drv == "PY_RATIO_SALES":
                if acc not in ratio_cache:
                    ratio_cache[acc] = py_ratio(acc)
                val = sales_tot[m] * ratio_cache[acc] * (1.0 + prm)
            elif drv == "CPI_INDEXED":
                val = prev_val * (1.0 + prm)
            elif drv == "OIL_LINKED_FREIGHT":
                val = sales_tot[m] * prm * st.session_state.get("_oil_ratio", 1.0)
            elif drv == "FX_CONVERTED_SALES":
                val = prev_val * prm
            df.loc[mask_calc, m] = float(val)

    return df

if st.button("Apply selected drivers to CALC (forecast months)"):
    st.session_state["_oil_ratio"] = st.session_state.get("_oil_ratio", 1.0)
    # Use global knobs to update the config rows that depend on them
    cfg = st.session_state["driver_cfg"].copy()
    cfg.loc[cfg["Driver"]=="CPI_INDEXED","Param"] = cpi_yoy
    cfg.loc[cfg["Driver"]=="FX_CONVERTED_SALES","Param"] = fx_rate
    # NOTE: OIL_LINKED_FREIGHT param is a percent of Sales; the global oil_ratio multiplies it
    st.session_state["_oil_ratio"] = oil_ratio

    updated = apply_driver_calc(cube, cfg)
    updated = recalc(updated)
    st.session_state["cube"] = updated
    st.success("Applied drivers to CALC for months after the actuals cutoff, then recomputed totals.")

# ---------------- Manual edits still possible ----------------
def apply_manual_edits(original: pd.DataFrame, before_disp: pd.DataFrame, after_disp: pd.DataFrame) -> pd.DataFrame:
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
            if m in locked_set:
                continue
            val = after_disp.loc[idx, m]
            if pd.isna(val): val = before_disp.loc[idx, m]
            df.loc[(df.Account==acc)&(df.Component==comp), m] = float(val)
    return df

if st.button("Recalculate & Save manual edits"):
    updated = apply_manual_edits(cube, display_df, edited)
    updated = recalc(updated)
    st.session_state["cube"] = updated
    st.success("Recalculated from manual grid edits.")

st.caption("Workflow: Upload → Map → Slice → Review suggestions → Choose drivers per combo → Apply (forecasts). You can still make manual tweaks and save.")
