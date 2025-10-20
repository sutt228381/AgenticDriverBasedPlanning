
import io
import streamlit as st
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List

st.set_page_config(page_title="Agentic Driver-Based Planning — v13 Test (Auto Sample)", layout="wide")
APP_TITLE = "Agentic Driver-Based Planning — v13 TEST (Auto Sample Loaded)"
MONTHS: List[str] = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

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

st.title(APP_TITLE)
st.info("This build auto-loads an embedded sample P&L. Use the uploader below to override with your own CSV any time.")

# --- Load embedded sample ---
SAMPLE_CSV = st.secrets.get("SAMPLE_CSV", None)
if SAMPLE_CSV is None:
    # Fallback: sample bundled in README; injected below as a code-block
    from textwrap import dedent
    SAMPLE_CSV = st.session_state.get("_embedded_csv_text", "")
    if not SAMPLE_CSV:
        # placeholder if user didn't paste; we'll populate from README section in repo deployments
        SAMPLE_CSV = ""

# We'll actually embed the CSV text into this file at build time (see zip).
sample_bytes = st.secrets.get("SAMPLE_BYTES", None)

if "src_df" not in st.session_state:
    # prefer bytes if available (binary-safe), else string
    if sample_bytes:
        df = pd.read_csv(io.BytesIO(sample_bytes))
    else:
        df = pd.read_csv(io.StringIO(SAMPLE_CSV)) if SAMPLE_CSV else pd.DataFrame()
    st.session_state["src_df"] = df

# Allow override via uploader
with st.expander("Upload your own CSV (optional, overrides sample)"):
    uf = st.file_uploader("CSV file", type=["csv"])
    if uf is not None:
        st.session_state["src_df"] = pd.read_csv(uf)

df = st.session_state["src_df"]
if df.empty:
    st.warning("No data loaded. If you see this in Streamlit Cloud, ensure the app's secrets include SAMPLE_BYTES.")
    st.stop()

# --- Normalize schema ---
expected_cols = {"Entity","Product","Channel","Currency","Account","Period","Value"}
missing = expected_cols - set(df.columns)
if missing:
    st.error(f"Missing columns: {sorted(missing)}")
    st.stop()

# Keep only known months, sum dupes
df = df.copy()
df["Section"] = df["Account"].map(lambda a: ACCOUNT_TO_SECTION.get(a, "Opex"))
df = df[df["Period"].isin(MONTHS)]
df["Value"] = pd.to_numeric(df["Value"], errors="coerce").fillna(0.0)

# --- Filters ---
with st.sidebar:
    st.header("Filters")
    ent = st.multiselect("Entity", sorted(df["Entity"].unique()), default=sorted(df["Entity"].unique()))
    prod= st.multiselect("Product", sorted(df["Product"].unique()), default=sorted(df["Product"].unique()))
    chan= st.multiselect("Channel", sorted(df["Channel"].unique()), default=sorted(df["Channel"].unique()))
    # Actuals cutoff
    cutoff = st.selectbox("Actuals cutoff (locked)", MONTHS, index=2)
locked = set(MONTHS[:MONTHS.index(cutoff)+1])

sliced = df[df["Entity"].isin(ent) & df["Product"].isin(prod) & df["Channel"].isin(chan)].copy()

# Pivot
by_acc = sliced.pivot_table(index=["Section","Account"], columns="Period", values="Value", aggfunc="sum").reindex(columns=MONTHS, fill_value=0.0)

# Seed CALC/ADJ/TOTAL
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

# Totals preview
st.subheader("Top-line totals (current slice)")
totals = by_acc.groupby(level=0).sum().reset_index()
st.dataframe(totals, use_container_width=True)

# Expand/collapse
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

# Recalc
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

def computed_lines(sec_map):
    out = {}
    out["Gross Profit"]     = sec_map.get("Revenue",0.0) - sec_map.get("COGS",0.0)
    out["Operating Income"] = out["Gross Profit"] - sec_map.get("Opex",0.0)
    out["Pre-Tax Income"]   = out["Operating Income"] + sec_map.get("Other",0.0)
    out["Net Income"]       = out["Pre-Tax Income"] - sec_map.get("Taxes",0.0)
    return out

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

display_df = build_display(st.session_state["cube"], expanded)

# Lock Jan–Mar by default for this test build
locked = set(["Jan","Feb","Mar"])
col_cfg = {m: st.column_config.NumberColumn(disabled=(m in locked)) for m in MONTHS}

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
            if m in locked:
                continue
            val = after_disp.loc[idx, m]
            if pd.isna(val): val = before_disp.loc[idx, m]
            df.loc[(df.Account==acc)&(df.Component==comp), m] = float(val)
    return df

if st.button("Recalculate & Save", type="primary"):
    updated = apply_edits(st.session_state["cube"], display_df, edited)
    updated = recalc(updated)
    st.session_state["cube"] = updated
    st.success("Recalculated.")

st.caption("This test build autoloads a sample dataset. Upload your own CSV in the expander to override.")
