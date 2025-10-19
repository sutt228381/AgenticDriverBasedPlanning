import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import requests
import pandas as pd
import streamlit as st

APP_TITLE = "Agentic Driver-Based Planning — P&L Grid v8 (Hierarchy + Single Grid)"
SHEET_NAME = "AgenticPlanner"
TAB_PRIOR = "prior_year"
TAB_INPUT = "forecast_input"
DEFAULT_ENTITY = "Orvis"
DEFAULT_CURRENCY = "USD"

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
ACTUALIZED_PERIODS = ["Jan","Feb","Mar"]

MONTH_ALIASES = { "jan":"Jan","january":"Jan","01":"Jan","1":"Jan",
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
    "dec":"Dec","december":"Dec","12":"Dec"}
def _norm_period(x:str)->str:
    if x is None: return ""
    s=str(x).strip().lower()
    if s in MONTH_ALIASES: return MONTH_ALIASES[s]
    if s[:3] in MONTH_ALIASES: return MONTH_ALIASES[s[:3]]
    return s[:3].title()

# ---------------- P&L schema ----------------
# RowType: LINE (editable), SUBTOTAL (read-only, by Type), COMPUTED (read-only)
ACCOUNT_TYPES = ["Revenue","COGS","Opex","Other","Taxes"]

DEFAULT_LINES = [
    # Revenue
    {"Account":"Sales","Type":"Revenue","Driver":"MANUAL","Param":"Amount"},
    {"Account":"Returns & Allowances","Type":"Revenue","Driver":"PCT_OF_SALES","Param":"% of Sales"},
    {"Account":"Royalty Income","Type":"Revenue","Driver":"PY_RATIO_SALES","Param":""},

    # COGS
    {"Account":"COGS","Type":"COGS","Driver":"PCT_GROWTH","Param":"Growth%"},
    {"Account":"Freight","Type":"COGS","Driver":"OIL_LINKED_FREIGHT","Param":"% of Sales"},
    {"Account":"Fulfillment","Type":"COGS","Driver":"PCT_OF_SALES","Param":"% of Sales"},

    # Opex (SG&A breakdown)
    {"Account":"Marketing","Type":"Opex","Driver":"PCT_OF_SALES","Param":"% of Sales"},
    {"Account":"Payroll","Type":"Opex","Driver":"CPI_INDEXED","Param":""},
    {"Account":"G&A","Type":"Opex","Driver":"CPI_INDEXED","Param":""},
    {"Account":"Depreciation","Type":"Opex","Driver":"MANUAL","Param":"Amount"},

    # Other & Taxes
    {"Account":"Interest Expense","Type":"Other","Driver":"MANUAL","Param":"Amount"},
    {"Account":"Other Income","Type":"Other","Driver":"MANUAL","Param":"Amount"},
    {"Account":"Taxes","Type":"Taxes","Driver":"MANUAL","Param":"Amount"},
]

PRESENTATION_ORDER = [
    ("Revenue", ["Sales","Returns & Allowances","Royalty Income"]),
    ("COGS", ["COGS","Freight","Fulfillment"]),
    ("Opex", ["Marketing","Payroll","G&A","Depreciation"]),
    ("Other", ["Other Income","Interest Expense"]),
    ("Taxes", ["Taxes"]),
]

COMPUTED_LINES = [
    ("Gross Profit",    lambda sec: sec.get("Revenue",0) - sec.get("COGS",0)),
    ("EBITDA",          lambda sec: (sec.get("Revenue",0)-sec.get("COGS",0)) - (sec.get("Opex",0) - 0.0)), # Depreciation kept in Opex lines; tweak if you move it
    ("Operating Income",lambda sec: (sec.get("Revenue",0)-sec.get("COGS",0)) - sec.get("Opex",0)),
    ("Pre-Tax Income",  lambda sec: (sec.get("Revenue",0)-sec.get("COGS",0) - sec.get("Opex",0)) + sec.get("Other",0)),
    ("Net Income",      lambda sec: (sec.get("Revenue",0)-sec.get("COGS",0) - sec.get("Opex",0) + sec.get("Other",0)) - sec.get("Taxes",0)),
]

COMP_CHOICES = ["CALC","MANUAL_ADJ","TOTAL"]
DRIVERS = ["MANUAL","PCT_GROWTH","PCT_OF_SALES","PY_RATIO_SALES","CPI_INDEXED","OIL_LINKED_FREIGHT","FX_CONVERTED_SALES"]

# -------- External drivers (demo/fallbacks) --------
@st.cache_data(ttl=3600)
def fetch_cpi_yoy()->Tuple[float,str]:
    return 0.022, "CPI YoY (demo) = 2.2%"

@st.cache_data(ttl=3600)
def fetch_fx_rate(base:str,target:str)->Tuple[float,str]:
    try:
        r=requests.get(f"https://api.frankfurter.app/latest?from={base}&to={target}",timeout=10)
        if r.ok:
            rate=float(r.json()["rates"][target]); return rate, f"FX {base}->{target} = {rate:.4f}"
    except Exception: ...
    return 1.08, "FX fallback = 1.08"

@st.cache_data(ttl=3600)
def fetch_oil_index_ratio(prev=75.0)->Tuple[float,str]:
    cur=82.0; ratio=cur/max(prev,1.0)
    return ratio, f"Oil index ratio = {cur:.1f}/{prev:.1f} = {ratio:.3f} (demo)"

# -------- Demo prior (for actuals Jan–Mar) --------
@st.cache_data
def demo_prior()->pd.DataFrame:
    rows=[]
    for m, sales, returns, royalty, cogs, freight in [
        ("Jan", 100000, -2000, 3000, 55000, 8000),
        ("Feb", 110000, -2200, 3200, 60500, 8300),
        ("Mar", 105000, -2100, 3100, 58000, 8200),
    ]:
        rows += [
          {"Account":"Sales","Type":"Revenue","Period":m,"Value":sales},
          {"Account":"Returns & Allowances","Type":"Revenue","Period":m,"Value":returns},
          {"Account":"Royalty Income","Type":"Revenue","Period":m,"Value":royalty},
          {"Account":"COGS","Type":"COGS","Period":m,"Value":cogs},
          {"Account":"Freight","Type":"COGS","Period":m,"Value":freight},
        ]
    return pd.DataFrame(rows)

# -------- Unified grid seeding --------
def seed_unified()->pd.DataFrame:
    base=[]
    order=0
    for section, accounts in PRESENTATION_ORDER:
        # LINE rows
        for acc in accounts:
            meta = next((d for d in DEFAULT_LINES if d["Account"]==acc), None)
            if meta is None: continue
            for comp in ["CALC","MANUAL_ADJ","TOTAL"]:
                for m in MONTHS:
                    base.append({
                        "Order": order, "RowType":"LINE",
                        "Type":section, "Account":acc, "Component":comp,
                        "Period":m,
                        "Driver": meta["Driver"] if comp=="CALC" else "MANUAL",
                        "Param": meta["Param"]  if comp=="CALC" else "Adj",
                        "Value":0.0
                    })
            order+=1
        # SUBTOTAL row (one per section, TOTAL component only)
        for m in MONTHS:
            base.append({
                "Order": order+0.5, "RowType":"SUBTOTAL", "Type":section, "Account":f"{section} Subtotal",
                "Component":"TOTAL","Period":m,"Driver":"LOCK","Param":"","Value":0.0
            })
    # COMPUTED rows
    for name,_ in COMPUTED_LINES:
        for m in MONTHS:
            base.append({
                "Order": 999 + list(zip(*COMPUTED_LINES))[0].index(name), "RowType":"COMPUTED", "Type":"Computed",
                "Account":name,"Component":"TOTAL","Period":m,"Driver":"LOCK","Param":"","Value":0.0
            })
    df=pd.DataFrame(base)
    # Seed demo Sales CALC Jan–Mar
    seeds={"Jan":120000,"Feb":118000,"Mar":119500}
    for m,v in seeds.items():
        mask=(df["RowType"]=="LINE")&(df["Account"]=="Sales")&(df["Component"]=="CALC")&(df["Period"]==m)
        df.loc[mask,"Value"]=float(v)
    return df

# -------- Agent engine --------
@dataclass
class Ctx:
    entity:str
    currency:str
    cpi_yoy:float
    fx_eur_to_target:float
    oil_ratio:float

def say(msg):
    st.session_state.setdefault("explain",[])
    st.session_state["explain"].append(msg)

def clear_explain():
    st.session_state.pop("explain",None)

def _pivot_prior(prior_df: pd.DataFrame)->pd.DataFrame:
    pri=prior_df.copy()
    pri["Period"]=pri["Period"].apply(_norm_period)
    return pri.pivot_table(index=["Account","Type"], columns="Period", values="Value", aggfunc="sum").fillna(0.0)

def recalc(df_long: pd.DataFrame, prior_df: pd.DataFrame, ctx: Ctx, actuals_pivot: Optional[pd.DataFrame]) -> pd.DataFrame:
    df=df_long.copy()
    df["Value"]=pd.to_numeric(df["Value"],errors="coerce").fillna(0.0)
    df["Period"]=df["Period"].apply(_norm_period)
    py=_pivot_prior(prior_df)

    # Helper for sales base
    def get_sales(period:str)->float:
        # Prefer TOTAL Sales current df
        m=(df["RowType"]=="LINE")&(df["Account"]=="Sales")&(df["Component"]=="TOTAL")&(df["Period"]==period)
        if m.any(): return float(df.loc[m,"Value"].sum())
        # else CALC
        m=(df["RowType"]=="LINE")&(df["Account"]=="Sales")&(df["Component"]=="CALC")&(df["Period"]==period)
        if m.any(): return float(df.loc[m,"Value"].sum())
        # else prior
        try: return float(py.loc[("Sales","Revenue"), period])
        except Exception: return 0.0

    # 1) Compute CALC for LINE rows
    mask_line_calc = (df["RowType"]=="LINE")&(df["Component"]=="CALC")
    for idx,row in df[mask_line_calc].iterrows():
        acc=row["Account"]; typ=row["Type"]; per=row["Period"]
        drv=str(row["Driver"]).upper().strip()
        param=float(row["Value"])
        py_val=float(py.loc[(acc,typ), per]) if (acc,typ) in py.index and per in py.columns else 0.0

        # Lock actualized months to actuals
        if per in ACTUALIZED_PERIODS:
            actual=None
            if actuals_pivot is not None:
                try: actual=float(actuals_pivot.loc[(acc,typ), per])
                except Exception: actual=None
            df.at[idx,"Value"]=actual if actual is not None else py_val
            say(f"{acc} {per} CALC → Actual {df.at[idx,'Value']:.2f}")
            continue

        if drv=="MANUAL":
            df.at[idx,"Value"]=param
            say(f"{acc} {per} CALC: MANUAL {param:.2f}")
        elif drv=="PCT_GROWTH":
            df.at[idx,"Value"]=py_val*(1.0+param)
            say(f"{acc} {per} CALC: PY {py_val:.2f}*(1+{param:.2%})")
        elif drv=="PCT_OF_SALES":
            sales=get_sales(per); df.at[idx,"Value"]=sales*param
            say(f"{acc} {per} CALC: {param:.2%} * Sales {sales:.2f}")
        elif drv=="PY_RATIO_SALES":
            try: py_sales=float(py.loc[("Sales","Revenue"), per])
            except Exception: py_sales=0.0
            ratio=(py_val/py_sales) if py_sales else 0.0
            cpi_mult=1.0+ctx.cpi_yoy
            df.at[idx,"Value"]=(py_sales*ratio)*cpi_mult
            say(f"{acc} {per} CALC: Sales*Ratio({ratio:.2%})*CPI({ctx.cpi_yoy:.2%})")
        elif drv=="CPI_INDEXED":
            df.at[idx,"Value"]=py_val*(1.0+ctx.cpi_yoy)
            say(f"{acc} {per} CALC: PY {py_val:.2f}*(1+CPI {ctx.cpi_yoy:.2%})")
        elif drv=="OIL_LINKED_FREIGHT":
            sales=get_sales(per); df.at[idx,"Value"]=sales*param*ctx.oil_ratio
            say(f"{acc} {per} CALC: %Sales {param:.2%} * Oil {ctx.oil_ratio:.3f}")
        elif drv=="FX_CONVERTED_SALES":
            base_amt=param; df.at[idx,"Value"]=base_amt*ctx.fx_eur_to_target
            say(f"{acc} {per} CALC: EUR {base_amt:.2f} * FX {ctx.fx_eur_to_target:.4f}")
        else:
            df.at[idx,"Value"]=py_val
            say(f"{acc} {per} CALC: fallback PY {py_val:.2f}")

    # 2) TOTAL = CALC + MANUAL_ADJ (for LINE rows)
    for (acc,typ,per), sub in df[df["RowType"]=="LINE"].groupby(["Account","Type","Period"]):
        calc=float(sub.loc[sub["Component"]=="CALC","Value"].sum()) if (sub["Component"]=="CALC").any() else 0.0
        adj =float(sub.loc[sub["Component"]=="MANUAL_ADJ","Value"].sum()) if (sub["Component"]=="MANUAL_ADJ").any() else 0.0
        df.loc[(df["RowType"]=="LINE")&(df["Account"]==acc)&(df["Type"]==typ)&(df["Component"]=="TOTAL")&(df["Period"]==per),"Value"]=calc+adj

    # 3) SUBTOTAL per section (sum of LINE TOTALs by Type)
    for sec in ACCOUNT_TYPES:
        for m in MONTHS:
            subtotal=float(df[(df["RowType"]=="LINE")&(df["Type"]==sec)&(df["Component"]=="TOTAL")&(df["Period"]==m)]["Value"].sum())
            df.loc[(df["RowType"]=="SUBTOTAL")&(df["Type"]==sec)&(df["Period"]==m),"Value"]=subtotal

    # 4) COMPUTED lines using section subtotals
    # Build section totals dict per month
    for m in MONTHS:
        sec = { t: float(df[(df["RowType"]=="SUBTOTAL")&(df["Type"]==t)&(df["Period"]==m)]["Value"].sum()) for t in ACCOUNT_TYPES }
        for name, formula in COMPUTED_LINES:
            val=float(formula(sec))
            df.loc[(df["RowType"]=="COMPUTED")&(df["Account"]==name)&(df["Period"]==m),"Value"]=val

    return df

# ---------------- UI ----------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Single grid with hierarchy: LINE (editable), SUBTOTAL & COMPUTED (read-only). CALC + MANUAL_ADJ → TOTAL. Jan–Mar actuals locked.")

with st.sidebar:
    st.header("Configuration")
    entity = st.text_input("Entity", value=DEFAULT_ENTITY)
    currency = st.selectbox("Currency", ["USD","EUR","GBP"], index=0)
    cpi, cpi_info = fetch_cpi_yoy()
    fx_rate, fx_info = fetch_fx_rate("EUR", currency)
    oil_ratio, oil_info = fetch_oil_index_ratio()
    st.write(cpi_info); st.write(fx_info); st.write(oil_info)
ctx = Ctx(entity=entity, currency=currency, cpi_yoy=cpi, fx_eur_to_target=fx_rate, oil_ratio=oil_ratio)

# Prior / actuals
prior_df = demo_prior()
prior_df["Period"]=prior_df["Period"].apply(_norm_period)
actuals_df = prior_df[prior_df["Period"].isin(ACTUALIZED_PERIODS)].copy()
actuals_pivot = actuals_df.pivot_table(index=["Account","Type"], columns="Period", values="Value", aggfunc="sum")

# Unified model
if "unified" not in st.session_state:
    st.session_state["unified"] = seed_unified()
unified = st.session_state["unified"].copy()

# ---- Display the unified grid (hierarchical) ----
st.subheader("Unified P&L Grid")
# Wide pivot per months
wide = unified.pivot_table(index=["Order","RowType","Type","Account","Component","Driver","Param"], columns="Period", values="Value", aggfunc="sum").reindex(columns=MONTHS, fill_value=0.0).reset_index()
wide = wide.sort_values(["Order","RowType","Account","Component"]).reset_index(drop=True)

# Column configs
rowtype_col = st.column_config.SelectboxColumn("RowType", options=["LINE","SUBTOTAL","COMPUTED"], disabled=True)
type_col    = st.column_config.SelectboxColumn("Type", options=ACCOUNT_TYPES+["Computed"])
acc_col     = st.column_config.TextColumn("Account")
comp_col    = st.column_config.SelectboxColumn("Component", options=COMP_CHOICES, help="TOTAL for LINE is computed; SUBTOTAL/COMPUTED always TOTAL")
drv_col     = st.column_config.SelectboxColumn("Driver", options=DRIVERS+["LOCK"])
param_col   = st.column_config.TextColumn("Param")

# Month columns: lock Jan–Mar; and lock SUBTOTAL/COMPUTED rows for all months; lock TOTAL rows for LINE for all months.
num_cols={}
for m in MONTHS:
    num_cols[m] = st.column_config.NumberColumn(m, format="%.2f")

# Build disabled map row-by-row after edit using rules; Streamlit data_editor can't dynamically per-cell lock via config,
# so we pre-enforce by zeroing edits later if they violate rules.
grid = st.data_editor(
    wide,
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "RowType":rowtype_col,"Type":type_col,"Account":acc_col,"Component":comp_col,"Driver":drv_col,"Param":param_col, **num_cols
    },
    key="pl_v8_grid",
)

st.caption("Edit months only on **LINE + (CALC or MANUAL_ADJ)** rows for **Apr–Dec**. **TOTAL**, **SUBTOTAL**, and **COMPUTED** rows are read-only and will be recomputed.")

# ---- Recalculate ----
col1, col2 = st.columns(2)
with col1:
    if st.button("Recalculate", type="primary"):
        # Convert back to long
        long = grid.melt(id_vars=["Order","RowType","Type","Account","Component","Driver","Param"], value_vars=[c for c in grid.columns if c in MONTHS], var_name="Period", value_name="Value")
        long["Value"]=pd.to_numeric(long["Value"],errors="coerce").fillna(0.0)
        long["Period"]=long["Period"].apply(_norm_period)

        # Enforce lock rules before compute:
        # - Jan–Mar: no edits allowed (revert to previous unified)
        # - SUBTOTAL/COMPUTED: ignore edits
        # - LINE TOTAL: ignore edits (computed)
        prev = st.session_state["unified"]
        for m in ACTUALIZED_PERIODS:
            mask = long["Period"]==m
            # restore prior values for actualized months from prev snapshot
            join_cols=["Order","RowType","Type","Account","Component","Driver","Param","Period"]
            long = long.merge(prev[join_cols+["Value"]].rename(columns={"Value":"PrevValue"}), on=join_cols, how="left")
            long.loc[mask,"Value"]=long.loc[mask,"PrevValue"].fillna(long.loc[mask,"Value"])
            long = long.drop(columns=["PrevValue"])

        mask_lock = (long["RowType"].isin(["SUBTOTAL","COMPUTED"])) | ((long["RowType"]=="LINE") & (long["Component"]=="TOTAL"))
        if mask_lock.any():
            # restore locked lines fully from previous
            join_cols=["Order","RowType","Type","Account","Component","Driver","Param","Period"]
            long = long.merge(prev[join_cols+["Value"]].rename(columns={"Value":"PrevValue2"}), on=join_cols, how="left")
            long.loc[mask_lock,"Value"]=long.loc[mask_lock,"PrevValue2"].fillna(long.loc[mask_lock,"Value"])
            long = long.drop(columns=["PrevValue2"])

        # Compute results
        clear_explain()
        out = recalc(long, prior_df, ctx, actuals_pivot)
        st.session_state["unified"]=out
        st.success("Recalculated. See updated grid below.")

with col2:
    if st.button("Restore Demo Grid"):
        st.session_state["unified"]=seed_unified()
        st.experimental_rerun()

# ---- Show result (same unified grid) ----
st.subheader("Result (same grid)")
res = st.session_state["unified"]
wide_out = res.pivot_table(index=["Order","RowType","Type","Account","Component","Driver","Param"], columns="Period", values="Value", aggfunc="sum").reindex(columns=MONTHS, fill_value=0.0).reset_index()
wide_out = wide_out.sort_values(["Order","RowType","Account","Component"]).reset_index(drop=True)
st.dataframe(wide_out, use_container_width=True)

st.subheader("Explainability")
for m in st.session_state.get("explain",[]):
    st.write("•", m)
