import time
from dataclasses import dataclass
from typing import Tuple, Optional
import requests
import pandas as pd
import streamlit as st

APP_TITLE = "Agentic Driver-Based Planning — P&L Grid v8.1 (Single Grid In-Place Updates)"
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

ACCOUNT_TYPES = ["Revenue","COGS","Opex","Other","Taxes"]
DEFAULT_LINES = [
    {"Account":"Sales","Type":"Revenue","Driver":"MANUAL","Param":"Amount"},
    {"Account":"Returns & Allowances","Type":"Revenue","Driver":"PCT_OF_SALES","Param":"% of Sales"},
    {"Account":"Royalty Income","Type":"Revenue","Driver":"PY_RATIO_SALES","Param":""},
    {"Account":"COGS","Type":"COGS","Driver":"PCT_GROWTH","Param":"Growth%"},
    {"Account":"Freight","Type":"COGS","Driver":"OIL_LINKED_FREIGHT","Param":"% of Sales"},
    {"Account":"Fulfillment","Type":"COGS","Driver":"PCT_OF_SALES","Param":"% of Sales"},
    {"Account":"Marketing","Type":"Opex","Driver":"PCT_OF_SALES","Param":"% of Sales"},
    {"Account":"Payroll","Type":"Opex","Driver":"CPI_INDEXED","Param":""},
    {"Account":"G&A","Type":"Opex","Driver":"CPI_INDEXED","Param":""},
    {"Account":"Depreciation","Type":"Opex","Driver":"MANUAL","Param":"Amount"},
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
    ("EBITDA",          lambda sec: (sec.get("Revenue",0)-sec.get("COGS",0)) - (sec.get("Opex",0) - 0.0)),
    ("Operating Income",lambda sec: (sec.get("Revenue",0)-sec.get("COGS",0)) - sec.get("Opex",0)),
    ("Pre-Tax Income",  lambda sec: (sec.get("Revenue",0)-sec.get("COGS",0) - sec.get("Opex",0)) + sec.get("Other",0)),
    ("Net Income",      lambda sec: (sec.get("Revenue",0)-sec.get("COGS",0) - sec.get("Opex",0) + sec.get("Other",0)) - sec.get("Taxes",0)),
]
COMP_CHOICES = ["CALC","MANUAL_ADJ","TOTAL"]
DRIVERS = ["MANUAL","PCT_GROWTH","PCT_OF_SALES","PY_RATIO_SALES","CPI_INDEXED","OIL_LINKED_FREIGHT","FX_CONVERTED_SALES"]

# ---- External drivers (demo/fallback) ----
@st.cache_data(ttl=3600)
def fetch_cpi_yoy()->Tuple[float,str]: return 0.022, "CPI YoY (demo) = 2.2%"
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

# ---- Demo prior (for actuals) ----
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

# ---- Seed unified hierarchical grid ----
def seed_unified()->pd.DataFrame:
    base=[]
    order=0
    for section, accounts in PRESENTATION_ORDER:
        for acc in accounts:
            meta = next((d for d in DEFAULT_LINES if d["Account"]==acc), None)
            for comp in ["CALC","MANUAL_ADJ","TOTAL"]:
                for m in MONTHS:
                    base.append({
                        "Order": order, "RowType":"LINE",
                        "Type":section, "Account":acc, "Component":comp,
                        "Period":m,
                        "Driver": (meta["Driver"] if comp=="CALC" else "MANUAL"),
                        "Param":  (meta["Param"]  if comp=="CALC" else "Adj"),
                        "Value":0.0
                    })
            order+=1
        # SUBTOTAL row (TOTAL only)
        for m in MONTHS:
            base.append({
                "Order": order+0.5, "RowType":"SUBTOTAL", "Type":section, "Account":f"{section} Subtotal",
                "Component":"TOTAL","Period":m,"Driver":"LOCK","Param":"","Value":0.0
            })
    # COMPUTED rows
    for i,(name,_) in enumerate(COMPUTED_LINES):
        for m in MONTHS:
            base.append({
                "Order": 900+i, "RowType":"COMPUTED","Type":"Computed",
                "Account":name,"Component":"TOTAL","Period":m,"Driver":"LOCK","Param":"","Value":0.0
            })
    df=pd.DataFrame(base)
    # Seed Sales CALC Jan–Mar
    for m,v in {"Jan":120000,"Feb":118000,"Mar":119500}.items():
        mask=(df["RowType"]=="LINE")&(df["Account"]=="Sales")&(df["Component"]=="CALC")&(df["Period"]==m)
        df.loc[mask,"Value"]=float(v)
    return df

# ---- Agent engine ----
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

    def get_sales(period:str)->float:
        m=(df["RowType"]=="LINE")&(df["Account"]=="Sales")&(df["Component"]=="TOTAL")&(df["Period"]==period)
        if m.any(): return float(df.loc[m,"Value"].sum())
        m=(df["RowType"]=="LINE")&(df["Account"]=="Sales")&(df["Component"]=="CALC")&(df["Period"]==period)
        if m.any(): return float(df.loc[m,"Value"].sum())
        try: return float(py.loc[("Sales","Revenue"), period])
        except Exception: return 0.0

    # 1) CALC for LINE
    mask_calc=(df["RowType"]=="LINE")&(df["Component"]=="CALC")
    for idx,row in df[mask_calc].iterrows():
        acc=row["Account"]; typ=row["Type"]; per=row["Period"]
        drv=str(row["Driver"]).upper().strip()
        param=float(row["Value"])
        py_val=float(py.loc[(acc,typ), per]) if (acc,typ) in py.index and per in py.columns else 0.0

        if per in ACTUALIZED_PERIODS:
            actual=None
            if actuals_pivot is not None:
                try: actual=float(actuals_pivot.loc[(acc,typ), per])
                except Exception: actual=None
            df.at[idx,"Value"]=actual if actual is not None else py_val
            say(f"{acc} {per} CALC → Actual {df.at[idx,'Value']:.2f}")
            continue

        if   drv=="MANUAL":          df.at[idx,"Value"]=param
        elif drv=="PCT_GROWTH":      df.at[idx,"Value"]=py_val*(1.0+param)
        elif drv=="PCT_OF_SALES":    df.at[idx,"Value"]=get_sales(per)*param
        elif drv=="PY_RATIO_SALES":
            try: py_sales=float(py.loc[("Sales","Revenue"), per])
            except Exception: py_sales=0.0
            ratio=(py_val/py_sales) if py_sales else 0.0
            df.at[idx,"Value"]=(py_sales*ratio)*(1.0+ctx.cpi_yoy)
        elif drv=="CPI_INDEXED":     df.at[idx,"Value"]=py_val*(1.0+ctx.cpi_yoy)
        elif drv=="OIL_LINKED_FREIGHT": df.at[idx,"Value"]=get_sales(per)*param*ctx.oil_ratio
        elif drv=="FX_CONVERTED_SALES": df.at[idx,"Value"]=param*ctx.fx_eur_to_target
        else:                         df.at[idx,"Value"]=py_val

    # 2) TOTAL = CALC + MANUAL_ADJ (LINE rows)
    for (acc,typ,per), sub in df[df["RowType"]=="LINE"].groupby(["Account","Type","Period"]):
        calc=float(sub.loc[sub["Component"]=="CALC","Value"].sum()) if (sub["Component"]=="CALC").any() else 0.0
        adj =float(sub.loc[sub["Component"]=="MANUAL_ADJ","Value"].sum()) if (sub["Component"]=="MANUAL_ADJ").any() else 0.0
        df.loc[(df["RowType"]=="LINE")&(df["Account"]==acc)&(df["Type"]==typ)&(df["Component"]=="TOTAL")&(df["Period"]==per),"Value"]=calc+adj

    # 3) SUBTOTAL (sum of LINE TOTALs by Type)
    for sec in ACCOUNT_TYPES:
        for m in MONTHS:
            subtotal=float(df[(df["RowType"]=="LINE")&(df["Type"]==sec)&(df["Component"]=="TOTAL")&(df["Period"]==m)]["Value"].sum())
            df.loc[(df["RowType"]=="SUBTOTAL")&(df["Type"]==sec)&(df["Period"]==m),"Value"]=subtotal

    # 4) COMPUTED lines
    for m in MONTHS:
        sec = { t: float(df[(df["RowType"]=="SUBTOTAL")&(df["Type"]==t)&(df["Period"]==m)]["Value"].sum()) for t in ACCOUNT_TYPES }
        for name, formula in COMPUTED_LINES:
            val=float(formula(sec))
            df.loc[(df["RowType"]=="COMPUTED")&(df["Account"]==name)&(df["Period"]==m),"Value"]=val
    return df

# ---- UI ----
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Single grid only. Edit LINE rows (CALC params & MANUAL_ADJ) for Apr–Dec. Jan–Mar locked. SUBTOTAL/COMPUTED are read-only.")

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

# Build wide grid from session state
wide = unified.pivot_table(index=["Order","RowType","Type","Account","Component","Driver","Param"], columns="Period", values="Value", aggfunc="sum").reindex(columns=MONTHS, fill_value=0.0).reset_index()
wide = wide.sort_values(["Order","RowType","Account","Component"]).reset_index(drop=True)

# Column configs
rowtype_col = st.column_config.SelectboxColumn("RowType", options=["LINE","SUBTOTAL","COMPUTED"], disabled=True)
type_col    = st.column_config.SelectboxColumn("Type", options=ACCOUNT_TYPES+["Computed"])
acc_col     = st.column_config.TextColumn("Account")
comp_col    = st.column_config.SelectboxColumn("Component", options=COMP_CHOICES)
drv_col     = st.column_config.SelectboxColumn("Driver", options=DRIVERS+["LOCK"])
param_col   = st.column_config.TextColumn("Param")

num_cols={m: st.column_config.NumberColumn(m, format="%.2f") for m in MONTHS}

grid = st.data_editor(
    wide,
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "RowType":rowtype_col,"Type":type_col,"Account":acc_col,"Component":comp_col,"Driver":drv_col,"Param":param_col,
        **num_cols
    },
    key="pl_v81_grid",
)

st.caption("Edits allowed only on LINE + (CALC or MANUAL_ADJ) for Apr–Dec. TOTAL, SUBTOTAL, COMPUTED are recomputed.")

# Recalculate in-place: update session and rerun so the SAME grid reflects updates
if st.button("Recalculate", type="primary"):
    # Back to long
    long = grid.melt(id_vars=["Order","RowType","Type","Account","Component","Driver","Param"], value_vars=[c for c in grid.columns if c in MONTHS], var_name="Period", value_name="Value")
    long["Value"]=pd.to_numeric(long["Value"],errors="coerce").fillna(0.0)
    long["Period"]=long["Period"].apply(_norm_period)

    # Enforce lock rules by restoring locked cells from previous snapshot
    prev = st.session_state["unified"]
    join_cols=["Order","RowType","Type","Account","Component","Driver","Param","Period"]

    # Jan–Mar locked for all
    long = long.merge(prev[join_cols+["Value"]].rename(columns={"Value":"PrevA"}), on=join_cols, how="left")
    long.loc[long["Period"].isin(ACTUALIZED_PERIODS),"Value"] = long.loc[long["Period"].isin(ACTUALIZED_PERIODS),"PrevA"].fillna(long.loc[long["Period"].isin(ACTUALIZED_PERIODS),"Value"])
    long = long.drop(columns=["PrevA"])

    # SUBTOTAL/COMPUTED always locked; LINE TOTAL locked
    mask_lock = (long["RowType"].isin(["SUBTOTAL","COMPUTED"])) | ((long["RowType"]=="LINE") & (long["Component"]=="TOTAL"))
    long = long.merge(prev[join_cols+["Value"]].rename(columns={"Value":"PrevB"}), on=join_cols, how="left")
    long.loc[mask_lock,"Value"] = long.loc[mask_lock,"PrevB"].fillna(long.loc[mask_lock,"Value"])
    long = long.drop(columns=["PrevB"])

    # Compute & store
    clear_explain()
    out = recalc(long, prior_df, ctx, actuals_pivot)
    st.session_state["unified"] = out

    # Rerun to refresh the SAME grid with new values
    st.experimental_rerun()
