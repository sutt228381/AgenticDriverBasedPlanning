# v7 START
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import requests
import pandas as pd
import streamlit as st

APP_TITLE = "Agentic Driver-Based Planning — P&L Grid v7 (Calc + Manual Adj + Total)"
SHEET_NAME = "AgenticPlanner"
TAB_PRIOR = "prior_year"         # actuals source
TAB_INPUT = "forecast_input"     # we’ll store/edit the unified grid here
TAB_SCENARIOS = "scenarios"
DEFAULT_ENTITY = "Orvis"
DEFAULT_CURRENCY = "USD"

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
ACTUALIZED_PERIODS = ["Jan","Feb","Mar"]

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
def _norm_period(x:str)->str:
    if x is None: return ""
    s=str(x).strip().lower()
    if s in MONTH_ALIASES: return MONTH_ALIASES[s]
    if s[:3] in MONTH_ALIASES: return MONTH_ALIASES[s[:3]]
    return s[:3].title()

ACCOUNT_TYPES = ["Revenue","COGS","Opex","Other","Taxes"]
DEFAULT_ACCOUNTS = [
    {"Account":"Sales","Type":"Revenue","Driver":"MANUAL","Param":"Amount"},
    {"Account":"Returns & Allowances","Type":"Revenue","Driver":"PCT_OF_SALES","Param":"% of Sales"},
    {"Account":"Royalty Income","Type":"Other","Driver":"PY_RATIO_SALES","Param":""},
    {"Account":"COGS","Type":"COGS","Driver":"PCT_GROWTH","Param":"Growth%"},
    {"Account":"Freight","Type":"COGS","Driver":"OIL_LINKED_FREIGHT","Param":"% of Sales"},
    {"Account":"Fulfillment","Type":"COGS","Driver":"PCT_OF_SALES","Param":"% of Sales"},
    {"Account":"Marketing","Type":"Opex","Driver":"PCT_OF_SALES","Param":"% of Sales"},
    {"Account":"Payroll","Type":"Opex","Driver":"CPI_INDEXED","Param":""},
    {"Account":"G&A","Type":"Opex","Driver":"CPI_INDEXED","Param":""},
    {"Account":"Depreciation","Type":"Opex","Driver":"MANUAL","Param":"Amount"},
    {"Account":"Interest Expense","Type":"Other","Driver":"MANUAL","Param":"Amount"},
    {"Account":"Taxes","Type":"Taxes","Driver":"MANUAL","Param":"Amount"},
]
PRESENTATION_ORDER = [
    ("Revenue", ["Sales","Returns & Allowances","Royalty Income"]),
    ("COGS", ["COGS","Freight","Fulfillment"]),
    ("Opex", ["Marketing","Payroll","G&A","Depreciation"]),
    ("Other", ["Interest Expense"]),
    ("Taxes", ["Taxes"]),
]

COMP_CHOICES = ["CALC","MANUAL_ADJ","TOTAL"]
DRIVERS = ["MANUAL","PCT_GROWTH","PCT_OF_SALES","PY_RATIO_SALES","CPI_INDEXED","OIL_LINKED_FREIGHT","FX_CONVERTED_SALES"]

# ---------- Optional: Google Sheets ----------
@st.cache_resource(show_spinner=False)
def _gspread_client():
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except Exception:
        return None
    if "gcp_service_account" not in st.secrets:
        return None
    scopes=["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scopes)
    import gspread
    return gspread.authorize(creds)

def _open_or_create(client, name):
    if client is None: return None, {}
    try:
        sh = client.open(name)
    except Exception:
        sh = client.create(name)
    ws={}
    for t in ["prior_year","forecast_input","scenarios"]:
        try: ws[t]=sh.worksheet(t)
        except Exception: ws[t]=sh.add_worksheet(title=t, rows=1000, cols=40)
    return sh, ws

def read_sheet_df(ws):
    vals=ws.get_all_values()
    if not vals: return pd.DataFrame()
    head, rows = vals[0], vals[1:]
    return pd.DataFrame(rows, columns=head)

def write_sheet_df(ws, df):
    ws.clear()
    ws.update([df.columns.tolist()] + df.astype(str).values.tolist())

# ---------- External drivers (demo) ----------
@st.cache_data(ttl=3600)
def fetch_cpi_yoy()->Tuple[float,str]:
    return 0.022, "CPI YoY (demo) = 2.2%"

@st.cache_data(ttl=3600)
def fetch_fx_rate(base:str,target:str)->Tuple[float,str]:
    try:
        r=requests.get(f"https://api.frankfurter.app/latest?from={base}&to={target}",timeout=10)
        if r.ok:
            rate=float(r.json()["rates"][target])
            return rate, f"FX {base}->{target} = {rate:.4f}"
    except Exception: ...
    return 1.08, "FX fallback = 1.08"

@st.cache_data(ttl=3600)
def fetch_oil_index_ratio(prev=75.0)->Tuple[float,str]:
    cur=82.0; ratio=cur/max(prev,1.0)
    return ratio, f"Oil index ratio = {cur:.1f}/{prev:.1f} = {ratio:.3f} (demo)"

# ---------- Demo seeds ----------
@st.cache_data
def demo_prior()->pd.DataFrame:
    rows=[]
    for m, sales, royalty, cogs, freight in [
        ("Jan",100000,5000,55000,8000),
        ("Feb",110000,5500,60500,8300),
        ("Mar",105000,5250,58000,8200),
    ]:
        rows += [
          {"Account":"Sales","Type":"Revenue","Period":m,"Value":sales},
          {"Account":"Royalty Income","Type":"Other","Period":m,"Value":royalty},
          {"Account":"COGS","Type":"COGS","Period":m,"Value":cogs},
          {"Account":"Freight","Type":"COGS","Period":m,"Value":freight},
        ]
    return pd.DataFrame(rows)

def seed_unified_input()->pd.DataFrame:
    base=[]
    for d in DEFAULT_ACCOUNTS:
        acc, typ, drv, prm = d["Account"], d["Type"], d["Driver"], d["Param"]
        for comp in ["CALC","MANUAL_ADJ","TOTAL"]:
            for m in MONTHS:
                base.append({"Account":acc,"Type":typ,"Component":comp,"Period":m,"Driver":drv if comp=="CALC" else "MANUAL","Param":prm if comp=="CALC" else "Adj","Value":0.0})
    for m,v in {"Jan":120000,"Feb":118000,"Mar":119500}.items():
        # seed Sales CALC Jan–Mar
        pass
    return pd.DataFrame(base)

# ---------- Agent engine ----------
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

def recalc_components(prior_df: pd.DataFrame, unified_df: pd.DataFrame, ctx: Ctx, actuals_pivot: Optional[pd.DataFrame]) -> pd.DataFrame:
    df = unified_df.copy()
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce").fillna(0.0)
    df["Period"] = df["Period"].apply(_norm_period)
    pri = prior_df.copy()
    pri["Period"]=pri["Period"].apply(_norm_period)
    py = pri.pivot_table(index=["Account","Type"], columns="Period", values="Value", aggfunc="sum").fillna(0.0)

    def get_sales(period:str)->float:
        m = (df["Account"]=="Sales")&(df["Component"]=="CALC")&(df["Period"]==period)
        if m.any(): return float(df.loc[m,"Value"].sum())
        m = (df["Account"]=="Sales")&(df["Component"]=="TOTAL")&(df["Period"]==period)
        if m.any(): return float(df.loc[m,"Value"].sum())
        try: return float(py.loc[("Sales","Revenue"), period])
        except Exception: return 0.0

    # Recompute CALC
    calc_mask = df["Component"]=="CALC"
    for idx, row in df[calc_mask].iterrows():
        acc=row["Account"]; typ=row.get("Type",""); per=row["Period"]; drv=str(row.get("Driver","")).upper().strip()
        param_val = float(row.get("Value",0.0))
        py_val = float(py.loc[(acc,typ), per]) if (acc,typ) in py.index and per in py.columns else 0.0

        if per in ACTUALIZED_PERIODS:
            actual_val = None
            if actuals_pivot is not None:
                try: actual_val = float(actuals_pivot.loc[(acc,typ), per])
                except Exception: actual_val = None
            df.at[idx,"Value"] = actual_val if actual_val is not None else py_val
            say(f"{acc} {per} CALC: Actual = {df.at[idx,'Value']:.2f}")
            continue

        if drv=="MANUAL":
            df.at[idx,"Value"] = param_val
        elif drv=="PCT_GROWTH":
            df.at[idx,"Value"] = py_val * (1.0 + param_val)
        elif drv=="PCT_OF_SALES":
            sales=get_sales(per); df.at[idx,"Value"] = sales * param_val
        elif drv=="PY_RATIO_SALES":
            try: py_sales=float(py.loc[("Sales","Revenue"), per])
            except Exception: py_sales=0.0
            ratio=(py_val/py_sales) if py_sales else 0.0
            cpi_mult=1.0+ctx.cpi_yoy
            df.at[idx,"Value"] = (py_sales*ratio)*cpi_mult
        elif drv=="CPI_INDEXED":
            df.at[idx,"Value"] = py_val*(1.0+ctx.cpi_yoy)
        elif drv=="OIL_LINKED_FREIGHT":
            sales=get_sales(per); df.at[idx,"Value"] = sales*param_val*ctx.oil_ratio
        elif drv=="FX_CONVERTED_SALES":
            base_amt=param_val; df.at[idx,"Value"] = base_amt*ctx.fx_eur_to_target
        else:
            df.at[idx,"Value"] = py_val

    # TOTAL = CALC + MANUAL_ADJ
    totals=[]
    for (acc,typ,per), sub in df.groupby(["Account","Type","Period"]):
        calc_val=float(sub.loc[sub["Component"]=="CALC","Value"].sum()) if (sub["Component"]=="CALC").any() else 0.0
        adj_val=float(sub.loc[sub["Component"]=="MANUAL_ADJ","Value"].sum()) if (sub["Component"]=="MANUAL_ADJ").any() else 0.0
        totals.append(((acc,typ,per), calc_val+adj_val))
    total_map=dict(totals)
    tmask = df["Component"]=="TOTAL"
    for idx,row in df[tmask].iterrows():
        key=(row["Account"],row["Type"],row["Period"])
        df.at[idx,"Value"] = float(total_map.get(key,0.0))
    return df

# ---------- UI ----------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Single grid with CALC + MANUAL_ADJ (+TOTAL read-only). Jan–Mar actuals locked. Recalculate updates CALC and Totals.")

with st.sidebar:
    st.header("Configuration")
    entity = st.text_input("Entity", value=DEFAULT_ENTITY)
    currency = st.selectbox("Currency", ["USD","EUR","GBP"], index=0)
    use_sheets = st.toggle("Use Google Sheets storage", value=False)
    if use_sheets:
        client = _gspread_client()
        if client is None:
            st.error("Google Sheets not configured.")
            sh, ws = None, {}
        else:
            sh, ws = _open_or_create(client, SHEET_NAME)
            st.success(f"Connected to '{SHEET_NAME}'.")
    else:
        sh, ws = None, {}

    st.divider(); st.subheader("External Drivers")
    cpi, _ = fetch_cpi_yoy()
    fx_rate, _ = fetch_fx_rate("EUR", currency)
    oil_ratio, _ = fetch_oil_index_ratio()

ctx = Ctx(entity=entity, currency=currency, cpi_yoy=cpi, fx_eur_to_target=fx_rate, oil_ratio=oil_ratio)

# Load data
try:
    prior_df = read_sheet_df(ws[TAB_PRIOR]) if ws else pd.DataFrame()
except Exception:
    prior_df = pd.DataFrame()
if prior_df.empty:
    prior_df = demo_prior()
prior_df["Period"]=prior_df["Period"].apply(_norm_period)
if "Type" not in prior_df.columns and not prior_df.empty:
    type_map = {d["Account"]: d["Type"] for d in DEFAULT_ACCOUNTS}
    prior_df["Type"] = prior_df["Account"].map(type_map).fillna("Revenue")
actuals_df = prior_df[prior_df["Period"].isin(ACTUALIZED_PERIODS)].copy()
actuals_pivot = actuals_df.pivot_table(index=["Account","Type"], columns="Period", values="Value", aggfunc="sum")

# unified input grid
try:
    unified = read_sheet_df(ws[TAB_INPUT]) if ws else pd.DataFrame()
except Exception:
    unified = pd.DataFrame()
required_cols = ["Account","Type","Component","Period","Driver","Param","Value"]
if unified.empty or not all(c in unified.columns for c in required_cols):
    unified = seed_unified_input()
else:
    for c in required_cols:
        if c not in unified.columns: unified[c] = "" if c!="Value" else 0.0
    unified["Period"]=unified["Period"].apply(_norm_period)
    unified["Value"]=pd.to_numeric(unified["Value"],errors="coerce").fillna(0.0)

# Display grid (CALC/ADJ/TOTAL) — lock actualized months (all components) and TOTAL always read-only
st.subheader("Unified Inputs & Results (single grid)")
piv = unified.pivot_table(index=["Account","Type","Component","Driver","Param"], columns="Period", values="Value", aggfunc="sum").reindex(columns=MONTHS, fill_value=0.0).reset_index()

comp_col = st.column_config.SelectboxColumn("Component", options=COMP_CHOICES, help="TOTAL is computed")
type_col = st.column_config.SelectboxColumn("Type", options=ACCOUNT_TYPES)
acc_col  = st.column_config.TextColumn("Account")
drv_col  = st.column_config.SelectboxColumn("Driver", options=DRIVERS, help="Used for CALC rows")
param_col= st.column_config.TextColumn("Param", help="CALC: driver parameter; MANUAL_ADJ: 'Adj'")

num_cols = {}
for m in MONTHS:
    disabled = m in ACTUALIZED_PERIODS  # lock actualized months for all components
    num_cols[m] = st.column_config.NumberColumn(m, disabled=disabled, format="%.2f", help=("actual (locked)" if disabled else "edit"))

grid = st.data_editor(
    piv,
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "Account":acc_col,"Type":type_col,"Component":comp_col,"Driver":drv_col,"Param":param_col,
        **num_cols
    },
    key="unified_grid",
)

st.caption("Edit **driver parameters** (CALC rows, future months) and **amounts** (MANUAL_ADJ). **TOTAL** is computed on recalc.")

col1,col2 = st.columns(2)
with col1:
    if st.button("Recalculate", type="primary"):
        # back to long
        long = grid.melt(id_vars=["Account","Type","Component","Driver","Param"], value_vars=[c for c in grid.columns if c in MONTHS], var_name="Period", value_name="Value")
        long["Value"]=pd.to_numeric(long["Value"],errors="coerce").fillna(0.0)
        out = recalc_components(prior_df, long, ctx, actuals_pivot)
        st.session_state["unified"] = out
        if ws:
            try: write_sheet_df(ws[TAB_INPUT], out)
            except Exception: ...

with col2:
    if st.button("Restore Demo Grid"):
        st.session_state.pop("unified", None)
        st.experimental_rerun()

st.subheader("Result (same grid)")
if "unified" in st.session_state:
    res = st.session_state["unified"].copy()
    show = res.pivot_table(index=["Account","Type","Component","Driver","Param"], columns="Period", values="Value", aggfunc="sum").reindex(columns=MONTHS, fill_value=0.0).reset_index()
    st.dataframe(show, use_container_width=True)
else:
    st.info("Click **Recalculate** to compute CALC and TOTAL values.")
# v7 END
