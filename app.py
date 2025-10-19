import time
from dataclasses import dataclass
from typing import Tuple, Optional
import requests
import pandas as pd
import streamlit as st

APP_TITLE = "Agentic Driver-Based Planning — P&L Grid v9 (Hierarchical + Single Grid)"
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
    {"Account":"Other Income","Type":"Other","Driver":"MANUAL","Param":"Amount"},
    {"Account":"Interest Expense","Type":"Other","Driver":"MANUAL","Param":"Amount"},
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

@st.cache_data(ttl=3600)
def fetch_cpi_yoy()->Tuple[float,str]: 
    return 0.022, "CPI YoY (demo) = 2.2%"

@st.cache_data(ttl=3600)
def fetch_fx_rate(base:str,target:str)->Tuple[float,str]:
    try:
        r=requests.get(f"https://api.frankfurter.app/latest?from={base}&to={target}",timeout=10)
        if r.ok:
            rate=float(r.json()["rates"][target]); 
            return rate, f"FX {base}->{target} = {rate:.4f}"
    except Exception: ...
    return 1.08, "FX fallback = 1.08"

@st.cache_data(ttl=3600)
def fetch_oil_index_ratio(prev=75.0)->Tuple[float,str]:
    cur=82.0; ratio=cur/max(prev,1.0)
    return ratio, f"Oil index ratio = {cur:.1f}/{prev:.1f} = {ratio:.3f} (demo)"

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
        for m in MONTHS:
            base.append({
                "Order": order+0.5, "RowType":"SUBTOTAL", "Type":section, "Account":f"{section} Subtotal",
                "Component":"TOTAL","Period":m,"Driver":"LOCK","Param":"","Value":0.0
            })
    for i,(name,_) in enumerate(COMPUTED_LINES):
        for m in MONTHS:
            base.append({
                "Order": 900+i, "RowType":"COMPUTED","Type":"Computed",
                "Account":name,"Component":"TOTAL","Period":m,"Driver":"LOCK","Param":"","Value":0.0
            })
    df=pd.DataFrame(base)
    for m,v in {"Jan":120000,"Feb":118000,"Mar":119500}.items():
        mask=(df["RowType"]=="LINE")&(df["Account"]=="Sales")&(df["Component"]=="CALC")&(df["Period"]==m)
        df.loc[mask,"Value"]=float(v)
    return df

@dataclass
class Ctx:
    entity:str
    currency:str
    cpi_yoy:float
    fx_eur_to_target:float
    oil_ratio:float

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
            continue

        if   drv=="MANUAL":              df.at[idx,"Value"]=param
        elif drv=="PCT_GROWTH":          df.at[idx,"Value"]=py_val*(1.0+param)
        elif drv=="PCT_OF_SALES":        df.at[idx,"Value"]=get_sales(per)*param
        elif drv=="PY_RATIO_SALES":
            try: py_sales=float(py.loc[("Sales","Revenue"), per])
            except Exception: py_sales=0.0
            ratio=(py_val/py_sales) if py_sales else 0.0
            df.at[idx,"Value"]=(py_sales*ratio)*(1.0+ctx.cpi_yoy)
        elif drv=="CPI_INDEXED":         df.at[idx,"Value"]=py_val*(1.0+ctx.cpi_yoy)
        elif drv=="OIL_LINKED_FREIGHT":  df.at[idx,"Value"]=get_sales(per)*param*ctx.oil_ratio
        elif drv=="FX_CONVERTED_SALES":  df.at[idx,"Value"]=param*ctx.fx_eur_to_target
        else:                             df.at[idx,"Value"]=py_val

    for (acc,typ,per), sub in df[df["RowType"]=="LINE"].groupby(["Account","Type","Period"]):
        calc=float(sub.loc[sub["Component"]=="CALC","Value"].sum()) if (sub["Component"]=="CALC").any() else 0.0
        adj =float(sub.loc[sub["Component"]=="MANUAL_ADJ","Value"].sum()) if (sub["Component"]=="MANUAL_ADJ").any() else 0.0
        df.loc[(df["RowType"]=="LINE")&(df["Account"]==acc)&(df["Type"]==typ)&(df["Component"]=="TOTAL")&(df["Period"]==per),"Value"]=calc+adj

    for sec in ACCOUNT_TYPES:
        for m in MONTHS:
            subtotal=float(df[(df["RowType"]=="LINE")&(df["Type"]==sec)&(df["Component"]=="TOTAL")&(df["Period"]==m)]["Value"].sum())
            df.loc[(df["RowType"]=="SUBTOTAL")&(df["Type"]==sec)&(df["Period"]==m),"Value"]=subtotal

    for m in MONTHS:
        sec = { t: float(df[(df["RowType"]=="SUBTOTAL")&(df["Type"]==t)&(df["Period"]==m)]["Value"].sum()) for t in ACCOUNT_TYPES }
        for name, formula in COMPUTED_LINES:
            val=float(formula(sec))
            df.loc[(df["RowType"]=="COMPUTED")&(df["Account"]==name)&(df["Period"]==m),"Value"]=val
    return df

# UI
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Single grid with indentation & shading. Edit LINE rows for Apr–Dec. Jan–Mar locked. SUBTOTAL/COMPUTED read-only.")

with st.sidebar:
    st.header("Configuration")
    entity = st.text_input("Entity", value=DEFAULT_ENTITY)
    currency = st.selectbox("Currency", ["USD","EUR","GBP"], index=0)
    cpi, cpi_info = fetch_cpi_yoy()
    fx_rate, fx_info = fetch_fx_rate("EUR", currency)
    oil_ratio, oil_info = fetch_oil_index_ratio()
    st.write(cpi_info); st.write(fx_info); st.write(oil_info)

@dataclass
class Ctx:
    entity:str
    currency:str
    cpi_yoy:float
    fx_eur_to_target:float
    oil_ratio:float

ctx = Ctx(entity=entity, currency=currency, cpi_yoy=cpi, fx_eur_to_target=fx_rate, oil_ratio=oil_ratio)

prior_df = demo_prior()
prior_df["Period"]=prior_df["Period"].apply(_norm_period)
actuals_df = prior_df[prior_df["Period"].isin(ACTUALIZED_PERIODS)].copy()
actuals_pivot = actuals_df.pivot_table(index=["Account","Type"], columns="Period", values="Value", aggfunc="sum")

if "unified" not in st.session_state:
    st.session_state["unified"] = seed_unified()
unified = st.session_state["unified"].copy()

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode, JsCode

SHADE_SUBTOTAL = "#f3f4f6"
SHADE_COMPUTED = "#eef2ff"
SHADE_LINE     = "#ffffff"

def build_display_df(unified_df: pd.DataFrame) -> pd.DataFrame:
    wide = unified_df.pivot_table(
        index=["Order","RowType","Type","Account","Component","Driver","Param"],
        columns="Period",
        values="Value",
        aggfunc="sum"
    ).reindex(columns=MONTHS, fill_value=0.0).reset_index()
    def label_row(r):
        indent = "   " if r["RowType"] == "LINE" else ""
        base = f"{indent}{r['Account']}"
        if r["RowType"] == "LINE" and r["Component"] in ("CALC", "MANUAL_ADJ"):
            base += f"  · {r['Component']}"
        return base
    wide["Display"] = wide.apply(label_row, axis=1)
    wide = wide.sort_values(["Order","RowType","Account","Component"]).reset_index(drop=True)
    return wide

def aggrid_pl(wide_df: pd.DataFrame):
    numeric_months = [c for c in wide_df.columns if c in MONTHS]
    gb = GridOptionsBuilder.from_dataframe(
        wide_df[["Display","RowType","Type","Account","Component","Driver","Param"] + numeric_months],
        enableRowGroup=False, enableValue=False, enablePivot=False
    )
    for col in ["RowType","Type","Account","Component","Driver","Param"]:
        gb.configure_column(col, editable=False)
    gb.configure_column("Display", header_name="P&L", editable=False, pinned="left", width=320)
    editable_js = JsCode(f"""
        function(params) {{
            const rt = params.data.RowType;
            const comp = params.data.Component;
            const month = params.colDef.field;
            const locked = {ACTUALIZED_PERIODS};
            if (rt !== 'LINE') return false;
            if (comp === 'TOTAL') return false;
            if (locked.includes(month)) return false;
            return true;
        }}
    """)
    cell_style_js = JsCode(f"""
        function(params) {{
            const rt = params.data.RowType;
            let style = {{}};
            if (rt === 'SUBTOTAL') {{
                style = {{'backgroundColor': '{SHADE_SUBTOTAL}', 'fontWeight':'600'}};
            }} else if (rt === 'COMPUTED') {{
                style = {{'backgroundColor': '{SHADE_COMPUTED}', 'fontWeight':'600'}};
            }} else {{
                style = {{'backgroundColor': '{SHADE_LINE}'}};
            }}
            return style;
        }}
    """)
    for m in numeric_months:
        gb.configure_column(
            m,
            type=["numericColumn"],
            valueFormatter=JsCode("function(p){return (p.value===undefined||p.value===null)?'':Number(p.value).toLocaleString(undefined,{minimumFractionDigits:2, maximumFractionDigits:2});}"),
            editable=editable_js,
            cellStyle=cell_style_js
        )
    for meta in ["Display","RowType","Type","Account","Component","Driver","Param"]:
        gb.configure_column(meta, cellStyle=cell_style_js)
    gb.configure_grid_options(suppressMenuHide=True,rowSelection="single",animateRows=True,enableRangeSelection=True)
    grid_options = gb.build()
    grid_response = AgGrid(
        wide_df,
        gridOptions=grid_options,
        data_return_mode=DataReturnMode.AS_INPUT,
        update_mode=GridUpdateMode.VALUE_CHANGED,
        fit_columns_on_grid_load=False,
        enable_enterprise_modules=False,
        height=600
    )
    return grid_response["data"]

st.subheader("Unified P&L Grid")
display_df = build_display_df(unified)
edited_df = aggrid_pl(display_df)
st.caption("Shaded rows = SUBTOTAL (gray) & COMPUTED (indigo). Indented rows are accounts; CALC/MANUAL ADJ under each account. Edit Apr–Dec; Recalculate updates in-place.")

if st.button("Recalculate", type="primary"):
    long = edited_df.melt(
        id_vars=["Order","RowType","Type","Account","Component","Driver","Param","Display"],
        value_vars=[c for c in edited_df.columns if c in MONTHS],
        var_name="Period",
        value_name="Value"
    ).drop(columns=["Display"])
    long["Value"] = pd.to_numeric(long["Value"], errors="coerce").fillna(0.0)
    long["Period"]= long["Period"].apply(_norm_period)

    prev = st.session_state["unified"]
    join = ["Order","RowType","Type","Account","Component","Driver","Param","Period"]
    merged = long.merge(prev[join+["Value"]].rename(columns={"Value":"Prev"}), on=join, how="left")
    is_locked_month = merged["Period"].isin(ACTUALIZED_PERIODS)
    merged.loc[is_locked_month, "Value"] = merged.loc[is_locked_month, "Prev"].fillna(merged.loc[is_locked_month, "Value"])
    merged = merged.drop(columns=["Prev"])

    locked_mask = (merged["RowType"].isin(["SUBTOTAL","COMPUTED"])) | ((merged["RowType"]=="LINE") & (merged["Component"]=="TOTAL"))
    merged = merged.merge(prev[join+["Value"]].rename(columns={"Value":"Prev2"}), on=join, how="left")
    merged.loc[locked_mask, "Value"] = merged.loc[locked_mask, "Prev2"].fillna(merged.loc[locked_mask, "Value"])
    merged = merged.drop(columns=["Prev2"])

    # Compute & store
    out = recalc(merged, prior_df, ctx, actuals_pivot)
    st.session_state["unified"] = out

    # Rerun to refresh this same grid
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
