import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import requests
import pandas as pd
import streamlit as st

# =============================
# CONFIG
# =============================
APP_TITLE = "Agentic Driver-Based Planning — P&L Grid v6"
SHEET_NAME = "AgenticPlanner"
TAB_PRIOR = "prior_year"       # actuals source
TAB_INPUT = "forecast_input"
TAB_SCENARIOS = "scenarios"
DEFAULT_ENTITY = "Orvis"
DEFAULT_CURRENCY = "USD"

# =============================
# Grid & Period helpers
# =============================
MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
# 🔒 Months that are actualized and locked in the grid
ACTUALIZED_PERIODS = ["Jan", "Feb", "Mar"]

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
def _normalize_period(val: str) -> str:
    if val is None:
        return ""
    s = str(val).strip().lower()
    if s in MONTH_ALIASES:
        return MONTH_ALIASES[s]
    s3 = s[:3]
    if s3 in MONTH_ALIASES:
        return MONTH_ALIASES[s3]
    valid_prefixes = {m.lower()[:3] for m in MONTHS}
    return s3.title() if s3 in valid_prefixes else s.title()

# =============================
# P&L schema
# =============================
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

COMPUTED_LINES = [
    ("Gross Profit", lambda sec: sec.get("Revenue",0) - sec.get("COGS",0)),
    ("Operating Income", lambda sec: (sec.get("Revenue",0)-sec.get("COGS",0)) - sec.get("Opex",0)),
    ("Pre-Tax Income", lambda sec: (sec.get("Revenue",0)-sec.get("COGS",0) - sec.get("Opex",0)) + sec.get("Other",0)),
    ("Net Income", lambda sec: (sec.get("Revenue",0)-sec.get("COGS",0) - sec.get("Opex",0) + sec.get("Other",0)) - sec.get("Taxes",0)),
]

DRIVER_CHOICES = ["MANUAL","PCT_GROWTH","PCT_OF_SALES","PY_RATIO_SALES","CPI_INDEXED","OIL_LINKED_FREIGHT","FX_CONVERTED_SALES"]

# =============================
# IO transforms
# =============================
def to_wide_input(long_df: pd.DataFrame) -> pd.DataFrame:
    base = long_df.copy()
    for c in ["Account","Type","Period","Driver","Param","Value"]:
        if c not in base.columns:
            base[c] = "" if c not in ["Value"] else 0.0
    base["Period"] = base["Period"].apply(_normalize_period)
    base["Value"] = pd.to_numeric(base["Value"], errors="coerce").fillna(0.0)
    piv = base.pivot_table(index=["Account","Type","Driver","Param"], columns="Period", values="Value", aggfunc="sum")
    piv = piv.reindex(columns=MONTHS, fill_value=0.0).reset_index()
    return piv.reindex(columns=["Account","Type","Driver","Param"] + MONTHS)

def to_long_input(wide_df: pd.DataFrame) -> pd.DataFrame:
    keep = ["Account","Type","Driver","Param"]
    for k in keep:
        if k not in wide_df.columns:
            wide_df[k] = ""
    val_cols = [c for c in wide_df.columns if c in MONTHS]
    melted = wide_df.melt(id_vars=keep, value_vars=val_cols, var_name="Period", value_name="Value")
    melted["Period"] = melted["Period"].apply(_normalize_period)
    melted["Value"] = pd.to_numeric(melted["Value"], errors="coerce").fillna(0.0)
    return melted.reindex(columns=["Account","Type","Period","Driver","Param","Value"])

# =============================
# OPTIONAL: Google Sheets
# =============================
@st.cache_resource(show_spinner=False)
def get_gspread_client():
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except Exception:
        return None
    if "gcp_service_account" not in st.secrets:
        return None
    creds_info = st.secrets["gcp_service_account"]
    scopes = ["https://www.googleapis.com/auth/spreadsheets","https://www.googleapis.com/auth/drive"]
    credentials = Credentials.from_service_account_info(creds_info, scopes=scopes)
    import gspread
    return gspread.authorize(credentials)

def get_or_create_sheet(client, spreadsheet_name=SHEET_NAME):
    if client is None:
        return None, None
    try:
        sh = client.open(spreadsheet_name)
    except Exception:
        sh = client.create(spreadsheet_name)
    for name in [TAB_PRIOR, TAB_INPUT, TAB_SCENARIOS]:
        try:
            sh.worksheet(name)
        except Exception:
            sh.add_worksheet(title=name, rows=500, cols=30)
    return sh, {TAB_PRIOR: sh.worksheet(TAB_PRIOR), TAB_INPUT: sh.worksheet(TAB_INPUT), TAB_SCENARIOS: sh.worksheet(TAB_SCENARIOS)}

def read_sheet_df(worksheet):
    values = worksheet.get_all_values()
    if not values:
        return pd.DataFrame()
    header, rows = values[0], values[1:]
    return pd.DataFrame(rows, columns=header)

def write_sheet_df(worksheet, df: pd.DataFrame):
    worksheet.clear()
    worksheet.update([df.columns.values.tolist()] + df.astype(str).values.tolist())

# =============================
# External Drivers (cached)
# =============================
@st.cache_data(ttl=3600)
def fetch_cpi_yoy() -> Tuple[float, str]:
    return 0.022, "CPI YoY (demo) = 2.2%"

@st.cache_data(ttl=3600)
def fetch_fx_rate(base: str, target: str) -> Tuple[float, str]:
    try:
        url = f"https://api.frankfurter.app/latest?from={base}&to={target}"
        resp = requests.get(url, timeout=10)
        if resp.ok:
            data = resp.json()
            rate = float(data["rates"][target])
            return rate, f"FX {base}->{target} = {rate:.4f} (Frankfurter)"
    except Exception:
        pass
    return 1.08, f"FX {base}->{target} fallback = 1.08"

@st.cache_data(ttl=3600)
def fetch_oil_index_ratio(prev_baseline: float = 75.0) -> Tuple[float, str]:
    current = 82.0
    ratio = current / max(prev_baseline, 1.0)
    return ratio, f"Oil index ratio = {current:.1f}/{prev_baseline:.1f} = {ratio:.3f} (demo)"

# =============================
# Demo seed data
# =============================
@st.cache_data
def demo_prior_year() -> pd.DataFrame:
    data = []
    for m, sales, royalty, cogs, freight in [
        ("Jan", 100000, 5000, 55000, 8000),
        ("Feb", 110000, 5500, 60500, 8300),
        ("Mar", 105000, 5250, 58000, 8200),
    ]:
        data += [
            {"Account":"Sales","Type":"Revenue","Period":m,"Value":sales},
            {"Account":"Royalty Income","Type":"Other","Period":m,"Value":royalty},
            {"Account":"COGS","Type":"COGS","Period":m,"Value":cogs},
            {"Account":"Freight","Type":"COGS","Period":m,"Value":freight},
        ]
    return pd.DataFrame(data)

@st.cache_data
def demo_forecast_input() -> pd.DataFrame:
    rows = []
    for base in DEFAULT_ACCOUNTS:
        acc, typ, drv, param = base["Account"], base["Type"], base["Driver"], base["Param"]
        for m in ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]:
            v = 0.0
            if acc=="Sales" and m in ["Jan","Feb","Mar"]:
                v = {"Jan":120000,"Feb":118000,"Mar":119500}[m]
            rows.append({"Account":acc,"Type":typ,"Period":m,"Driver":drv,"Param":param,"Value":v})
    return pd.DataFrame(rows)

# =============================
# Agentic Context & engine
# =============================
@dataclass
class Context:
    entity: str
    currency: str
    cpi_yoy: float
    fx_eur_to_target: float
    oil_ratio: float

def say(msg):
    st.session_state.setdefault("explain", [])
    st.session_state["explain"].append(msg)

def clear_explain():
    st.session_state.pop("explain", None)

def agent_recalculate(
    prior_df: pd.DataFrame,
    input_df: pd.DataFrame,
    ctx: Context,
    actuals_pivot: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    for df in (prior_df, input_df):
        if "Value" in df.columns:
            df["Value"] = pd.to_numeric(df["Value"], errors="coerce").fillna(0.0)
    py = prior_df.pivot_table(index=["Account","Type"], columns="Period", values="Value", aggfunc="sum").fillna(0.0)

    def get_sales(period: str) -> float:
        rows = input_df[(input_df["Account"] == "Sales") & (input_df["Period"] == period)]
        for _, r in rows.iterrows():
            if str(r.get("Driver","")).upper() == "MANUAL":
                try:
                    return float(r.get("Value", 0.0))
                except Exception:
                    pass
        try:
            return float(py.loc[("Sales","Revenue"), period])
        except Exception:
            return 0.0

    outputs: List[Dict] = []
    for _, row in input_df.iterrows():
        acc = str(row.get("Account",""))
        typ = str(row.get("Type",""))
        per = _normalize_period(str(row.get("Period","")))
        drv = str(row.get("Driver","")).upper().strip()
        try:
            pval = float(row.get("Value", 0.0))
        except Exception:
            pval = 0.0
        try:
            py_val = float(py.loc[(acc,typ), per])
        except Exception:
            py_val = 0.0

        # 🔒 Actualized months use actuals / prior
        if per in ACTUALIZED_PERIODS:
            actual_val = None
            if actuals_pivot is not None:
                try:
                    actual_val = float(actuals_pivot.loc[(acc,typ), per])
                except Exception:
                    try:
                        actual_val = float(actuals_pivot.xs(acc, level=0, drop_level=False)[per].iloc[0])
                    except Exception:
                        actual_val = None
            val = actual_val if actual_val is not None else py_val
            src = "Actual"
            say(f"{acc} {per}: using Actual = {val:,.2f}.")
            outputs.append({"Account": acc, "Type": typ, "Period": per, "Value": round(val, 2), "Source": src})
            continue

        if drv == "MANUAL":
            val = pval; src = "Manual Input"; say(f"{acc} {per}: set to {val:,.2f}.")
        elif drv == "PCT_GROWTH":
            val = py_val * (1.0 + pval); src = "PY*(1+Growth)"; say(f"{acc} {per}: {py_val:,.2f}*(1+{pval:.2%})={val:,.2f}")
        elif drv == "PCT_OF_SALES":
            sales = get_sales(per); val = sales * pval; src = "% of Sales"; say(f"{acc} {per}: {pval:.2%}*Sales {sales:,.2f}={val:,.2f}")
        elif drv == "PY_RATIO_SALES":
            try:
                py_sales = float(py.loc[("Sales","Revenue"), per])
            except Exception:
                py_sales = 0.0
            ratio = (py_val / py_sales) if py_sales else 0.0
            cpi_mult = 1.0 + ctx.cpi_yoy
            val = (py_sales * ratio) * cpi_mult; src = "Sales*(PY Ratio)*CPI"; say(f"{acc} {per}: ratio={ratio:.2%}, CPI={cpi_mult:.3f} → {val:,.2f}")
        elif drv == "CPI_INDEXED":
            cpi_mult = 1.0 + ctx.cpi_yoy; val = py_val * cpi_mult; src = "PY*(1+CPI)"; say(f"{acc} {per}: {py_val:,.2f}*(1+CPI {ctx.cpi_yoy:.2%})={val:,.2f}")
        elif drv == "OIL_LINKED_FREIGHT":
            sales = get_sales(per); val = sales * pval * ctx.oil_ratio; src = "%Sales*OilIndex"; say(f"{acc} {per}: {pval:.2%}*{sales:,.2f}*{ctx.oil_ratio:.3f}={val:,.2f}")
        elif drv == "FX_CONVERTED_SALES":
            base_amt = pval; val = base_amt * ctx.fx_eur_to_target; src = "FX(EUR→Target)"; say(f"{acc} {per}: {base_amt:,.2f} EUR*{ctx.fx_eur_to_target:.4f}={val:,.2f} {ctx.currency}")
        else:
            val = py_val; src = "Fallback=PY"; say(f"{acc} {per}: unknown driver {drv}, using PY {py_val:,.2f}.")
        outputs.append({"Account": acc, "Type": typ, "Period": per, "Value": round(val, 2), "Source": src})
    return pd.DataFrame(outputs)

# =============================
# P&L assembly helpers
# =============================
def build_pl_table(result_df: pd.DataFrame) -> pd.DataFrame:
    acc_per = result_df.groupby(["Account","Type","Period"])["Value"].sum().reset_index()
    wide = acc_per.pivot_table(index=["Type","Account"], columns="Period", values="Value", aggfunc="sum").reindex(columns=MONTHS, fill_value=0.0)
    wide["Total"] = wide.sum(axis=1)
    rows = []
    for section, accounts in PRESENTATION_ORDER:
        for acc in accounts:
            if (section, acc) in wide.index:
                rows.append((section, acc))
        existing = set(accounts)
        others = [a for (t,a) in wide.index if t==section and a not in existing]
        for acc in others:
            rows.append((section, acc))
    known_secs = {s for s,_ in PRESENTATION_ORDER}
    for t,a in wide.index:
        if (t not in known_secs) and (t in ACCOUNT_TYPES):
            rows.append((t,a))
    ordered = wide.loc[pd.MultiIndex.from_tuples(rows)] if rows else wide
    return ordered.reset_index()

def compute_section_totals(pl_table: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    totals_by_type = {}
    for t in pl_table["Type"].unique():
        block = pl_table[pl_table["Type"]==t].drop(columns=["Type","Account"]).sum(numeric_only=True)
        totals_by_type[t] = block
    return pl_table, totals_by_type

def add_computed_lines(pl_table: pd.DataFrame, totals_by_type: Dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, formula in COMPUTED_LINES:
        series = formula(totals_by_type)
        s = series.reindex(pl_table.columns.drop(["Type","Account"]), fill_value=0.0)
        row = {"Type":"Computed","Account":name}
        row.update({c: float(s.get(c, 0.0)) for c in pl_table.columns if c not in ["Type","Account"]})
        rows.append(row)
    comp_df = pd.DataFrame(rows)
    return pd.concat([pl_table, comp_df], ignore_index=True)

# =============================
# UI
# =============================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("P&L-style grid: Types (Revenue/COGS/Opex/Other/Taxes), Jan–Mar actuals locked, Apr–Dec editable. Subtotals + computed lines.")

with st.sidebar:
    st.header("Configuration")
    entity = st.text_input("Entity", value=DEFAULT_ENTITY)
    currency = st.selectbox("Currency", ["USD","EUR","GBP"], index=0)
    use_sheets = st.toggle("Use Google Sheets storage", value=False)
    if use_sheets:
        client = get_gspread_client()
        if client is None:
            st.error("Google Sheets not configured. Add secrets and gspread/google-auth in requirements.")
            sh, worksheets = None, None
        else:
            sh, worksheets = get_or_create_sheet(client)
            st.success(f"Connected to '{SHEET_NAME}'.")
    else:
        sh, worksheets = None, None

    st.divider(); st.subheader("External Drivers")
    cpi, cpi_info = fetch_cpi_yoy()
    fx_rate, fx_info = fetch_fx_rate("EUR", currency)
    oil_ratio, oil_info = fetch_oil_index_ratio()
    st.write(cpi_info); st.write(fx_info); st.write(oil_info)

ctx = Context(entity=entity, currency=currency, cpi_yoy=cpi, fx_eur_to_target=fx_rate, oil_ratio=oil_ratio)

# Load or seed data
if 'worksheets' in locals() and worksheets:
    try:
        prior_df = read_sheet_df(worksheets[TAB_PRIOR]); 
        if prior_df.empty: prior_df = demo_prior_year(); write_sheet_df(worksheets[TAB_PRIOR], prior_df)
    except Exception:
        prior_df = demo_prior_year()
    try:
        input_df = read_sheet_df(worksheets[TAB_INPUT]); 
        if input_df.empty: input_df = demo_forecast_input(); write_sheet_df(worksheets[TAB_INPUT], input_df)
    except Exception:
        input_df = demo_forecast_input()
else:
    prior_df = demo_prior_year(); input_df = demo_forecast_input()

# Normalize & build actuals (Jan–Mar)
prior_df["Period"] = prior_df["Period"].apply(_normalize_period)
if "Type" not in prior_df.columns and not prior_df.empty:
    type_map = {d["Account"]: d["Type"] for d in DEFAULT_ACCOUNTS}
    prior_df["Type"] = prior_df["Account"].map(type_map).fillna("Revenue")
actuals_df = prior_df[prior_df["Period"].isin(ACTUALIZED_PERIODS)].copy()
actuals_pivot = actuals_df.pivot_table(index=["Account","Type"], columns="Period", values="Value", aggfunc="sum")
actuals_pivot = actuals_pivot.reindex(columns=MONTHS, fill_value=0.0).fillna(0.0)

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["Inputs & Results", "Scenarios", "About"])

with tab1:
    st.subheader("Forecast Inputs (P&L grid)")
    if input_df.empty:
        input_df = demo_forecast_input()
    else:
        if "Type" not in input_df.columns:
            type_map = {d["Account"]: d["Type"] for d in DEFAULT_ACCOUNTS}
            input_df["Type"] = input_df["Account"].map(type_map).fillna("Revenue")

    wide_input = to_wide_input(input_df)

    have = set(wide_input["Account"])
    missing = [d for d in DEFAULT_ACCOUNTS if d["Account"] not in have]
    if missing:
        add_rows = [{"Account":d["Account"],"Type":d["Type"],"Driver":d["Driver"],"Param":d["Param"]} for d in missing]
        wide_input = pd.concat([wide_input, pd.DataFrame(add_rows)], ignore_index=True)

    if actuals_pivot is not None and not actuals_pivot.empty:
        wide_input = wide_input.set_index(["Account","Type","Driver","Param"])
        for m in ACTUALIZED_PERIODS:
            if m in wide_input.columns and m in actuals_pivot.columns:
                for (acc, typ) in actuals_pivot.index:
                    mask = (wide_input.index.get_level_values(0) == acc) & (wide_input.index.get_level_values(1) == typ)
                    if mask.any():
                        try:
                            wide_input.loc[mask, m] = float(actuals_pivot.loc[(acc, typ), m])
                        except Exception:
                            pass
        wide_input = wide_input.reset_index()

    driver_col = st.column_config.SelectboxColumn("Driver", options=DRIVER_CHOICES, help="Driver for this row")
    type_col = st.column_config.SelectboxColumn("Type", options=ACCOUNT_TYPES, help="P&L section")
    account_col = st.column_config.TextColumn("Account", help="Account (e.g., Sales, COGS, Marketing)")
    param_col = st.column_config.TextColumn("Param", help="Label like 'Growth%' or '% of Sales'")

    num_cols = {}
    for m in MONTHS:
        if m in ACTUALIZED_PERIODS:
            num_cols[m] = st.column_config.NumberColumn(m, help=f"{m} actual (locked)", disabled=True, format="%.2f")
        else:
            num_cols[m] = st.column_config.NumberColumn(m, help=f"Forecast for {m}", step=1.0, format="%.2f")

    wide_edited = st.data_editor(
        wide_input,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "Account": account_col,
            "Type": type_col,
            "Driver": driver_col,
            "Param":  param_col,
            **num_cols,
        },
        key="pl_input_grid",
    )
    st.caption(f"🔒 Actuals locked: {', '.join(ACTUALIZED_PERIODS)}.")

    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("Recalculate P&L", type="primary"):
            clear_explain()
            edited_long = to_long_input(wide_edited)
            st.session_state["input_editor"] = edited_long
            result = agent_recalculate(prior_df, edited_long, ctx, actuals_pivot=actuals_pivot)
            st.session_state["result_df"] = result
            if 'worksheets' in locals() and worksheets:
                write_sheet_df(worksheets[TAB_INPUT], edited_long)
                snap = result.copy(); snap["ScenarioName"] = f"Run {time.strftime('%Y-%m-%d %H:%M:%S')}"
                try:
                    scen_df = read_sheet_df(worksheets[TAB_SCENARIOS])
                    comb = pd.concat([scen_df, snap], ignore_index=True) if not scen_df.empty else snap
                    write_sheet_df(worksheets[TAB_SCENARIOS], comb)
                except Exception:
                    pass
    with colB:
        if st.button("Reset Explanations"):
            clear_explain()
    with colC:
        if st.button("Restore Demo Inputs"):
            demo = demo_forecast_input()
            st.session_state["input_editor"] = demo
            st.experimental_rerun()

    st.subheader("P&L (with section subtotals & computed lines)")
    result_df = st.session_state.get("result_df")
    if result_df is not None and not result_df.empty:
        pl_table = build_pl_table(result_df)
        pl_table, totals_by_type = compute_section_totals(pl_table)

        subtotal_rows = []
        for section, _ in PRESENTATION_ORDER:
            if section in totals_by_type:
                row = {"Type":"Subtotal","Account":f"{section} Subtotal"}
                for c in MONTHS + ["Total"]:
                    row[c] = float(totals_by_type[section].get(c, 0.0))
                subtotal_rows.append(row)
        if subtotal_rows:
            pl_table = pd.concat([pl_table, pd.DataFrame(subtotal_rows)], ignore_index=True)

        pl_table = add_computed_lines(pl_table, totals_by_type)
        st.dataframe(pl_table, use_container_width=True)
    else:
        st.info("Click 'Recalculate P&L' to compute results.")

    st.subheader("Explainability Pane")
    for msg in st.session_state.get("explain", []):
        st.write("•", msg)

with tab2:
    st.subheader("Saved Scenarios")
    if 'worksheets' in locals() and worksheets:
        try:
            scen_df = read_sheet_df(worksheets[TAB_SCENARIOS])
            if not scen_df.empty:
                st.dataframe(scen_df.tail(300), use_container_width=True)
                names = sorted(list(set(scen_df["ScenarioName"])))
                pick = st.selectbox("Compare scenario to current:", options=["(none)"] + names, index=0)
                if pick != "(none)" and st.session_state.get("result_df") is not None:
                    current = st.session_state["result_df"].copy()
                    comp = scen_df[scen_df["ScenarioName"] == pick][["Account","Period","Value"]].rename(columns={"Value":"Scenario"})
                    merged = current.merge(comp, on=["Account","Period"], how="left")
                    merged["Variance"] = merged["Value"] - merged["Scenario"]
                    st.dataframe(merged, use_container_width=True)
            else:
                st.caption("No scenarios saved yet. Recalculate once with Sheets enabled.")
        except Exception:
            st.warning("Couldn't load scenarios sheet.")
    else:
        st.info("Enable Google Sheets in the sidebar to persist scenarios.")

with tab3:
    st.markdown("""
**Included drivers**
- `MANUAL`, `PCT_GROWTH`, `PCT_OF_SALES`, `PY_RATIO_SALES` (with CPI)
- `CPI_INDEXED`, `OIL_LINKED_FREIGHT`, `FX_CONVERTED_SALES`

**Extend**
- Replace `fetch_cpi_yoy()` with BLS/FRED
- Add weather/commodity drivers
- Add Entities/Categories to the schema and UI filters
""")
