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
APP_TITLE = "Agentic Driver-Based Planning — Actualized + Grid v4"
SHEET_NAME = "AgenticPlanner"
TAB_PRIOR = "prior_year"       # legacy name; this is your actuals source
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

DRIVER_CHOICES = [
    "MANUAL",
    "PCT_GROWTH",
    "PCT_OF_SALES",
    "PY_RATIO_SALES",
    "CPI_INDEXED",
    "OIL_LINKED_FREIGHT",
    "FX_CONVERTED_SALES",
]

def to_wide_input(long_df: pd.DataFrame) -> pd.DataFrame:
    base = long_df.copy()
    for c in ["Account","Period","Driver","Param","Value"]:
        if c not in base.columns:
            base[c] = "" if c != "Value" else 0.0
    base["Period"] = base["Period"].apply(_normalize_period)
    base["Value"] = pd.to_numeric(base["Value"], errors="coerce").fillna(0.0)
    piv = base.pivot_table(
        index=["Account","Driver","Param"],
        columns="Period",
        values="Value",
        aggfunc="sum",
    )
    piv = piv.reindex(columns=MONTHS, fill_value=0.0).reset_index()
    ordered = ["Account","Driver","Param"] + MONTHS
    return piv.reindex(columns=ordered)

def to_long_input(wide_df: pd.DataFrame) -> pd.DataFrame:
    keep = ["Account","Driver","Param"]
    for k in keep:
        if k not in wide_df.columns:
            wide_df[k] = ""
    val_cols = [c for c in wide_df.columns if c in MONTHS]
    melted = wide_df.melt(
        id_vars=keep,
        value_vars=val_cols,
        var_name="Period",
        value_name="Value",
    )
    melted["Period"] = melted["Period"].apply(_normalize_period)
    melted["Value"] = pd.to_numeric(melted["Value"], errors="coerce").fillna(0.0)
    return melted.reindex(columns=["Account","Period","Driver","Param","Value"])

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
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
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
    return sh, {
        TAB_PRIOR: sh.worksheet(TAB_PRIOR),
        TAB_INPUT: sh.worksheet(TAB_INPUT),
        TAB_SCENARIOS: sh.worksheet(TAB_SCENARIOS),
    }

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
            {"Account":"Sales","Period":m,"Value":sales},
            {"Account":"Royalty Income","Period":m,"Value":royalty},
            {"Account":"COGS","Period":m,"Value":cogs},
            {"Account":"Freight","Period":m,"Value":freight},
        ]
    return pd.DataFrame(data)

@st.cache_data
def demo_forecast_input() -> pd.DataFrame:
    data = [
        {"Account":"Sales","Period":"Jan","Driver":"MANUAL","Param":"Amount","Value":120000},
        {"Account":"Royalty Income","Period":"Jan","Driver":"PY_RATIO_SALES","Param":"","Value":0},
        {"Account":"COGS","Period":"Jan","Driver":"PCT_GROWTH","Param":"Growth%","Value":0.06},
        {"Account":"Freight","Period":"Jan","Driver":"OIL_LINKED_FREIGHT","Param":"% of Sales","Value":0.07},

        {"Account":"Sales","Period":"Feb","Driver":"MANUAL","Param":"Amount","Value":118000},
        {"Account":"Royalty Income","Period":"Feb","Driver":"PY_RATIO_SALES","Param":"","Value":0},
        {"Account":"COGS","Period":"Feb","Driver":"PCT_GROWTH","Param":"Growth%","Value":0.05},
        {"Account":"Freight","Period":"Feb","Driver":"OIL_LINKED_FREIGHT","Param":"% of Sales","Value":0.07},

        {"Account":"Sales","Period":"Mar","Driver":"MANUAL","Param":"Amount","Value":119500},
        {"Account":"Royalty Income","Period":"Mar","Driver":"PY_RATIO_SALES","Param":"","Value":0},
        {"Account":"COGS","Period":"Mar","Driver":"PCT_GROWTH","Param":"Growth%","Value":0.05},
        {"Account":"Freight","Period":"Mar","Driver":"OIL_LINKED_FREIGHT","Param":"% of Sales","Value":0.07},
    ]
    return pd.DataFrame(data)

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
    py = prior_df.pivot_table(index="Account", columns="Period", values="Value", aggfunc="sum").fillna(0.0)

    def get_sales(period: str) -> float:
        rows = input_df[(input_df["Account"] == "Sales") & (input_df["Period"] == period)]
        for _, r in rows.iterrows():
            if str(r.get("Driver","")).upper() == "MANUAL":
                try:
                    return float(r.get("Value", 0.0))
                except Exception:
                    pass
        return float(py.loc["Sales", period]) if ("Sales" in py.index and period in py.columns) else 0.0

    outputs: List[Dict] = []
    for _, row in input_df.iterrows():
        acc = str(row.get("Account",""))
        per = _normalize_period(str(row.get("Period","")))
        drv = str(row.get("Driver","")).upper().strip()
        try:
            pval = float(row.get("Value", 0.0))
        except Exception:
            pval = 0.0
        py_val = float(py.loc[acc, per]) if (acc in py.index and per in py.columns) else 0.0

        # 🔒 Bypass drivers for actualized months: use actuals if available
        if per in ACTUALIZED_PERIODS:
            actual_val = None
            if actuals_pivot is not None:
                try:
                    actual_val = float(actuals_pivot.loc[acc, per])
                except Exception:
                    actual_val = None
            val = actual_val if actual_val is not None else py_val
            src = "Actual"
            say(f"{acc} {per}: using Actual = {val:,.2f}.")
            outputs.append({"Account": acc, "Period": per, "Value": round(val, 2), "Source": src})
            continue

        if drv == "MANUAL":
            val = pval; src = "Manual Input"; say(f"{acc} {per}: set to {val:,.2f}.")
        elif drv == "PCT_GROWTH":
            val = py_val * (1.0 + pval); src = "PY*(1+Growth)"; say(f"{acc} {per}: {py_val:,.2f}*(1+{pval:.2%})={val:,.2f}")
        elif drv == "PCT_OF_SALES":
            sales = get_sales(per); val = sales * pval; src = "% of Sales"; say(f"{acc} {per}: {pval:.2%}*Sales {sales:,.2f}={val:,.2f}")
        elif drv == "PY_RATIO_SALES":
            py_sales = float(py.loc["Sales", per]) if ("Sales" in py.index and per in py.columns) else 0.0
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
        outputs.append({"Account": acc, "Period": per, "Value": round(val, 2), "Source": src})
    return pd.DataFrame(outputs)

# =============================
# UI
# =============================
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Actualized Jan–Mar (locked) + Grid inputs for Apr–Dec. Periods normalized, driver picklists per row.")

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
if worksheets:
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

# === Build Actuals (Jan–Mar) from prior_df ===
prior_df["Period"] = prior_df["Period"].apply(_normalize_period)
actuals_df = prior_df[prior_df["Period"].isin(ACTUALIZED_PERIODS)].copy()
actuals_pivot = actuals_df.pivot_table(index="Account", columns="Period", values="Value", aggfunc="sum")
actuals_pivot = actuals_pivot.reindex(columns=MONTHS, fill_value=0.0).fillna(0.0)

tab1, tab2, tab3 = st.tabs(["Actuals & Inputs", "Scenarios", "About"])

with tab1:
    # 1) Actuals table (replaces the old "Prior Year" table)
    st.subheader("1) Actualized (locked) — Jan–Mar")
    if actuals_df.empty:
        st.info("No actuals found for Jan–Mar. Populate the actuals source (tab 'prior_year').")
    else:
        st.dataframe(actuals_df.sort_values(["Account","Period"]), use_container_width=True)

    # 2) Forecast inputs grid
    st.subheader("2) Forecast Inputs (accounts ↓, periods →)")
    wide_input = to_wide_input(input_df)

    # Overlay actuals into the grid and lock those columns
    if not actuals_pivot.empty:
        missing_accounts = sorted(set(actuals_pivot.index) - set(wide_input["Account"]))
        if missing_accounts:
            add_rows = [{"Account": acc, "Driver": "MANUAL", "Param": ""} for acc in missing_accounts]
            if add_rows:
                wide_input = pd.concat([wide_input, pd.DataFrame(add_rows)], ignore_index=True)

        wide_input = wide_input.set_index(["Account","Driver","Param"])
        for m in ACTUALIZED_PERIODS:
            if m in wide_input.columns and m in actuals_pivot.columns:
                # write the same actual value across all rows for that account (common FP&A pattern)
                for acc in actuals_pivot.index:
                    mask = wide_input.index.get_level_values(0) == acc
                    if mask.any():
                        try:
                            wide_input.loc[mask, m] = float(actuals_pivot.loc[acc, m])
                        except Exception:
                            pass
        wide_input = wide_input.reset_index()

    driver_col = st.column_config.SelectboxColumn("Driver", options=DRIVER_CHOICES, help="Choose a driver for this account row")
    account_col = st.column_config.TextColumn("Account", help="Account name (e.g., Sales, COGS, Freight, Royalty Income)")
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
            "Driver": driver_col,
            "Param":  param_col,
            **num_cols,
        },
        key="input_grid",
    )
    st.caption(f"🔒 Actuals locked: {', '.join(ACTUALIZED_PERIODS)}. Edit Apr–Dec to plan.")

    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("Recalculate Forecast", type="primary"):
            clear_explain()
            edited_long = to_long_input(wide_edited)
            st.session_state["input_editor"] = edited_long
            result = agent_recalculate(prior_df, edited_long, ctx, actuals_pivot=actuals_pivot)
            st.session_state["result_df"] = result
            if worksheets:
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

    st.subheader("3) Recalculated Forecast")
    result_df = st.session_state.get("result_df")
    if result_df is not None:
        st.dataframe(result_df, use_container_width=True)
        by_acc = result_df.groupby("Account")["Value"].sum().reset_index()
        st.caption("Totals by Account (current run)"); st.dataframe(by_acc, use_container_width=True)
    else:
        st.info("Click 'Recalculate Forecast' to compute results.")

    st.subheader("4) Explainability Pane")
    for msg in st.session_state.get("explain", []):
        st.write("•", msg)

with tab2:
    st.subheader("Saved Scenarios")
    if worksheets:
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
