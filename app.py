import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import requests
import pandas as pd
import streamlit as st

APP_TITLE = "Agentic Driver-Based Planning — Full MVP v1"
SHEET_NAME = "AgenticPlanner"
TAB_PRIOR = "prior_year"
TAB_INPUT = "forecast_input"
TAB_SCENARIOS = "scenarios"
DEFAULT_ENTITY = "Orvis"
DEFAULT_CURRENCY = "USD"

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

@st.cache_data(ttl=3600)
def fetch_cpi_yoy():
    return 0.022, "CPI YoY (demo) = 2.2%"

@st.cache_data(ttl=3600)
def fetch_fx_rate(base: str, target: str):
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
def fetch_oil_index_ratio(prev_baseline: float = 75.0):
    current = 82.0
    ratio = current / max(prev_baseline, 1.0)
    return ratio, f"Oil index ratio = {current:.1f}/{prev_baseline:.1f} = {ratio:.3f} (demo)"

@st.cache_data
def demo_prior_year() -> pd.DataFrame:
    data = [
        {"Account":"Sales","Period":"Jan","Value":100000},
        {"Account":"Royalty Income","Period":"Jan","Value":5000},
        {"Account":"COGS","Period":"Jan","Value":55000},
        {"Account":"Freight","Period":"Jan","Value":8000},
        {"Account":"Sales","Period":"Feb","Value":110000},
        {"Account":"Royalty Income","Period":"Feb","Value":5500},
        {"Account":"COGS","Period":"Feb","Value":60500},
        {"Account":"Freight","Period":"Feb","Value":8300},
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
    ]
    return pd.DataFrame(data)

from dataclasses import dataclass
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

def agent_recalculate(prior_df: pd.DataFrame, input_df: pd.DataFrame, ctx: Context) -> pd.DataFrame:
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

    outputs = []
    for _, row in input_df.iterrows():
        acc = str(row.get("Account",""))
        per = str(row.get("Period",""))
        drv = str(row.get("Driver","")).upper().strip()
        try:
            pval = float(row.get("Value", 0.0))
        except Exception:
            pval = 0.0
        py_val = float(py.loc[acc, per]) if (acc in py.index and per in py.columns) else 0.0

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

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Fuller MVP with additional drivers, scenarios, optional Google Sheets, and explainability.")

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

tab1, tab2, tab3 = st.tabs(["Inputs & Results", "Scenarios", "About"])

with tab1:
    st.subheader("1) Prior Year (reference)"); st.dataframe(prior_df, use_container_width=True)
    st.subheader("2) Forecast Inputs"); 
    edited_input = st.data_editor(input_df, use_container_width=True, num_rows="dynamic", key="input_editor")
    colA, colB, colC = st.columns(3)
    with colA:
        if st.button("Recalculate Forecast", type="primary"):
            st.session_state.pop("explain", None)
            result = agent_recalculate(prior_df, edited_input, ctx)
            st.session_state["result_df"] = result
            if worksheets:
                write_sheet_df(worksheets[TAB_INPUT], edited_input)
                snap = result.copy(); snap["ScenarioName"] = f"Run {time.strftime('%Y-%m-%d %H:%M:%S')}"
                try:
                    scen_df = read_sheet_df(worksheets[TAB_SCENARIOS])
                    comb = pd.concat([scen_df, snap], ignore_index=True) if not scen_df.empty else snap
                    write_sheet_df(worksheets[TAB_SCENARIOS], comb)
                except Exception: pass
    with colB:
        if st.button("Reset Explanations"): st.session_state.pop("explain", None)
    with colC:
        if st.button("Restore Demo Inputs"):
            demo = demo_forecast_input(); st.session_state["input_editor"] = demo

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
                st.dataframe(scen_df.tail(200), use_container_width=True)
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
