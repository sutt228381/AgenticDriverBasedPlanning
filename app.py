import os
import json
import time
from dataclasses import dataclass
from typing import Dict, Any, Tuple

import requests
import pandas as pd
import streamlit as st

# =============================
# CONFIG & CONSTANTS
# =============================
APP_TITLE = "Agentic Driver-Based Planning — MVP"
SHEET_NAME = "AgenticPlanner"
TAB_PRIOR = "prior_year"
TAB_INPUT = "forecast_input"
TAB_SCENARIOS = "scenarios"

DEFAULT_ENTITY = "Orvis"
DEFAULT_CURRENCY = "USD"

# =============================
# UTIL: Google Sheets (optional)
# =============================
@st.cache_resource(show_spinner=False)
def get_gspread_client():
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except Exception as e:
        st.warning("gspread / google-auth not installed.")
        return None

    if "gcp_service_account" not in st.secrets:
        return None

    creds_info = st.secrets["gcp_service_account"]
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    credentials = Credentials.from_service_account_info(creds_info, scopes=scopes)
    client = gspread.authorize(credentials)
    return client


def get_or_create_sheet(client, spreadsheet_name=SHEET_NAME):
    if client is None:
        return None, None
    try:
        sh = client.open(spreadsheet_name)
    except Exception:
        sh = client.create(spreadsheet_name)
    for wks in [TAB_PRIOR, TAB_INPUT, TAB_SCENARIOS]:
        try:
            sh.worksheet(wks)
        except Exception:
            sh.add_worksheet(title=wks, rows=100, cols=20)
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
# External Drivers (CPI, FX)
# =============================
@st.cache_data(ttl=3600)
def fetch_cpi_yoy() -> Tuple[float, str]:
    try:
        return 0.022, "CPI YoY (demo) = 2.2%"
    except Exception:
        return 0.022, "CPI YoY fallback = 2.2%"


@st.cache_data(ttl=3600)
def fetch_fx_rate(base: str = "EUR", target: str = "USD") -> Tuple[float, str]:
    try:
        url = f"https://api.frankfurter.app/latest?from={base}&to={target}"
        resp = requests.get(url, timeout=10)
        if resp.ok:
            data = resp.json()
            rate = data["rates"][target]
            return float(rate), f"FX {base}->{target} = {rate} (Frankfurter)"
    except Exception:
        pass
    return 1.08, f"FX {base}->{target} fallback = 1.08"


# =============================
# Agentic Logic
# =============================
@dataclass
class Context:
    entity: str
    currency: str
    cpi_yoy: float
    fx_eur_usd: float


def explain(msgs):
    st.session_state.setdefault("explain", [])
    st.session_state["explain"].extend(msgs if isinstance(msgs, list) else [msgs])


def agent_recalculate(prior_df: pd.DataFrame, input_df: pd.DataFrame, ctx: Context) -> pd.DataFrame:
    for df in (prior_df, input_df):
        if 'Value' in df.columns:
            df['Value'] = pd.to_numeric(df['Value'], errors='coerce').fillna(0.0)
    py = prior_df.pivot_table(index='Account', columns='Period', values='Value', aggfunc='sum').fillna(0.0)
    outputs = []
    for _, row in input_df.iterrows():
        account = row.get('Account', '')
        period = row.get('Period', '')
        driver = (row.get('Driver', '') or '').strip()
        try:
            param_val = float(row.get('Value', 0.0))
        except Exception:
            param_val = 0.0
        py_val = py.loc[account, period] if account in py.index and period in py.columns else 0.0
        if driver.upper() == 'MANUAL':
            val = param_val
            src = 'Manual Input'
            explain(f"{account} {period}: set manually to {val:,.2f}.")
        elif driver.upper() == 'PCT_GROWTH':
            val = py_val * (1.0 + param_val)
            src = f"PY * (1+Growth%)"
            explain([f"{account} {period}: {py_val:,.2f} * (1+{param_val:.2%}) = {val:,.2f}"])
        elif driver.upper() == 'PCT_OF_SALES':
            sales_rows = input_df[(input_df['Account'] == 'Sales') & (input_df['Period'] == period)]
            sales_val = None
            for _, srow in sales_rows.iterrows():
                if (srow.get('Driver','').upper() == 'MANUAL'):
                    try:
                        sales_val = float(srow.get('Value', 0.0))
                        break
                    except Exception:
                        pass
            sales_val = sales_val if sales_val is not None else py.loc['Sales', period] if 'Sales' in py.index and period in py.columns else 0.0
            val = sales_val * param_val
            src = "% of Sales"
            explain([f"{account} {period}: {param_val:.2%} of Sales {sales_val:,.2f} = {val:,.2f}"])
        elif driver.upper() == 'PY_RATIO_SALES':
            py_sales = py.loc['Sales', period] if 'Sales' in py.index and period in py.columns else 0.0
            ratio = (py_val / py_sales) if py_sales else 0.0
            cpi_mult = 1.0 + ctx.cpi_yoy
            val = (py_sales * ratio) * cpi_mult
            src = "Sales * (PY Ratio) * CPI"
            explain([f"{account} {period}: ratio={ratio:.2%}, CPI={cpi_mult:.3f}, value={val:,.2f}"])
        else:
            val = py_val
            src = "Fallback = PY"
            explain(f"{account} {period}: unknown driver {driver}, fallback {py_val:,.2f}.")
        outputs.append({'Account': account,'Period': period,'Value': round(val, 2),'Source': src})
    return pd.DataFrame(outputs)


@st.cache_data
def demo_prior_year() -> pd.DataFrame:
    return pd.DataFrame([
        {"Account":"Sales","Period":"Jan","Value":100000},
        {"Account":"Royalty Income","Period":"Jan","Value":5000},
        {"Account":"COGS","Period":"Jan","Value":55000}
    ])

@st.cache_data
def demo_forecast_input() -> pd.DataFrame:
    return pd.DataFrame([
        {"Account":"Sales","Period":"Jan","Driver":"MANUAL","Value":120000},
        {"Account":"Royalty Income","Period":"Jan","Driver":"PY_RATIO_SALES","Value":0},
        {"Account":"COGS","Period":"Jan","Driver":"PCT_GROWTH","Value":0.06}
    ])

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    entity = st.text_input("Entity", value=DEFAULT_ENTITY)
    currency = st.selectbox("Currency", ["USD","EUR"], index=0)
    cpi, cpi_info = fetch_cpi_yoy()
    fx, fx_info = fetch_fx_rate("EUR", currency)
    st.write(cpi_info)
    st.write(fx_info)
ctx = Context(entity=entity, currency=currency, cpi_yoy=cpi, fx_eur_usd=fx)
prior_df = demo_prior_year()
input_df = demo_forecast_input()
st.subheader("Prior Year")
st.dataframe(prior_df)
st.subheader("Forecast Inputs")
edited_input = st.data_editor(input_df, use_container_width=True)
if st.button("Recalculate"):
    st.session_state.pop("explain", None)
    result = agent_recalculate(prior_df, edited_input, ctx)
    st.session_state["result_df"] = result
st.subheader("Forecast Result")
if "result_df" in st.session_state:
    st.dataframe(st.session_state["result_df"])
else:
    st.info("Click Recalculate")
st.subheader("Explainability")
for e in st.session_state.get("explain", []):
    st.write("•", e)
