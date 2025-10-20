
import streamlit as st
import pandas as pd

st.set_page_config(page_title="P&L (WSP-Style) — v10", layout="wide")

# -------------------- THEME / CSS --------------------
WSP_BLUE = "#0B3D91"
WSP_GRAY = "#F4F6F8"
WSP_DARK = "#1F2937"

st.markdown(f"""
<style>
:root {{
  --wsp-blue: {WSP_BLUE};
  --wsp-gray: {WSP_GRAY};
  --wsp-dark: {WSP_DARK};
}}
html, body, [data-testid="stApp"] {{
  background: white;
}}
h1, h2, h3 {{
  color: var(--wsp-dark);
  margin-top: 0.25rem;
}}
.section-card {{
  border: 1px solid #E5E7EB;
  border-radius: 12px;
  padding: 1rem 1.25rem;
  background: white;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}}
.kpi {{
  background: var(--wsp-gray);
  border-radius: 10px;
  padding: 0.75rem 1rem;
  text-align: center;
}}
.kpi .label {{
  color: #6B7280; font-size: 0.8rem; text-transform: uppercase; letter-spacing: .08em;
}}
.kpi .value {{
  color: var(--wsp-dark); font-weight: 700; font-size: 1.15rem;
}}
table.pandastable {{
  border-collapse: collapse;
  width: 100%;
  font-size: 0.95rem;
}}
table.pandastable th, table.pandastable td {{
  border-bottom: 1px solid #E5E7EB;
  padding: 6px 8px;
}}
table.pandastable th {{
  background: var(--wsp-gray);
  color: var(--wsp-dark);
  font-weight: 700;
}}
.row-label {{
  font-weight: 600;
}}
.subtle {{
  color: #6B7280;
  font-size: 0.9rem;
}}
.input-blue input {{
  color: #0B3D91 !important;
  font-weight: 700 !important;
}}
.negative {{ color: #B91C1C; }}
.positive {{ color: #065F46; }}
</style>
""", unsafe_allow_html=True)

# -------------------- INTRO / HEADER --------------------
st.markdown("<h1>P&amp;L — Profit &amp; Loss (WSP-Style)</h1>", unsafe_allow_html=True)
st.caption("Based on Wall Street Prep’s structure: Net Revenue → COGS → Gross Profit → Opex → EBIT → Interest → EBT → Taxes → Net Income.")

# -------------------- INPUTS / CALCULATOR (like article's example) --------------------
with st.sidebar:
    st.header("Assumptions (blue = inputs)")
    st.markdown('<div class="subtle">All expenses entered as negatives.</div>', unsafe_allow_html=True)
    rev = st.number_input("Revenue", value=100_000_000.0, step=1_000_000.0, format="%.0f", key="rev", help="Top line (Net Revenue)")
    cogs = st.number_input("COGS", value=-40_000_000.0, step=1_000_000.0, format="%.0f", key="cogs", help="Cost of Goods Sold (enter negative)")
    sga = st.number_input("SG&A", value=-20_000_000.0, step=1_000_000.0, format="%.0f", key="sga", help="Operating Expenses (enter negative)")
    interest = st.number_input("Interest Expense", value=-5_000_000.0, step=500_000.0, format="%.0f", key="int", help="Non-operating item (enter negative)")
    tax_rate = st.number_input("Effective Tax Rate (%)", value=30.0, step=0.5, key="tax", help="Applied to EBT")

# -------------------- CALC --------------------
gross_profit = rev + cogs
ebit = gross_profit + sga
ebt = ebit + interest
taxes = -(max(ebt, 0.0) * (tax_rate/100.0))  # negative outflow
net_income = ebt + taxes

def fmt(n):
    return f"${n:,.0f}"

# KPI header
k1, k2, k3, k4 = st.columns(4)
with k1: st.markdown(f'<div class="kpi"><div class="label">Gross Profit</div><div class="value">{fmt(gross_profit)}</div></div>', unsafe_allow_html=True)
with k2: st.markdown(f'<div class="kpi"><div class="label">EBIT</div><div class="value">{fmt(ebit)}</div></div>', unsafe_allow_html=True)
with k3: st.markdown(f'<div class="kpi"><div class="label">EBT</div><div class="value">{fmt(ebt)}</div></div>', unsafe_allow_html=True)
with k4: st.markdown(f'<div class="kpi"><div class="label">Net Income</div><div class="value">{fmt(net_income)}</div></div>', unsafe_allow_html=True)

st.markdown("---")

# -------------------- SINGLE-PERIOD STATEMENT --------------------
st.subheader("Single-Period P&L (Calculator)")
calc_df = pd.DataFrame({
    "Line Item": [
        "Net Revenue",
        "Less: COGS",
        "Gross Profit",
        "Less: Operating Expenses (SG&A)",
        "Operating Income (EBIT)",
        "Less: Interest Expense",
        "Pre-Tax Income (EBT)",
        "Less: Income Taxes",
        "Net Income"
    ],
    "Amount": [
        rev,
        cogs,
        gross_profit,
        sga,
        ebit,
        interest,
        ebt,
        taxes,
        net_income
    ]
})

# Percent margins
calc_df["% of Revenue"] = (calc_df["Amount"] / rev).map(lambda x: f"{x*100:.1f}%") if rev != 0 else "—"

def style_amount(x):
    cls = "negative" if x < 0 else "positive"
    return f'<span class="{cls}">{fmt(x)}</span>'

styled_calc = calc_df.copy()
styled_calc["Amount"] = styled_calc["Amount"].map(style_amount)
styled_calc.rename(columns={"Line Item":"", "Amount":"Amount (USD)"}, inplace=True)

st.markdown(styled_calc.to_html(escape=False, index=False, classes="pandastable"), unsafe_allow_html=True)

st.markdown("---")

# -------------------- MULTI-PERIOD EDITABLE GRID --------------------
st.subheader("Multi-Period P&L (Editable)")

# Default grid: 12 months, inputs blue-ish, computed black (like the article convention)
months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
if "grid" not in st.session_state:
    st.session_state["grid"] = pd.DataFrame({
        "Line Item": ["Net Revenue", "COGS", "SG&A", "Interest Expense", "Tax Rate %"],
        **{m: [10_000_000, -4_000_000, -2_000_000, -500_000, 30.0] for m in months}
    })

grid = st.session_state["grid"].copy()

# Editor
edited = st.data_editor(
    grid,
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        **{m: st.column_config.NumberColumn() for m in months}
    }
)

# Recompute a derived statement with margins & subtotals for each month
def compute_statement(df):
    out_rows = []
    for m in months:
        rev = float(df.loc[df["Line Item"]=="Net Revenue", m])
        cogs = float(df.loc[df["Line Item"]=="COGS", m])
        sga = float(df.loc[df["Line Item"]=="SG&A", m])
        intr = float(df.loc[df["Line Item"]=="Interest Expense", m])
        tax_pct = float(df.loc[df["Line Item"]=="Tax Rate %", m])
        gross = rev + cogs
        ebit = gross + sga
        ebt = ebit + intr
        taxes = -(max(ebt,0.0) * (tax_pct/100.0))
        ni = ebt + taxes
        out_rows.append([m, rev, cogs, gross, sga, ebit, intr, ebt, taxes, ni])
    out = pd.DataFrame(out_rows, columns=["Period","Net Revenue","COGS","Gross Profit","SG&A","EBIT","Interest Expense","EBT","Taxes","Net Income"])
    # common-size
    for col in ["COGS","Gross Profit","SG&A","EBIT","Interest Expense","EBT","Taxes","Net Income"]:
        out[f"{col} %"] = (out[col] / out["Net Revenue"]).map(lambda x: f"{x*100:.1f}%" if pd.notnull(x) and x!=0 and out["Net Revenue"].ne(0).any() else "—")
    return out

derived = compute_statement(edited)

# Show two tabs: Statement and Common-Size
t1, t2 = st.tabs(["Statement", "Common-Size %"])

with t1:
    show_cols = ["Period","Net Revenue","COGS","Gross Profit","SG&A","EBIT","Interest Expense","EBT","Taxes","Net Income"]
    st.dataframe(derived[show_cols], use_container_width=True)

with t2:
    pct_cols = ["Period"] + [c for c in derived.columns if c.endswith("%")]
    st.dataframe(derived[pct_cols], use_container_width=True)

# Save edits
st.session_state["grid"] = edited

st.markdown("---")
st.caption("Notes: Inputs shown in blue on the left (sidebar). In tables, expenses are negative. Common-size percentages are relative to Net Revenue.")
