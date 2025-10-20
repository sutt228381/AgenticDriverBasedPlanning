# P&L (WSP-Style) — v10
This Streamlit app replicates Wall Street Prep's P&L layout and calculator feel:
- Net Revenue → COGS → Gross Profit → Opex (SG&A) → EBIT → Interest → EBT → Taxes → Net Income
- Single-period calculator with margins
- Multi-period editable P&L grid + common-size %

## Run locally
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py

## Deploy (Streamlit Cloud)
- Main file: `app.py`
- Python: 3.12.5 (runtime.txt included)
