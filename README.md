# Agentic Driver-Based Planning — v13
**Upload a CSV → Slice by dimensions → Hierarchical P&L with months across → Driver suggestions.**

## CSV expectations
- Required: Account, Period (month name/number), Value
- Optional: Entity, Product, Channel, Currency
- Period is normalized (Jan/1/01/january → Jan, etc.)

## Run locally
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py

## Deploy (Streamlit Cloud)
- Main file: `app.py`
- Python: 3.12.5 (runtime.txt included)
