# Agentic Driver-Based Planning — Hierarchical P&L (No-AgGrid Stable)
- Single-grid hierarchical P&L
- Jan–Mar actuals locked; Apr–Dec editable
- SUBTOTAL & COMPUTED read-only; LINE TOTAL auto = CALC + MANUAL_ADJ
- Uses Streamlit's built-in editor (no aggrid)

## Local
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py

## Cloud
Set main file = `app.py`, then Reboot.
