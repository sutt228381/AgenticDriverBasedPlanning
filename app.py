import io
import csv
import math
import streamlit as st
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple

st.set_page_config(page_title="Agentic Driver-Based Planning — v13.3.1", layout="wide")

APP_TITLE = "Agentic Driver-Based Planning — v13.3.1 (Simple UI • Per-Combo Drivers)"
MONTHS: List[str] = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# ---------------- Hierarchy ----------------
SECTIONS = [
    ("Revenue", ["Sales","Returns & Allowances","Royalty Income"]),
    ("COGS",    ["COGS","Freight","Fulfillment"]),
    ("Opex",    ["Marketing","Payroll","G&A","Depreciation"]),
    ("Other",   ["Other Income","Interest Expense"]),
    ("Taxes",   ["Taxes"]),
]
ACCOUNT_TO_SECTION = {acc:sec for sec, accs in SECTIONS for acc in accs}
KNOWN_ACCOUNTS = set(ACCOUNT_TO_SECTION.keys())

@dataclass
class LineMeta:
    driver: str
    param: float

DEFAULT_DRIVERS: Dict[str, LineMeta] = {
    "Sales": LineMeta("PCT_GROWTH", 0.02),
    "Returns & Allowances": LineMeta("PCT_OF_SALES", -0.02),
    "Royalty Income": LineMeta("PY_RATIO_SALES", 0.00),
    "COGS": LineMeta("PCT_OF_SALES", -0.45),
    "Freight": LineMeta("OIL_LINKED_FREIGHT", -0.02),
    "Fulfillment": LineMeta("PCT_OF_SALES", -0.015),
    "Marketing": LineMeta("PCT_OF_SALES", -0.07),
    "Payroll": LineMeta("CPI_INDEXED", 0.02),
    "G&A": LineMeta("CPI_INDEXED", 0.02),
    "Depreciation": LineMeta("MANUAL", 0.0),
    "Other Income": LineMeta("MANUAL", 0.0),
    "Interest Expense": LineMeta("MANUAL", 0.0),
    "Taxes": LineMeta("PCT_OF_SALES", -0.025),
}

# ---------------- Utils ----------------
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
def norm_period(x:str)->str:
    if x is None: return ""
    s=str(x).strip().lower()
    if s in MONTH_ALIASES: return MONTH_ALIASES[s]
    if s[:3] in MONTH_ALIASES: return MONTH_ALIASES[s[:3]]
    return s[:3].title()

def computed_lines(sec_map):
    out = {}
    out["Gross Profit"]     = sec_map.get("Revenue",0.0) - sec_map.get("COGS",0.0)
    out["Operating Income"] = out["Gross Profit"] - sec_map.get("Opex",0.0)
    out["Pre-Tax Income"]   = out["Operating Income"] + sec_map.get("Other",0.0)
    out["Net Income"]       = out["Pre-Tax Income"] - sec_map.get("Taxes",0.0)
    return out

# ---------------- Header ----------------
st.title(APP_TITLE)
st.caption("① Load Data → ② Drivers (suggest & select) → ③ Plan Grid (months across). Actuals lock by cutoff; drivers apply only to forecast months.")

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(["① Load Data", "② Drivers", "③ Plan Grid"])

with tab1:
    st.subheader("Load a dimensional CSV")
    uf = st.file_uploader("CSV file", type=["csv","txt"])
    st.caption("Header should include: Entity, Product, Channel, Currency, Account, Period, Value")

    def try_read_csv(file_bytes: bytes):
        """Try multiple parse strategies with useful errors and delimiter detection."""
        sample = file_bytes[:4096]
        try:
            dialect = csv.Sniffer().sniff(sample.decode("utf-8", errors="ignore"))
            guessed_sep = dialect.delimiter
        except Exception:
            guessed_sep = ","

        attempts = []
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            for sep in (guessed_sep, ",", ";", "\t"):
                for header in ("infer", None):
                    try:
                        df = pd.read_csv(
                            io.BytesIO(file_bytes),
                            sep=sep,
                            encoding=enc,
                            header=0 if header == "infer" else None,
                            engine="python",
                            on_bad_lines="skip",
                        )
                        if header is None:
                            if df.shape[1] >= 7:
                                df = df.iloc[:, :7]
                                df.columns = ["Entity","Product","Channel","Currency","Account","Period","Value"]
                        attempts.append(("OK", enc, sep, header))
                        return df, (enc, sep, header), attempts
                    except Exception as e:
                        attempts.append((repr(e), enc, sep, header))
                        continue
        return None, None, attempts

    if uf is not None:
        st.info(f"Selected: **{uf.name}** · {(uf.size/1024):.1f} KB")
        file_bytes = uf.read()
        df, meta, attempts = try_read_csv(file_bytes)

        with st.expander("Diagnostics (CSV parsing)", expanded=False):
            st.dataframe(pd.DataFrame(attempts, columns=["result","encoding","sep","header"]), use_container_width=True)

        if df is None or df.empty:
            st.error("Could not parse your file. Try saving as a comma-delimited CSV with a single header row.")
        else:
            expected = ["Entity","Product","Channel","Currency","Account","Period","Value"]
            cols = list(df.columns)
            if set(expected) - set(cols):
                st.warning("Columns don’t match expected names — map them below.")
                col_map = {}
                for want in expected:
                    col_map[want] = st.selectbox(f"Map to **{want}**", ["<none>"] + cols, index=(cols.index(want)+1) if want in cols else 0)
                norm = pd.DataFrame()
                for want in expected:
                    pick = col_map[want]
                    if pick == "<none>":
                        st.error(f"Please map a column to **{want}**.")
                        st.stop()
                    norm[want] = df[pick]
                df = norm

            # Normalize
            df["Entity"]   = df["Entity"].astype(str).str.strip()
            df["Product"]  = df["Product"].astype(str).str.strip()
            df["Channel"]  = df["Channel"].astype(str).str.strip()
            df["Currency"] = df["Currency"].astype(str).str.strip()
            df["Account"]  = df["Account"].astype(str).str.strip()
            df["Period"]   = df["Period"].apply(norm_period)
            df["Value"]    = pd.to_numeric(df["Value"], errors="coerce").fillna(0.0)
            df = df[df["Period"].isin(MONTHS)].copy()

            df["Section"] = df["Account"].map(lambda a: ACCOUNT_TO_SECTION.get(a, "Opex"))
            st.session_state["uploaded_df"] = df
            st.success(f"Parsed OK (encoding/sep/header = {meta}). Rows: {len(df):,}. Go to tab ② Drivers.")

with tab2:
    if "uploaded_df" not in st.session_state:
        st.info("Load a CSV in tab ① first.")
    else:
        src = st.session_state["uploaded_df"]

        st.subheader("Slice & Actuals")
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            ent = st.multiselect("Entity", sorted(src["Entity"].unique().tolist()), default=sorted(src["Entity"].unique().tolist()))
        with c2:
            prod= st.multiselect("Product", sorted(src["Product"].unique().tolist()), default=sorted(src["Product"].unique().tolist()))
        with c3:
            chan= st.multiselect("Channel", sorted(src["Channel"].unique().tolist()), default=sorted(src["Channel"].unique().tolist()))
        with c4:
            cutoff = st.selectbox("Actuals cutoff (locked)", MONTHS, index=2)
        locked_set = set(MONTHS[:MONTHS.index(cutoff)+1])

        sliced = src[src["Entity"].isin(ent) & src["Product"].isin(prod) & src["Channel"].isin(chan)].copy()

        # Build combo cube at (Section, Account, Channel, Product, Component)
        by_combo = sliced.pivot_table(index=["Section","Account","Channel","Product"],
                                      columns="Period", values="Value", aggfunc="sum").reindex(columns=MONTHS, fill_value=0.0)

        def seed_cube_from_combo(pvt: pd.DataFrame)->pd.DataFrame:
            rows=[]
            for (sec, acc, ch, pr), vals in pvt.iterrows():
                for comp in ["CALC","MANUAL_ADJ","TOTAL"]:
                    row={"Section":sec,"Account":acc,"Channel":ch,"Product":pr,"Component":comp}
                    for m in MONTHS:
                        row[m] = float(vals[m]) if comp=="CALC" else 0.0
                    rows.append(row)
            return pd.DataFrame(rows)

        if "combo_cube" not in st.session_state:
            st.session_state["combo_cube"] = seed_cube_from_combo(by_combo)

        combo_cube = st.session_state["combo_cube"]

        # Driver suggestions per combo with confidence & why
        st.subheader("Choose drivers per combo")
        st.caption("We recommend a driver (with confidence and 'why'). Accept all or adjust per row, then Apply.")
        def suggest(acc:str, channel:str, product:str)->Tuple[str,float,str]:
            a=acc.lower(); ch=str(channel).lower(); pr=str(product).lower()
            if "freight" in a or "postage" in a or ch in ("magazine","catalog"):
                return ("OIL_LINKED_FREIGHT", 0.8, "Shipping/postage sensitive by channel/account.")
            if a in ("payroll","g&a","general & administrative") or "salary" in a:
                return ("CPI_INDEXED", 0.7, "Labor/overhead tracks CPI.")
            if "marketing" in a or "ad" in a:
                return ("PCT_OF_SALES", 0.75, "Spend scales with revenue in digital channels.")
            if "return" in a:
                return ("PCT_OF_SALES", 0.85, "Returns as a % of gross sales.")
            if "royalty" in a:
                return ("PY_RATIO_SALES", 0.6, "Royalties track sales via stable ratio.")
            if acc=="COGS":
                return ("PCT_OF_SALES", 0.7, "COGS broadly scales with revenue at stable mix.")
            if acc=="Sales":
                return ("PCT_GROWTH", 0.6, "Revenue modeled via period growth.")
            return ("MANUAL", 0.4, "No strong signal; manual/bespoke driver.")

        combos = combo_cube[combo_cube["Account"].isin(KNOWN_ACCOUNTS)][["Account","Channel","Product"]].drop_duplicates().reset_index(drop=True)
        if len(combos)==0:
            st.warning("No known account rows found in this slice. Ensure your Accounts match the hierarchy names.")
        combos["Suggested"], combos["Confidence"], combos["Why"] = zip(*combos.apply(lambda r: suggest(r["Account"], r["Channel"], r["Product"]), axis=1))

        # Persist config
        if "driver_cfg" not in st.session_state:
            cfg = combos.copy()
            cfg["Driver"] = cfg["Suggested"]
            cfg["Param"]  = cfg["Account"].map(lambda a: DEFAULT_DRIVERS.get(a, LineMeta("MANUAL",0.0)).param)
            st.session_state["driver_cfg"] = cfg
        else:
            prev = st.session_state["driver_cfg"]
            cfg = pd.merge(combos, prev[["Account","Channel","Product","Driver","Param"]],
                           on=["Account","Channel","Product"], how="left")
            cfg["Driver"] = cfg["Driver"].fillna(cfg["Suggested"])
            cfg["Param"]  = cfg["Param"].fillna(0.0)
            st.session_state["driver_cfg"] = cfg

        cL, cR = st.columns([1,3])
        with cL:
            if st.button("Accept all suggestions"):
                st.session_state["driver_cfg"]["Driver"] = st.session_state["driver_cfg"]["Suggested"]
        with cR:
            st.write("")

        choices = ["MANUAL","PCT_GROWTH","PCT_OF_SALES","PY_RATIO_SALES","CPI_INDEXED","OIL_LINKED_FREIGHT","FX_CONVERTED_SALES"]
        edited_cfg = st.data_editor(
            st.session_state["driver_cfg"],
            use_container_width=True,
            num_rows="fixed",
            hide_index=True,
            column_config={
                "Driver": st.column_config.SelectboxColumn(options=choices),
                "Param": st.column_config.NumberColumn(help="Param meaning: % (0.02), growth (0.03), FX (1.05), etc."),
                "Confidence": st.column_config.NumberColumn(format="%.2f", disabled=True),
                "Suggested": st.column_config.TextColumn(disabled=True),
                "Why": st.column_config.TextColumn(disabled=True),
            }
        )
        st.session_state["driver_cfg"] = edited_cfg

        st.subheader("Global assumptions")
        c1,c2,c3,c4 = st.columns(4)
        with c1: cutoff = st.selectbox("Actuals cutoff", MONTHS, index=2, key="cutoff2")
        with c2: cpi_yoy = st.number_input("CPI YoY", value=0.02, step=0.005, format="%.3f")
        with c3: oil_idx = st.number_input("Oil/Postage index", value=1.00, step=0.05, format="%.2f")
        with c4: fx_rate = st.number_input("FX rate", value=1.05, step=0.01, format="%.2f")
        st.session_state["_globals"] = {"cutoff": cutoff, "cpi": cpi_yoy, "oil": oil_idx, "fx": fx_rate}

        def apply_drivers_per_combo(cube_df: pd.DataFrame, cfg_df: pd.DataFrame) -> pd.DataFrame:
            df = cube_df.copy()
            cutoff_idx = MONTHS.index(cutoff)
            # Ensure Sales TOTAL per combo
            for m in MONTHS:
                df.loc[(df["Account"]=="Sales")&(df["Component"]=="TOTAL"), m] = (
                    df.loc[(df["Account"]=="Sales")&(df["Component"]=="CALC"), m].values +
                    df.loc[(df["Account"]=="Sales")&(df["Component"]=="MANUAL_ADJ"), m].values
                )
            # Q1 ratio cache
            def q1_ratio(account:str, ch:str, pr:str)->float:
                acc_q1 = sum(float(df[(df["Account"]==account)&(df["Channel"]==ch)&(df["Product"]==pr)&(df["Component"]=="CALC")][m].sum()) for m in MONTHS[:3])
                sales_q1 = sum(float(df[(df["Account"]=="Sales")&(df["Channel"]==ch)&(df["Product"]==pr)&(df["Component"]=="CALC")][m].sum()) for m in MONTHS[:3])
                return (acc_q1 / sales_q1) if sales_q1 else 0.0
            ratio_cache: Dict[Tuple[str,str,str], float] = {}

            for _, r in cfg_df.iterrows():
                acc, ch, pr = r["Account"], r["Channel"], r["Product"]
                drv = str(r.get("Driver","MANUAL")).upper()
                prm = float(r.get("Param", 0.0))
                mask_calc = (df["Account"]==acc)&(df["Channel"]==ch)&(df["Product"]==pr)&(df["Component"]=="CALC")
                for idx, m in enumerate(MONTHS):
                    if idx <= cutoff_idx: continue
                    prev_m = MONTHS[idx-1] if idx>0 else m
                    prev_val = float(df.loc[mask_calc, prev_m].sum())
                    sales_m = float(df[(df["Account"]=="Sales")&(df["Channel"]==ch)&(df["Product"]==pr)&(df["Component"]=="TOTAL")][m].sum())
                    val = prev_val
                    if drv == "MANUAL":
                        val = prev_val
                    elif drv == "PCT_GROWTH":
                        val = prev_val * (1.0 + prm)
                    elif drv == "PCT_OF_SALES":
                        val = sales_m * prm
                    elif drv == "PY_RATIO_SALES":
                        key=(acc,ch,pr)
                        if key not in ratio_cache: ratio_cache[key]= q1_ratio(acc,ch,pr)
                        val = sales_m * ratio_cache[key] * (1.0 + prm)
                    elif drv == "CPI_INDEXED":
                        val = prev_val * (1.0 + cpi_yoy)
                    elif drv == "OIL_LINKED_FREIGHT":
                        val = sales_m * prm * oil_idx
                    elif drv == "FX_CONVERTED_SALES":
                        val = prev_val * fx_rate
                    df.loc[mask_calc, m] = float(val)
            # Recompute TOTAL
            for m in MONTHS:
                df.loc[df["Component"]=="TOTAL", m] = (
                    df.loc[df["Component"]=="CALC", m].values + df.loc[df["Component"]=="MANUAL_ADJ", m].values
                )
            return df

        if st.button("Apply drivers to forecast months"):
            new_cube = apply_drivers_per_combo(combo_cube, st.session_state["driver_cfg"])
            st.session_state["combo_cube"] = new_cube
            st.success("Drivers applied per combo. See tab ③.")

with tab3:
    if "uploaded_df" not in st.session_state or "combo_cube" not in st.session_state:
        st.info("Load data and set drivers first (tabs ① & ②).")
    else:
        cutoff = st.session_state.get("_globals",{}).get("cutoff","Mar")
        locked_set = set(MONTHS[:MONTHS.index(cutoff)+1])
        combo_cube = st.session_state["combo_cube"]

        # Aggregate for display
        agg = combo_cube.groupby(["Section","Account","Component"])[MONTHS].sum().reset_index()

        # Helpers
        def recalc(df: pd.DataFrame)->pd.DataFrame:
            out = df.copy()
            for m in MONTHS:
                out.loc[out["Component"]=="TOTAL", m] = (
                    out.loc[out["Component"]=="CALC", m].values + out.loc[out["Component"]=="MANUAL_ADJ", m].values
                )
            return out
        def section_totals(df_total: pd.DataFrame):
            res = {m:{} for m in MONTHS}
            for sec,_ in SECTIONS:
                mask = (df_total["Component"]=="TOTAL") & (df_total["Section"]==sec)
                for m in MONTHS:
                    res[m][sec] = float(df_total.loc[mask, m].sum())
            return res
        def build_display(df: pd.DataFrame) -> pd.DataFrame:
            df = recalc(df)
            rows=[]
            sec_sub = section_totals(df[df.Component=="TOTAL"])
            for sec, accounts in SECTIONS:
                parent={"Indent":0,"RowType":"PARENT","Line":sec}
                for m in MONTHS: parent[m]=float(sec_sub[m].get(sec,0.0))
                rows.append(parent)
                for acc in accounts:
                    tot={"Indent":1,"RowType":"LEAF_TOTAL","Line":acc}
                    for m in MONTHS: tot[m]=float(df[(df.Account==acc)&(df.Component=="TOTAL")][m].sum())
                    rows.append(tot)
                    calc={"Indent":2,"RowType":"LEAF_CALC","Line":f"{acc} · CALC"}
                    for m in MONTHS: calc[m]=float(df[(df.Account==acc)&(df.Component=="CALC")][m].sum())
                    rows.append(calc)
                    adj={"Indent":2,"RowType":"LEAF_ADJ","Line":f"{acc} · ADJ"}
                    for m in MONTHS: adj[m]=float(df[(df.Account==acc)&(df.Component=="MANUAL_ADJ")][m].sum())
                    rows.append(adj)
            # computed
            def comp_row(name):
                row={"Indent":0,"RowType":"COMPUTED","Line":name}
                for m in MONTHS:
                    sec_map = { s: float(df[(df.Component=="TOTAL")&(df.Section==s)][m].sum()) for s,_ in SECTIONS }
                    row[m]=computed_lines(sec_map)[name]
                return row
            for nm in ["Gross Profit","Operating Income","Pre-Tax Income","Net Income"]:
                rows.append(comp_row(nm))
            disp=pd.DataFrame(rows)
            def label(r):
                bullet = "» " if r["RowType"]=="PARENT" else ("  " * r["Indent"] + "- ")
                return ("  " * r["Indent"]) + bullet + r["Line"]
            disp["Account"] = disp.apply(label, axis=1)
            return disp[["Account"] + MONTHS + ["RowType","Line","Indent"]]

        st.subheader("P&L (months across)")
        display_df = build_display(agg)
        col_cfg = {m: st.column_config.NumberColumn(disabled=(m in locked_set)) for m in MONTHS}
        edited = st.data_editor(display_df, use_container_width=True, num_rows="fixed", hide_index=True, column_config=col_cfg)

        # Manual edits apportioned back to combos by share of CALC
        if st.button("Save manual edits"):
            before = display_df
            after = edited
            updated = combo_cube.copy()
            # shares
            shares = combo_cube[combo_cube["Component"]=="CALC"].groupby(["Account","Channel","Product"])[MONTHS].sum()
            acct_tot = shares.groupby(level=0)[MONTHS].sum()
            for idx in range(len(after)):
                rt = after.loc[idx,"RowType"]; label = after.loc[idx,"Line"]
                if rt not in ("LEAF_CALC","LEAF_ADJ"): continue
                comp = "CALC" if rt=="LEAF_CALC" else "MANUAL_ADJ"
                acc = label.replace(" · CALC","").replace(" · ADJ","")
                for m in MONTHS:
                    if m in locked_set: continue
                    delta = float(after.loc[idx,m]) - float(before.loc[idx,m])
                    if abs(delta) < 1e-9: continue
                    if acc in acct_tot.index and acct_tot.loc[acc,m] != 0:
                        for (a,ch,pr), row in shares.loc[acc].iterrows():
                            w = row[m] / acct_tot.loc[acc,m]
                            mask = (updated["Account"]==acc)&(updated["Channel"]==ch)&(updated["Product"]==pr)&(updated["Component"]==comp)
                            updated.loc[mask, m] = updated.loc[mask, m] + delta * float(w)
            # recompute totals
            for m in MONTHS:
                updated.loc[updated["Component"]=="TOTAL", m] = (
                    updated.loc[updated["Component"]=="CALC", m].values + updated.loc[updated["Component"]=="MANUAL_ADJ", m].values
                )
            st.session_state["combo_cube"] = updated
            st.success("Saved edits and recomputed totals.")
