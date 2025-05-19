
import streamlit as st
import pandas as pd
from pandas.errors import EmptyDataError
from optimization import optimize
from visualization import plot_network, summary

st.set_page_config(page_title="Warehouse Network Optimizer", layout="wide")

# ░░ Session state init ░░
if "scenarios" not in st.session_state:
    st.session_state["scenarios"] = {}

def _num_input(scn, key, label, default, fmt="%.4f", **kw):
    scn.setdefault(key, default)
    scn[key] = st.number_input(label, value=scn[key],
                               format=fmt, key=f"{key}_{scn['_name']}", **kw)

# ░░ Sidebar ░░
def sidebar(scn):
    name = scn['_name']
    with st.sidebar:
        st.header(f"Inputs — {name}")

        # Demand
        with st.expander("🗂️ Demand & Candidate Files", expanded=True):
            up = st.file_uploader("Store demand CSV (Longitude, Latitude, DemandLbs)",
                                  key=f"dem_{name}")
            if up:
                scn["demand_file"] = up
            if "demand_file" not in scn:
                st.info("Upload a demand file to begin.")
                return False

            if st.checkbox("Preview demand file", key=f"pre_{name}"):
                st.dataframe(pd.read_csv(scn["demand_file"]).head())

            cand_up = st.file_uploader("Candidate warehouse CSV (lon,lat[,cost/sqft])",
                                       key=f"cand_{name}")
            if cand_up is not None:
                if cand_up:
                    scn["cand_file"] = cand_up
                else:
                    scn.pop("cand_file", None)
            scn["restrict_cand"] = st.checkbox("Restrict to candidate sites",
                                               value=scn.get("restrict_cand", False),
                                               key=f"rc_{name}")

        # Transportation
        with st.expander("💰 Cost Parameters", expanded=False):
            st.subheader("Transportation $ / lb‑mile")
            _num_input(scn, "rate_out", "Outbound", 0.35)
            _num_input(scn, "in_rate", "Inbound", 0.30)
            _num_input(scn, "trans_rate", "Transfer (RDC → WH)", 0.32)

            st.subheader("Warehouse Cost Parameters")
            _num_input(scn, "sqft_per_lb", "Sq ft per lb", 0.02)
            _num_input(scn, "cost_sqft", "$/sq ft / yr", 6.0, "%.2f")
            _num_input(scn, "fixed_wh_cost", "Fixed $ per warehouse", 250000.0, "%.0f",
                       step=50000.0)

        # K selection
        with st.expander("🔢 Warehouse Count", expanded=False):
            scn["auto_k"] = st.checkbox("Optimize # warehouses",
                                        value=scn.get("auto_k", True),
                                        key=f"ak_{name}")
            if scn["auto_k"]:
                scn["k_rng"] = st.slider("k range", 1, 15,
                                         scn.get("k_rng", (3, 6)),
                                         key=f"kr_{name}")
            else:
                _num_input(scn, "k_fixed", "# warehouses", 4, "%.0f",
                           step=1, min_value=1, max_value=15)

        # Fixed WH & inbound
        with st.expander("📍 Locations", expanded=False):
            st.subheader("Fixed Warehouses")
            fixed_txt = st.text_area("lon,lat per line", value=scn.get("fixed_txt", ""),
                                     key=f"fx_{name}", height=80)
            scn["fixed_txt"] = fixed_txt
            fixed_centers = []
            for ln in fixed_txt.splitlines():
                try:
                    lon, lat = map(float, ln.split(","))
                    fixed_centers.append([lon, lat])
                except Exception:
                    continue
            scn["fixed_centers"] = fixed_centers

            st.subheader("Inbound Supply Points")
            scn["inbound_on"] = st.checkbox("Include inbound flow",
                                            value=scn.get("inbound_on", False),
                                            key=f"inb_{name}")
            inbound_pts = []
            if scn["inbound_on"]:
                sup_txt = st.text_area("lon,lat,percent (0‑100) per line",
                                       value=scn.get("sup_txt", ""),
                                       key=f"sup_{name}", height=100)
                scn["sup_txt"] = sup_txt
                for ln in sup_txt.splitlines():
                    try:
                        lon, lat, pct = map(float, ln.split(","))
                        inbound_pts.append([lon, lat, pct/100.0])
                    except Exception:
                        continue
            scn["inbound_pts"] = inbound_pts

        # RDC / SDC
        with st.expander("🏬 RDC / SDC (up to 3)", expanded=False):
            for idx in range(1,4):
                cols = st.columns([1,4])
                en = cols[0].checkbox(f"{idx}", key=f"rdc_en_{name}_{idx}",
                                      value=scn.get(f"rdc{idx}_en", False))
                scn[f"rdc{idx}_en"] = en
                if en:
                    with cols[1]:
                        lon = st.number_input("Longitude", key=f"lon_{name}_{idx}",
                                              value=float(scn.get(f"rdc{idx}_lon", 0.0)),
                                              format="%.6f")
                        lat = st.number_input("Latitude", key=f"lat_{name}_{idx}",
                                              value=float(scn.get(f"rdc{idx}_lat", 0.0)),
                                              format="%.6f")
                        typ = st.radio("Type", ["RDC","SDC"], horizontal=True,
                                       key=f"typ_{name}_{idx}",
                                       index=0 if scn.get(f"rdc{idx}_typ","RDC")=="RDC" else 1)
                        scn[f"rdc{idx}_lon"] = lon
                        scn[f"rdc{idx}_lat"] = lat
                        scn[f"rdc{idx}_typ"] = typ
            _num_input(scn, "rdc_sqft_per_lb", "RDC Sq ft per lb",
                       scn.get("sqft_per_lb",0.02))
            _num_input(scn, "rdc_cost_sqft", "RDC $/sq ft / yr",
                       scn.get("cost_sqft",6.0),"%.2f")

        st.markdown("---")
        if st.button("🚀 Run solver", key=f"run_{name}"):
            st.session_state["run_target"] = name
    return True

# ░░ Main ░░
tab_names = list(st.session_state["scenarios"].keys()) + ["➕ New scenario"]
tabs = st.tabs(tab_names)

# existing scenarios
for i, tab in enumerate(tabs[:-1]):
    name = tab_names[i]
    scn = st.session_state["scenarios"][name]
    scn["_name"] = name
    with tab:
        if not sidebar(scn):
            continue

        if scn.get("auto_k",True):
            k_vals = list(range(int(scn["k_rng"][0]), int(scn["k_rng"][1])+1))
        else:
            k_vals = [int(scn["k_fixed"])]

        if st.session_state.get("run_target")==name:
            with st.spinner("Running optimization…"):
                df = pd.read_csv(scn["demand_file"])
                df = df.dropna(subset=["Longitude","Latitude","DemandLbs"])
                candidate_sites = None
                candidate_costs = None
                if scn.get("cand_file"):
                    try:
                        cf = pd.read_csv(scn["cand_file"], header=None)
                        cf = cf.dropna(subset=[0,1])
                        if scn.get("restrict_cand"):
                            candidate_sites = cf.iloc[:,:2].values.tolist()
                        if cf.shape[1]>=3:
                            candidate_costs = { (round(r[0],6),round(r[1],6)):r[2]
                                                for r in cf.itertuples(index=False)}
                    except EmptyDataError:
                        scn.pop("cand_file",None)

                res = optimize(
                    df=df,
                    k_vals=k_vals,
                    rate_out=scn["rate_out"],
                    sqft_per_lb=scn["sqft_per_lb"],
                    cost_sqft=scn["cost_sqft"],
                    fixed_cost=scn["fixed_wh_cost"],
                    consider_inbound=scn["inbound_on"],
                    inbound_rate_mile=scn["in_rate"],
                    inbound_pts=scn["inbound_pts"],
                    fixed_centers=scn["fixed_centers"],
                    rdc_list=[{"coords":[scn[f"rdc{i}_lon"],scn[f"rdc{i}_lat"]],
                               "is_sdc":scn.get(f"rdc{i}_typ","RDC")=="SDC"}
                              for i in range(1,4) if scn.get(f"rdc{i}_en")],
                    transfer_rate_mile=scn["trans_rate"],
                    rdc_sqft_per_lb=scn["rdc_sqft_per_lb"],
                    rdc_cost_per_sqft=scn["rdc_cost_sqft"],
                    candidate_sites=candidate_sites,
                    restrict_cand=scn.get("restrict_cand",False),
                    candidate_costs=candidate_costs,
                )
            plot_network(res["assigned"], res["centers"])
            summary(res["assigned"], res["total_cost"], res["out_cost"],
                    res["in_cost"], res["trans_cost"], res["wh_cost"],
                    res["centers"], res["demand_per_wh"],
                    scn["sqft_per_lb"], bool(res.get("rdc_list")),
                    scn["inbound_on"], res["trans_cost"]>0)

# new scenario tab
with tabs[-1]:
    new_name = st.text_input("Scenario name")
    if st.button("Create scenario"):
        if new_name and new_name not in st.session_state["scenarios"]:
            st.session_state["scenarios"][new_name] = {}
            st.experimental_rerun()
