import streamlit as st
import pandas as pd
import numpy as np

from tcx_utils import load_tcx_to_df
from models import build_segments, build_reference_F_relative, build_reference_V_down_abs, interpolate_table, estimate_k_glide_from_downhill_abs, compute_effort_speed
from zones import parse_zone_edges_kmh, add_zone_column, zone_distribution

st.set_page_config(page_title="onFlows | Glide+Slope (2-model) Demo", layout="wide")
st.title("onFlows • Glide + Slope normalizer (2 models)")
st.caption("Downhill ABS model -> k_glide (global). Relative F(s) -> slope modulation. Filter speed < 2 km/h to cut stops/shooting.")

with st.sidebar:
    st.header("Inputs")
    ref_files = st.file_uploader("Reference TCX files (2+)", type=["tcx"], accept_multiple_files=True)
    test_file = st.file_uploader("Test TCX file (single)", type=["tcx"])
    st.divider()
    seg_len_s = st.number_input("Segment length (s)", 5, 60, 10, 1)
    min_speed_kmh = st.number_input("Min speed (km/h)", 0.0, 10.0, 2.0, 0.5)
    ref_slope_low = st.number_input("Flat slope low (%)", value=-1.0, step=0.5)
    ref_slope_high = st.number_input("Flat slope high (%)", value=1.0, step=0.5)
    st.divider()
    down_low = st.number_input("Downhill slope low (%)", value=-15.0, step=1.0)
    down_high = st.number_input("Downhill slope high (%)", value=-1.0, step=0.5)
    st.divider()
    F_min = st.number_input("F min clamp", value=0.65, step=0.05)
    F_max = st.number_input("F max clamp", value=1.75, step=0.05)
    st.divider()
    edges_txt = st.text_input("Zone edges (km/h)", value="12, 16, 20, 24, 28")
    zone_edges = parse_zone_edges_kmh(edges_txt)

min_speed_mps = float(min_speed_kmh)/3.6

def seg_from_upload(upl, name):
    df = load_tcx_to_df(upl)
    seg = build_segments(df, seg_len_s=int(seg_len_s), min_speed_mps=min_speed_mps, flat_slope_range=(float(ref_slope_low), float(ref_slope_high)))
    seg["activity"] = name
    return seg

if not ref_files or len(ref_files) < 2:
    st.info("Upload at least 2 reference TCX files.")
    st.stop()

with st.spinner("Building reference models…"):
    ref_segs = []
    ref_summ = []
    for f in ref_files:
        seg = seg_from_upload(f, f.name)
        ref_segs.append(seg)
        ref_summ.append({"activity": f.name, "n_segments": int(len(seg)), "flat_v_mps": float(seg.attrs.get("flat_v_mps", np.nan))})
    ref_df = pd.concat(ref_segs, ignore_index=True)
    F_table = build_reference_F_relative(ref_df, bin_width_pct=1.0, F_min=float(F_min), F_max=float(F_max))
    Vdown_table = build_reference_V_down_abs(ref_df, slope_range=(float(down_low), float(down_high)), bin_width_pct=1.0)

st.subheader("Reference models")
c1, c2 = st.columns(2)
with c1:
    st.write("Reference activities")
    st.dataframe(pd.DataFrame(ref_summ).sort_values("activity"), use_container_width=True, height=220)
with c2:
    st.write("Downhill ABS V_down_ref(s) (m/s)")
    st.dataframe(Vdown_table, use_container_width=True, height=220)

c3, c4 = st.columns(2)
with c3:
    st.write("Relative slope factor F_ref(s)")
    st.dataframe(F_table, use_container_width=True, height=220)
with c4:
    st.line_chart(F_table.set_index("slope_bin_pct")["F_ref"])
    if not Vdown_table.empty:
        st.line_chart(Vdown_table.set_index("slope_bin_pct")["v_down_ref_mps"])

if not test_file:
    st.info("Upload a test TCX file to compute k_glide and effort speed.")
    st.stop()

with st.spinner("Processing test activity…"):
    test_seg = seg_from_upload(test_file, test_file.name)
    test_seg["F_ref"] = interpolate_table(test_seg["slope_pct"].values, F_table, "slope_bin_pct", "F_ref", clip=(float(F_min), float(F_max)))
    test_seg["v_down_ref_mps"] = interpolate_table(test_seg["slope_pct"].values, Vdown_table, "slope_bin_pct", "v_down_ref_mps", clip=None)
    k_glide = estimate_k_glide_from_downhill_abs(test_seg, slope_range=(float(down_low), float(down_high)))
    test_seg["v_eff_mps"] = compute_effort_speed(test_seg["speed_mps"].values, test_seg["F_ref"].values, float(k_glide))

test_seg = add_zone_column(test_seg, "speed_mps", zone_edges, unit="mps", new_col="zone_raw")
test_seg = add_zone_column(test_seg, "v_eff_mps", zone_edges, unit="mps", new_col="zone_eff")

moving_time_s = float(test_seg["dt_s"].sum())
raw_mean = float(np.average(test_seg["speed_mps"], weights=test_seg["dt_s"]))
eff_mean = float(np.average(test_seg["v_eff_mps"], weights=test_seg["dt_s"]))
eff_dist = float((test_seg["v_eff_mps"] * test_seg["dt_s"]).sum())

st.subheader(f"Test: {test_file.name}")
m1, m2, m3, m4 = st.columns(4)
m1.metric("k_glide (downhill ABS)", f"{k_glide:.3f}")
m2.metric("Moving time", f"{moving_time_s/60:.1f} min")
m3.metric("Mean speed raw", f"{raw_mean*3.6:.2f} km/h")
m4.metric("Mean speed effort", f"{eff_mean*3.6:.2f} km/h")
st.metric("Effort distance (Σ v_eff·dt)", f"{eff_dist/1000:.2f} km")

st.write("Segments (filtered by min speed)")
cols = ["t0","t1","dt_s","slope_pct","speed_mps","F_ref","v_down_ref_mps","v_eff_mps","zone_raw","zone_eff"]
st.dataframe(test_seg[cols], use_container_width=True, height=420)

d1, d2 = st.columns(2)
with d1:
    st.write("Zones raw")
    zr = zone_distribution(test_seg, "zone_raw", "dt_s")
    st.dataframe(zr, use_container_width=True)
    st.bar_chart(zr.set_index("zone")["time_s"])
with d2:
    st.write("Zones effort")
    ze = zone_distribution(test_seg, "zone_eff", "dt_s")
    st.dataframe(ze, use_container_width=True)
    st.bar_chart(ze.set_index("zone")["time_s"])

st.write("Diagnostics")
a, b, c = st.columns(3)
with a:
    tmp = test_seg[["slope_pct","speed_mps"]].copy()
    tmp["speed_kmh"] = tmp["speed_mps"]*3.6
    st.scatter_chart(tmp, x="slope_pct", y="speed_kmh")
with b:
    tmp = test_seg[["slope_pct","v_eff_mps"]].copy()
    tmp["v_eff_kmh"] = tmp["v_eff_mps"]*3.6
    st.scatter_chart(tmp, x="slope_pct", y="v_eff_kmh")
with c:
    lo,hi=float(down_low),float(down_high)
    dd=test_seg[(test_seg["slope_pct"]>=lo)&(test_seg["slope_pct"]<=hi)].copy()
    if len(dd)>=5:
        dd["ratio_down"] = dd["speed_mps"]/np.maximum(dd["v_down_ref_mps"],1e-6)
        st.scatter_chart(dd, x="slope_pct", y="ratio_down")
    else:
        st.info("Not enough downhill segments after filters.")
