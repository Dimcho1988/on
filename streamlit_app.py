
import streamlit as st
import pandas as pd
import numpy as np

from tcx_utils import load_tcx_to_df
from slope_model import (
    build_segments,
    build_reference_F,
    interpolate_F,
    estimate_glide_factor,
    compute_effort_speed,
)
from zones import parse_zone_edges_kmh, add_zone_column, zone_distribution

st.set_page_config(page_title="onFlows | Slope+Glide Normalizer (Demo)", layout="wide")

st.title("onFlows • Slope + Glide normalizer (10 s segments)")
st.caption("Build a reference slope factor F(s) from multiple ski activities, then estimate session glide factor k and compute effort speed (slope+glide removed).")

with st.sidebar:
    st.header("Inputs")
    ref_files = st.file_uploader(
        "Reference TCX files (multiple) — used to build F_ref(s)",
        type=["tcx"],
        accept_multiple_files=True,
    )
    test_file = st.file_uploader("Test TCX file (single) — compute k_glide and effort speed", type=["tcx"])

    st.divider()
    st.subheader("Segmentation & filters")
    seg_len_s = st.number_input("Segment length (s)", min_value=5, max_value=60, value=10, step=1)
    min_speed_mps = st.number_input("Min speed (m/s) to keep segment", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
    ref_slope_low = st.number_input("Flat reference slope low (%)", value=-1.0, step=0.5)
    ref_slope_high = st.number_input("Flat reference slope high (%)", value=1.0, step=0.5)
    valid_slope_low = st.number_input("Valid slope low (%) for k_glide", value=-5.0, step=1.0)
    valid_slope_high = st.number_input("Valid slope high (%) for k_glide", value=12.0, step=1.0)

    st.divider()
    st.subheader("Slope factor limits")
    F_min = st.number_input("F(s) min clamp", value=0.65, step=0.05)
    F_max = st.number_input("F(s) max clamp", value=1.75, step=0.05)

    st.divider()
    st.subheader("Zones")
    edges_txt = st.text_input(
        "Zone edges in km/h (comma-separated). Example: 12, 16, 20, 24, 28",
        value="12, 16, 20, 24, 28",
    )
    zone_edges = parse_zone_edges_kmh(edges_txt)

def _load_and_segment(uploaded, name: str):
    df = load_tcx_to_df(uploaded)
    seg = build_segments(
        df,
        seg_len_s=int(seg_len_s),
        min_speed_mps=float(min_speed_mps),
        flat_slope_range=(float(ref_slope_low), float(ref_slope_high)),
    )
    seg["activity"] = name
    return df, seg

if not ref_files or len(ref_files) < 2:
    st.info("Upload 2+ reference TCX files to build the slope factor F_ref(s).")
    st.stop()

# Load reference files
ref_segs = []
ref_summ = []
with st.spinner("Parsing & segmenting reference activities…"):
    for f in ref_files:
        _, seg = _load_and_segment(f, f.name)
        ref_segs.append(seg)
        ref_summ.append({
            "activity": f.name,
            "n_segments": int(len(seg)),
            "flat_v_mps": float(seg.attrs.get("flat_v_mps", np.nan)),
        })

ref_segs_df = pd.concat(ref_segs, ignore_index=True)

# Build reference slope factor
with st.spinner("Building reference slope factor F_ref(s)…"):
    F_table = build_reference_F(
        ref_segs_df,
        bin_width_pct=1.0,
        F_min=float(F_min),
        F_max=float(F_max),
    )

st.subheader("Reference model")
c1, c2 = st.columns([1, 2])
with c1:
    st.write("**Reference activities summary**")
    st.dataframe(pd.DataFrame(ref_summ).sort_values("activity"), use_container_width=True)
with c2:
    st.write("**F_ref(s) table (1% bins)**")
    st.dataframe(F_table, use_container_width=True)

st.line_chart(F_table.set_index("slope_bin_pct")["F_ref"])

if not test_file:
    st.info("Now upload a test TCX file to compute k_glide and effort speed.")
    st.stop()

# Load test file
with st.spinner("Parsing & segmenting test activity…"):
    test_df, test_seg = _load_and_segment(test_file, test_file.name)

# Interpolate F on test segments
test_seg["F_ref"] = interpolate_F(test_seg["slope_pct"].values, F_table, F_min=float(F_min), F_max=float(F_max))

# Estimate glide factor k
k_glide = estimate_glide_factor(
    test_seg,
    slope_range=(float(valid_slope_low), float(valid_slope_high)),
)

# Compute effort speed
test_seg["v_eff_mps"] = compute_effort_speed(
    v_obs_mps=test_seg["speed_mps"].values,
    F_ref=test_seg["F_ref"].values,
    k_glide=float(k_glide),
)

# Zones
test_seg = add_zone_column(test_seg, "speed_mps", zone_edges, unit="mps", new_col="zone_raw")
test_seg = add_zone_column(test_seg, "v_eff_mps", zone_edges, unit="mps", new_col="zone_eff")

dist_raw = zone_distribution(test_seg, "zone_raw", weight_col="dt_s")
dist_eff = zone_distribution(test_seg, "zone_eff", weight_col="dt_s")

st.subheader(f"Test activity: {test_file.name}")

m1, m2, m3, m4 = st.columns(4)
moving_time_s = float(test_seg["dt_s"].sum())
raw_mean_mps = float(np.average(test_seg["speed_mps"], weights=test_seg["dt_s"]))
eff_mean_mps = float(np.average(test_seg["v_eff_mps"], weights=test_seg["dt_s"]))
eff_dist_m = float((test_seg["v_eff_mps"] * test_seg["dt_s"]).sum())

m1.metric("k_glide (session)", f"{k_glide:.3f}")
m2.metric("Moving time (filtered)", f"{moving_time_s/60:.1f} min")
m3.metric("Mean speed (raw, filtered)", f"{raw_mean_mps*3.6:.2f} km/h")
m4.metric("Mean speed (effort, filtered)", f"{eff_mean_mps*3.6:.2f} km/h")

st.caption("Effort distance = Σ(v_eff * dt) : distance equivalent on flat, average glide.")
st.metric("Effort distance (equiv.)", f"{eff_dist_m/1000:.2f} km")

# Segment table
st.write("### Segment table (10 s)")
show_cols = ["t0", "t1", "dt_s", "slope_pct", "speed_mps", "F_ref", "v_eff_mps", "zone_raw", "zone_eff"]
st.dataframe(test_seg[show_cols], use_container_width=True, height=380)

# Distributions
d1, d2 = st.columns(2)
with d1:
    st.write("### Zone distribution (raw speed)")
    st.dataframe(dist_raw, use_container_width=True)
    st.bar_chart(dist_raw.set_index("zone")["time_s"])
with d2:
    st.write("### Zone distribution (effort speed)")
    st.dataframe(dist_eff, use_container_width=True)
    st.bar_chart(dist_eff.set_index("zone")["time_s"])

# Diagnostics
st.write("### Diagnostic: speed vs slope (raw) and v_eff vs slope")
sc1, sc2 = st.columns(2)
with sc1:
    dfp = test_seg[["slope_pct", "speed_mps"]].copy()
    dfp["speed_kmh"] = dfp["speed_mps"] * 3.6
    st.scatter_chart(dfp, x="slope_pct", y="speed_kmh")
with sc2:
    dfp = test_seg[["slope_pct", "v_eff_mps"]].copy()
    dfp["v_eff_kmh"] = dfp["v_eff_mps"] * 3.6
    st.scatter_chart(dfp, x="slope_pct", y="v_eff_kmh")

with st.expander("How it works (short)"):
    st.markdown("""
- Build **F_ref(s)** from reference activities using 10 s segments:
  - Per activity flat baseline: median speed where slope in **[-1%, +1%]**
  - Per segment: **ratio = speed / flat_speed**
  - Aggregate ratios across activities by **1% slope bins** using a robust (weighted median).
- For the test activity:
  - Interpolate **F_ref(s)** for each segment
  - Estimate session **k_glide = median( (speed/flat_speed) / F_ref(s) )** on valid slopes (default [-5%, +12%])
  - Compute effort speed: **v_eff = speed / (k_glide * F_ref(s))**
- Segments with speed < 0.5 m/s are removed to cut shooting/stop phases.
""")
