
from __future__ import annotations
import numpy as np
import pandas as pd

def _weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if mask.sum() == 0:
        return float("nan")
    x = x[mask]; w = w[mask]
    idx = np.argsort(x)
    x = x[idx]; w = w[idx]
    cw = np.cumsum(w)
    cutoff = 0.5 * cw[-1]
    return float(x[np.searchsorted(cw, cutoff)])

def _rolling_mean(a: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return a
    s = pd.Series(a).rolling(win, center=True, min_periods=max(2, win//3)).mean()
    return s.to_numpy()

def build_segments(
    df: pd.DataFrame,
    seg_len_s: int = 10,
    min_speed_mps: float = 0.5,
    flat_slope_range: tuple[float, float] = (-1.0, 1.0),
    slope_window_s: int = 15,
) -> pd.DataFrame:
    t = df["time"]
    dt = t.diff().dt.total_seconds().to_numpy()
    dt[0] = np.nan

    dist = df["dist_m"].to_numpy(float)
    alt = df["alt_m"].to_numpy(float)
    sp = df["speed_mps"].to_numpy(float)

    win = max(3, int(slope_window_s))
    alt_s = _rolling_mean(alt, win)

    d_alt = np.r_[np.nan, np.diff(alt_s)]
    d_dist = np.r_[np.nan, np.diff(dist)]
    slope = np.where((d_dist > 1.0) & np.isfinite(d_alt), 100.0 * d_alt / d_dist, np.nan)
    slope = pd.Series(slope).rolling(win, center=True, min_periods=max(2, win//3)).mean().to_numpy()
    slope = np.clip(slope, -30.0, 30.0)

    work = pd.DataFrame({
        "time": df["time"].values,
        "dt_s": dt,
        "speed_mps": sp,
        "slope_pct": slope,
    }).dropna(subset=["dt_s", "speed_mps", "slope_pct"]).reset_index(drop=True)

    work = work[(work["dt_s"] > 0.2) & (work["dt_s"] < 10.0)].copy()

    t0 = work["time"].iloc[0]
    work["t_rel_s"] = (work["time"] - t0).dt.total_seconds()
    work["seg_idx"] = (work["t_rel_s"] // seg_len_s).astype(int)

    def wavg(x, w):
        x = np.asarray(x, dtype=float)
        w = np.asarray(w, dtype=float)
        m = np.isfinite(x) & np.isfinite(w) & (w > 0)
        if m.sum() == 0:
            return np.nan
        return float(np.sum(x[m] * w[m]) / np.sum(w[m]))

    rows = []
    for seg_idx, g in work.groupby("seg_idx", sort=True):
        dt_sum = float(g["dt_s"].sum())
        if dt_sum < 0.6 * seg_len_s:
            continue
        speed_seg = wavg(g["speed_mps"].values, g["dt_s"].values)
        slope_seg = wavg(g["slope_pct"].values, g["dt_s"].values)
        rows.append({
            "t0": g["time"].iloc[0],
            "t1": g["time"].iloc[-1],
            "dt_s": dt_sum,
            "speed_mps": speed_seg,
            "slope_pct": slope_seg,
        })

    seg = pd.DataFrame(rows)
    seg = seg[seg["speed_mps"] >= float(min_speed_mps)].copy().reset_index(drop=True)

    lo, hi = flat_slope_range
    flat = seg[(seg["slope_pct"] >= lo) & (seg["slope_pct"] <= hi)]
    if len(flat) < 8:
        flat = seg.iloc[seg["slope_pct"].abs().sort_values().index[:max(10, min(50, len(seg)//4))]]

    flat_v = float(np.median(flat["speed_mps"].values))
    seg.attrs["flat_v_mps"] = flat_v
    seg["flat_v_mps"] = flat_v
    seg["ratio_vs_flat"] = seg["speed_mps"] / np.maximum(flat_v, 0.1)
    return seg

def build_reference_F(segs_all: pd.DataFrame, bin_width_pct: float = 1.0, F_min: float = 0.65, F_max: float = 1.75) -> pd.DataFrame:
    df = segs_all.copy()
    bw = float(bin_width_pct)
    df["slope_bin_pct"] = np.round(df["slope_pct"] / bw) * bw

    bins = []
    for sb, g in df.groupby("slope_bin_pct"):
        F = _weighted_median(g["ratio_vs_flat"].values, g["dt_s"].values)
        bins.append({"slope_bin_pct": float(sb), "F_ref": float(F), "n": int(len(g)), "time_s": float(g["dt_s"].sum())})
    out = pd.DataFrame(bins).sort_values("slope_bin_pct").reset_index(drop=True)

    if (out["slope_bin_pct"] == 0.0).any():
        F0 = float(out.loc[out["slope_bin_pct"] == 0.0, "F_ref"].iloc[0])
    else:
        idx0 = int(out["slope_bin_pct"].abs().idxmin())
        F0 = float(out.loc[idx0, "F_ref"])
    if np.isfinite(F0) and F0 > 0:
        out["F_ref"] = out["F_ref"] / F0

    out["F_ref"] = out["F_ref"].clip(F_min, F_max)

    s_min = float(out["slope_bin_pct"].min())
    s_max = float(out["slope_bin_pct"].max())
    full = pd.DataFrame({"slope_bin_pct": np.arange(np.floor(s_min), np.ceil(s_max) + bw, bw)})
    merged = full.merge(out[["slope_bin_pct", "F_ref"]], on="slope_bin_pct", how="left")
    merged["F_ref"] = merged["F_ref"].interpolate(limit_direction="both").clip(F_min, F_max)
    return merged

def interpolate_F(slope_pct: np.ndarray, F_table: pd.DataFrame, F_min: float = 0.65, F_max: float = 1.75) -> np.ndarray:
    x = F_table["slope_bin_pct"].to_numpy(float)
    y = F_table["F_ref"].to_numpy(float)
    s = np.asarray(slope_pct, dtype=float)
    out = np.interp(s, x, y, left=y[0], right=y[-1])
    return np.clip(out, F_min, F_max)

def estimate_glide_factor(test_seg: pd.DataFrame, slope_range: tuple[float, float] = (-5.0, 12.0)) -> float:
    lo, hi = slope_range
    g = test_seg[(test_seg["slope_pct"] >= lo) & (test_seg["slope_pct"] <= hi)].copy()
    if len(g) < 20:
        g = test_seg.copy()

    r = (g["speed_mps"] / np.maximum(g["flat_v_mps"], 0.1)) / np.maximum(g["F_ref"], 1e-6)
    k = _weighted_median(r.to_numpy(), g["dt_s"].to_numpy())
    if not np.isfinite(k) or k <= 0:
        k = float(np.nanmedian(r.to_numpy()))
    return float(k)

def compute_effort_speed(v_obs_mps: np.ndarray, F_ref: np.ndarray, k_glide: float) -> np.ndarray:
    v_obs_mps = np.asarray(v_obs_mps, dtype=float)
    F_ref = np.asarray(F_ref, dtype=float)
    k = float(k_glide) if np.isfinite(k_glide) and k_glide > 0 else 1.0
    return v_obs_mps / (k * np.maximum(F_ref, 1e-6))
