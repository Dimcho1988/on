from __future__ import annotations
import numpy as np
import pandas as pd

def _weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, float); w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w>0)
    if m.sum()==0: return float("nan")
    x=x[m]; w=w[m]
    idx=np.argsort(x); x=x[idx]; w=w[idx]
    cw=np.cumsum(w)
    return float(x[np.searchsorted(cw, 0.5*cw[-1])])

def _rolling_mean(a: np.ndarray, win: int) -> np.ndarray:
    if win<=1: return a
    return pd.Series(a).rolling(win, center=True, min_periods=max(2, win//3)).mean().to_numpy()

def build_segments(df: pd.DataFrame, seg_len_s: int=10, min_speed_mps: float=0.55,
                   flat_slope_range: tuple[float,float]=(-1.0,1.0), slope_window_s: int=15) -> pd.DataFrame:
    dt = df["time"].diff().dt.total_seconds().to_numpy()
    dt[0]=np.nan
    dist=df["dist_m"].to_numpy(float); alt=df["alt_m"].to_numpy(float); sp=df["speed_mps"].to_numpy(float)
    win=max(3,int(slope_window_s))
    alt_s=_rolling_mean(alt,win)
    d_alt=np.r_[np.nan,np.diff(alt_s)]
    d_dist=np.r_[np.nan,np.diff(dist)]
    slope=np.where((d_dist>1.0)&np.isfinite(d_alt), 100.0*d_alt/d_dist, np.nan)
    slope=pd.Series(slope).rolling(win, center=True, min_periods=max(2, win//3)).mean().to_numpy()
    slope=np.clip(slope,-30.0,30.0)

    work=pd.DataFrame({"time":df["time"].values,"dt_s":dt,"speed_mps":sp,"slope_pct":slope}).dropna().reset_index(drop=True)
    work=work[(work["dt_s"]>0.2)&(work["dt_s"]<10.0)].copy()

    t0=work["time"].iloc[0]
    work["t_rel_s"]=(work["time"]-t0).dt.total_seconds()
    work["seg_idx"]=(work["t_rel_s"]//seg_len_s).astype(int)

    def wavg(x,w):
        x=np.asarray(x,float); w=np.asarray(w,float)
        m=np.isfinite(x)&np.isfinite(w)&(w>0)
        if m.sum()==0: return np.nan
        return float(np.sum(x[m]*w[m])/np.sum(w[m]))

    rows=[]
    for _,g in work.groupby("seg_idx", sort=True):
        dt_sum=float(g["dt_s"].sum())
        if dt_sum<0.6*seg_len_s: continue
        rows.append({"t0":g["time"].iloc[0],"t1":g["time"].iloc[-1],"dt_s":dt_sum,
                     "speed_mps":wavg(g["speed_mps"].values,g["dt_s"].values),
                     "slope_pct":wavg(g["slope_pct"].values,g["dt_s"].values)})
    seg=pd.DataFrame(rows)
    if seg.empty: raise ValueError("No segments after segmentation.")
    seg=seg[seg["speed_mps"]>=float(min_speed_mps)].copy().reset_index(drop=True)

    lo,hi=flat_slope_range
    flat=seg[(seg["slope_pct"]>=lo)&(seg["slope_pct"]<=hi)]
    if len(flat)<8:
        flat=seg.iloc[seg["slope_pct"].abs().sort_values().index[:max(10, min(50, len(seg)//4))]]
    v_flat=float(np.median(flat["speed_mps"].values))
    seg.attrs["flat_v_mps"]=v_flat
    seg["flat_v_mps"]=v_flat
    seg["ratio_vs_flat"]=seg["speed_mps"]/np.maximum(v_flat,0.1)
    return seg

def build_reference_F_relative(segs_all: pd.DataFrame, bin_width_pct: float=1.0, F_min: float=0.65, F_max: float=1.75) -> pd.DataFrame:
    df=segs_all.copy()
    bw=float(bin_width_pct)
    df["slope_bin_pct"]=np.round(df["slope_pct"]/bw)*bw
    rows=[]
    for sb,g in df.groupby("slope_bin_pct"):
        F=_weighted_median(g["ratio_vs_flat"].values,g["dt_s"].values)
        rows.append({"slope_bin_pct":float(sb),"F_ref":float(F)})
    out=pd.DataFrame(rows).sort_values("slope_bin_pct").reset_index(drop=True)
    if (out["slope_bin_pct"]==0.0).any():
        F0=float(out.loc[out["slope_bin_pct"]==0.0,"F_ref"].iloc[0])
    else:
        F0=float(out.loc[out["slope_bin_pct"].abs().idxmin(),"F_ref"])
    if np.isfinite(F0) and F0>0: out["F_ref"]=out["F_ref"]/F0
    out["F_ref"]=out["F_ref"].clip(F_min,F_max)
    s_min=float(out["slope_bin_pct"].min()); s_max=float(out["slope_bin_pct"].max())
    full=pd.DataFrame({"slope_bin_pct":np.arange(np.floor(s_min), np.ceil(s_max)+bw, bw)})
    full=full.merge(out, on="slope_bin_pct", how="left")
    full["F_ref"]=full["F_ref"].interpolate(limit_direction="both").clip(F_min,F_max)
    return full

def build_reference_V_down_abs(segs_all: pd.DataFrame, slope_range: tuple[float,float]=(-15.0,-1.0), bin_width_pct: float=1.0) -> pd.DataFrame:
    lo,hi=slope_range
    df=segs_all[(segs_all["slope_pct"]>=lo)&(segs_all["slope_pct"]<=hi)].copy()
    if df.empty: return pd.DataFrame({"slope_bin_pct":[], "v_down_ref_mps":[]})
    bw=float(bin_width_pct)
    df["slope_bin_pct"]=np.round(df["slope_pct"]/bw)*bw
    rows=[]
    for sb,g in df.groupby("slope_bin_pct"):
        v=_weighted_median(g["speed_mps"].values,g["dt_s"].values)
        rows.append({"slope_bin_pct":float(sb),"v_down_ref_mps":float(v)})
    out=pd.DataFrame(rows).sort_values("slope_bin_pct").reset_index(drop=True)
    s_min=float(out["slope_bin_pct"].min()); s_max=float(out["slope_bin_pct"].max())
    full=pd.DataFrame({"slope_bin_pct":np.arange(np.floor(s_min), np.ceil(s_max)+bw, bw)})
    full=full.merge(out, on="slope_bin_pct", how="left")
    full["v_down_ref_mps"]=full["v_down_ref_mps"].interpolate(limit_direction="both")
    return full

def interpolate_table(x: np.ndarray, table: pd.DataFrame, x_col: str, y_col: str, clip: tuple[float,float] | None=None) -> np.ndarray:
    xq=np.asarray(x,float)
    if table.empty: return np.full_like(xq, np.nan, dtype=float)
    xx=table[x_col].to_numpy(float); yy=table[y_col].to_numpy(float)
    out=np.interp(xq, xx, yy, left=yy[0], right=yy[-1])
    if clip is not None: out=np.clip(out, clip[0], clip[1])
    return out

def estimate_k_glide_from_downhill_abs(test_seg: pd.DataFrame, slope_range: tuple[float,float]=(-15.0,-1.0)) -> float:
    lo,hi=slope_range
    g=test_seg[(test_seg["slope_pct"]>=lo)&(test_seg["slope_pct"]<=hi)].copy()
    if "v_down_ref_mps" in g.columns: g=g[np.isfinite(g["v_down_ref_mps"])]
    if len(g)<10:
        g=test_seg.copy()
        if "v_down_ref_mps" in g.columns: g=g[np.isfinite(g["v_down_ref_mps"])]
    if len(g)==0: return 1.0
    ratio=g["speed_mps"]/np.maximum(g["v_down_ref_mps"],1e-6)
    k=_weighted_median(ratio.to_numpy(), g["dt_s"].to_numpy())
    if (not np.isfinite(k)) or k<=0: k=float(np.nanmedian(ratio.to_numpy()))
    return float(k)

def compute_effort_speed(v_obs_mps: np.ndarray, F_ref: np.ndarray, k_glide: float) -> np.ndarray:
    v=np.asarray(v_obs_mps,float); F=np.asarray(F_ref,float)
    k=float(k_glide) if np.isfinite(k_glide) and k_glide>0 else 1.0
    return v/(k*np.maximum(F,1e-6))
