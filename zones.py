
from __future__ import annotations
import numpy as np
import pandas as pd

def parse_zone_edges_kmh(txt: str) -> list[float]:
    parts = [p.strip() for p in (txt or "").split(",") if p.strip()]
    edges = []
    for p in parts:
        try:
            edges.append(float(p))
        except Exception:
            pass
    edges = sorted(list(dict.fromkeys(edges)))
    return edges

def add_zone_column(df: pd.DataFrame, value_col: str, edges_kmh: list[float], unit: str = "mps", new_col: str = "zone") -> pd.DataFrame:
    out = df.copy()
    v = out[value_col].to_numpy(float)
    if unit.lower() in ("mps", "m/s"):
        v_kmh = v * 3.6
    else:
        v_kmh = v
    edges = np.asarray(edges_kmh, dtype=float)
    idx = np.digitize(v_kmh, edges, right=True)
    out[new_col] = ["Z{}".format(i+1) for i in idx]
    return out

def zone_distribution(df: pd.DataFrame, zone_col: str, weight_col: str = "dt_s") -> pd.DataFrame:
    g = df.groupby(zone_col, as_index=False)[weight_col].sum().rename(columns={zone_col: "zone", weight_col: "time_s"})
    def znum(z):
        try:
            return int(str(z).replace("Z", ""))
        except Exception:
            return 999
    g["zone_n"] = g["zone"].apply(znum)
    g = g.sort_values("zone_n").drop(columns=["zone_n"]).reset_index(drop=True)
    g["time_min"] = g["time_s"] / 60.0
    return g
