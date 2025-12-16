
from __future__ import annotations
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R*c

def load_tcx_to_df(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.read()
    text = raw.decode("utf-8", errors="ignore")
    root = ET.fromstring(text)

    def tag_name(t):
        return t.split("}")[-1] if "}" in t else t

    tps = []
    for tp in root.iter():
        if tag_name(tp.tag) != "Trackpoint":
            continue
        rec = {"time": None, "lat": None, "lon": None, "alt_m": None, "dist_m": None, "hr": None, "speed_mps": None}
        for child in tp:
            tn = tag_name(child.tag)
            if tn == "Time":
                rec["time"] = child.text
            elif tn == "AltitudeMeters":
                rec["alt_m"] = child.text
            elif tn == "DistanceMeters":
                rec["dist_m"] = child.text
            elif tn == "Position":
                for pc in child:
                    ptn = tag_name(pc.tag)
                    if ptn == "LatitudeDegrees":
                        rec["lat"] = pc.text
                    elif ptn == "LongitudeDegrees":
                        rec["lon"] = pc.text
            elif tn == "HeartRateBpm":
                for hrc in child:
                    if tag_name(hrc.tag) == "Value":
                        rec["hr"] = hrc.text
            elif tn == "Extensions":
                for ext in child.iter():
                    if tag_name(ext.tag).lower() == "speed" and ext.text:
                        rec["speed_mps"] = ext.text

        if rec["time"] is None:
            continue
        tps.append(rec)

    df = pd.DataFrame(tps)
    if df.empty:
        raise ValueError("No trackpoints found in TCX.")

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    for c in ["lat","lon","alt_m","dist_m","hr","speed_mps"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Distance
    if df["dist_m"].isna().all():
        if df["lat"].isna().all() or df["lon"].isna().all():
            raise ValueError("TCX lacks both DistanceMeters and GPS positions.")
        lat = df["lat"].to_numpy(float)
        lon = df["lon"].to_numpy(float)
        d = np.zeros(len(df), dtype=float)
        d[1:] = _haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
        df["dist_m"] = np.cumsum(np.nan_to_num(d, nan=0.0))
    else:
        df["dist_m"] = df["dist_m"].interpolate(limit_direction="both").ffill().bfill()

    # Altitude
    if df["alt_m"].isna().all():
        df["alt_m"] = 0.0
    else:
        df["alt_m"] = df["alt_m"].interpolate(limit_direction="both").ffill().bfill()

    # Speed
    if df["speed_mps"].isna().all():
        dt = df["time"].diff().dt.total_seconds().to_numpy()
        dd = df["dist_m"].diff().to_numpy()
        sp = np.zeros(len(df), dtype=float)
        sp[1:] = np.where((dt[1:] > 0) & np.isfinite(dd[1:]), dd[1:] / dt[1:], np.nan)
        df["speed_mps"] = pd.Series(sp).interpolate(limit_direction="both").ffill().bfill()
    else:
        df["speed_mps"] = df["speed_mps"].interpolate(limit_direction="both").ffill().bfill()

    return df
