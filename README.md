# Glide + Slope (2-model) demo

- **Model A (Downhill ABS):** builds `V_down_ref(s)` using absolute speed on negative slopes only.  
  Session glide factor: `k_glide = weighted_median(v_obs / V_down_ref(s))` on downhill range.

- **Model B (Relative):** builds `F_ref(s)` from ratios `v / v_flat` across full slope range, with `v_flat` from [-1%, +1%], and normalizes so F(0)=1.

- Apply: `v_eff = v_obs / (k_glide * F_ref(s))`

- Filters out segments with speed < **2 km/h** by default.

Run:
```
pip install -r requirements.txt
streamlit run streamlit_app.py
```
