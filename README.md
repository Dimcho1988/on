
# onFlows • Slope + Glide Normalizer (Streamlit Demo)

This Streamlit app demonstrates the ratio-based slope + glide normalization:

- Build reference slope factor **F_ref(s)** from multiple ski TCX files using **10 s segments**.
- For a test activity, estimate session glide factor **k_glide**.
- Compute **effort speed** (slope + glide removed): `v_eff = v_obs / (k_glide * F_ref(s))`.
- Filter out stop/shoot phases by dropping segments with speed `< 0.5 m/s` (configurable).
- Show segment table + zone distributions for raw speed and effort speed.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
