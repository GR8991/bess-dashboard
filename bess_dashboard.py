"""
=============================================================================
BESS Engineering Dashboard — NLR GenAI Power Profiles Dataset
=============================================================================
Run locally:   streamlit run bess_dashboard.py
Deployed at:   Streamlit Community Cloud

Data source modes:
  1. Demo mode     — built-in synthetic data, works instantly
  2. NREL Auto     — fetches directly from the hardcoded NREL URL
  3. Google Drive  — downloads real dataset via Google Drive ID
  4. Local mode    — paste local folder path (PC use only)

Dataset: DOI 10.7799/3025227 — NLR Kestrel HPC
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pathlib
import io
import zipfile
import tempfile
import warnings
import requests 
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE — initialise before everything else
# ─────────────────────────────────────────────────────────────────────────────
for key, val in {
    "source_mode" : "demo",
    "base"        : "",
    "wl_found"    : {},
    "wl_cache"    : {},
    "demo_loaded" : False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="BESS Dashboard — NLR GenAI", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

PALETTE = px.colors.qualitative.Plotly
def get_color(idx): return PALETTE[int(idx) % len(PALETTE)]

# >>> HARDCODED DATA SOURCES <<<
GDRIVE_FILE_ID = "1lD6LWo6eKaWutR4q5Gku-bWUGggffM7pi"
NREL_DATASET_URL = "https://data.nrel.gov/system/files/164/dataset.zip" # Update this if the NREL link changes

WORKLOAD_DEFS = {
    "training": {"label": "🏋️  Training (Llama2 LoRA + Stable Diffusion)", "id_col": "Unnamed: 0", "group_col": "nodes", "group_lbl": "Nodes", "extra_cols": ["model", "repeat"]},
    "inference_offline_llama3_70b": {"label": "📦 Inference — Offline batch (Llama3 70B)", "id_col": None, "group_col": "batch_size", "group_lbl": "Batch size", "extra_cols": ["repeat", "elapsed", "peak_power[W]", "mean_power[W]"]},
    "inference_online_rate_llama3_70b": {"label": "⚡ Inference — Online rate (Llama3 70B)", "id_col": None, "group_col": "request_rate", "group_lbl": "Request rate (req/s)", "extra_cols": ["num-prompts", "burstiness", "peak_power[W]", "mean_power[W]"]},
    "inference_online_finite_llama3_70b": {"label": "🎯 Inference — Online finite (Llama3 70B)", "id_col": None, "group_col": "num_prompts", "group_lbl": "Num prompts", "extra_cols": ["request_rate_y", "duration", "peak_power[W]", "mean_power[W]"]},
}

# ─────────────────────────────────────────────────────────────────────────────
# AUTO-FETCH WEB DOWNLOADER (NO URL INPUT NEEDED)
# ─────────────────────────────────────────────────────────────────────────────
def download_from_url(url: str) -> pathlib.Path:
    tmp = pathlib.Path(tempfile.mkdtemp()) / "nlr_dataset"
    tmp.mkdir(parents=True, exist_ok=True)
    zip_path = tmp / "dataset.zip"

    st.info("⬇️  Fetching dataset from NREL servers...\nPlease do not close this tab.")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        progress_bar = st.progress(0, text="Downloading data...")
        downloaded = 0
        
        with open(zip_path, "wb") as file:
            for data in response.iter_content(chunk_size=1024 * 1024):
                file.write(data)
                downloaded += len(data)
                if total_size > 0:
                    progress_bar.progress(min(downloaded / total_size, 1.0), text=f"Downloading... {downloaded // (1024 * 1024)} MB")
                else:
                    progress_bar.progress(0.5, text=f"Downloading... {downloaded // (1024 * 1024)} MB (Unknown total size)")
        
        progress_bar.empty()
        
    except Exception as e:
        st.error(f"❌  Download error: Could not fetch from URL. Make sure the NREL link is still active. Details: {e}")
        return tmp

    if not zip_path.exists():
        st.error("❌  Download failed — file not found after download.")
        return tmp

    st.success(f"✅  Download complete — {zip_path.stat().st_size // (1024 * 1024)} MB")

    with st.spinner("📦  Extracting zip …"):
        try:
            with zipfile.ZipFile(zip_path) as zf: zf.extractall(tmp)
            st.success("✅  Extraction complete")
        except zipfile.BadZipFile:
            st.error("❌  File is not a valid zip. The download may have been corrupted.")
            return tmp

    candidates = [p for p in tmp.rglob("01_aggregated_datasets") if p.is_dir()]
    return candidates[0].parent if candidates else tmp

# ─────────────────────────────────────────────────────────────────────────────
# DATA GENERATORS & PARSERS
# ─────────────────────────────────────────────────────────────────────────────
def generate_demo_data():
    np.random.seed(42)
    dt = 0.2
    configs = [(2, "llama2_70b_lora", 0), (2, "llama2_70b_lora", 1), (4, "llama2_70b_lora", 0), (8, "llama2_70b_lora", 0)]
    meta_rows, all_runs = [], []
    for idx, (nodes, model, repeat) in enumerate(configs):
        duration_s = np.random.uniform(4000, 6000)
        n_steps = int(duration_s / dt)
        time_s = np.arange(n_steps) * dt
        peak_total, idle_total = np.random.uniform(2800, 3520) * nodes, np.random.uniform(380, 460) * nodes
        ramp_steps, plateau_end = int(30 / dt), n_steps - int(20 / dt)
        power = np.zeros(n_steps)
        for i in range(n_steps):
            if i < ramp_steps: power[i] = idle_total + (peak_total - idle_total) * (i / ramp_steps) + np.random.normal(0, peak_total * 0.01)
            elif i < plateau_end: power[i] = peak_total * np.random.uniform(0.94, 0.99)
            else: power[i] = peak_total * (1 - (i - plateau_end) / max(n_steps - plateau_end, 1)) + idle_total * ((i - plateau_end) / max(n_steps - plateau_end, 1))
        all_runs.append(pd.DataFrame({"time_s": time_s, "power_W": np.clip(power, idle_total * 0.8, peak_total * 1.02), "group": str(nodes), "file_num": idx}))
        meta_rows.append({"Unnamed: 0": idx, "model": model, "nodes": nodes})
    return pd.DataFrame(meta_rows), pd.concat(all_runs, ignore_index=True)

def download_from_gdrive(file_id: str) -> pathlib.Path:
    import gdown
    tmp = pathlib.Path(tempfile.mkdtemp()) / "nlr_dataset"
    tmp.mkdir(parents=True, exist_ok=True)
    zip_path = tmp / "dataset.zip"
    st.info("⬇️  Downloading dataset from Google Drive (~1 GB).")
    try: gdown.download(url=f"https://drive.google.com/uc?id={file_id}", output=str(zip_path), quiet=False)
    except Exception as e: st.error(f"❌  gdown error: {e}"); return tmp
    if not zip_path.exists(): return tmp
    with st.spinner("📦  Extracting zip …"):
        try:
            with zipfile.ZipFile(zip_path) as zf: zf.extractall(tmp)
        except zipfile.BadZipFile: st.error("❌  Invalid zip."); return tmp
    candidates = [p for p in tmp.rglob("01_aggregated_datasets") if p.is_dir()]
    return candidates[0].parent if candidates else tmp

def scan_workloads(base_str: str) -> dict:
    agg_dir = pathlib.Path(base_str) / "01_aggregated_datasets"
    return {f.name: WORKLOAD_DEFS[f.name]["label"] for f in sorted(agg_dir.iterdir()) if f.is_dir() and (f/"results").exists() and f.name in WORKLOAD_DEFS} if agg_dir.exists() else {}

def load_workload(base_str: str, folder: str, max_files: int = 150):
    wl_dir = pathlib.Path(base_str) / "01_aggregated_datasets" / folder
    wl_def = WORKLOAD_DEFS[folder]
    meta = next((pd.read_excel(wl_dir/fname) if fname.endswith(".xlsx") else pd.read_csv(wl_dir/fname) for fname in ["metadata.csv", "metadata", "metadata.xlsx"] if (wl_dir/fname).exists()), None)
    if meta is None: return None, None
    pairs = list(zip(meta[wl_def["id_col"]].astype(int), [r for _, r in meta.iterrows()])) if wl_def["id_col"] and wl_def["id_col"] in meta.columns else list(zip(range(len(meta)), [r for _, r in meta.iterrows()]))
    if len(pairs) > max_files: pairs = pairs[::max(1, len(pairs) // max_files)][:max_files]
    runs, bar = [], st.progress(0, text="Loading files…")
    for i, (fnum, row) in enumerate(pairs):
        fpath = wl_dir / "results" / f"{int(fnum):06d}.parquet"
        if not fpath.exists(): continue
        try: raw = pd.read_parquet(fpath).reset_index().dropna()
        except: continue
        if raw.empty: continue
        raw.columns = ["time_s", "power_W"]
        raw["group"] = str(row[wl_def["group_col"]]) if wl_def["group_col"] in row.index and pd.notna(row.get(wl_def["group_col"])) else "all"
        raw["file_num"] = int(fnum)
        runs.append(raw)
        bar.progress((i + 1) / len(pairs))
    bar.empty()
    return meta, pd.concat(runs, ignore_index=True) if runs else (None, None)

def compute_stats(data: pd.DataFrame, dt: float) -> pd.DataFrame:
    rows = []
    for (grp, fidx), chunk in data.groupby(["group", "file_num"]):
        pw = chunk.sort_values("time_s")["power_W"]
        ramp = pw.diff().abs() / max(dt, 1e-9)
        rows.append({"Group": grp, "File #": int(fidx), "Peak (kW)": round(pw.max()/1000, 2), "Mean (kW)": round(pw.mean()/1000, 2), "Idle (kW)": round(pw.quantile(0.05)/1000, 2), "Std (kW)": round(pw.std()/1000, 3), "LF (%)": round(pw.mean()/pw.max()*100, 1) if pw.max()>0 else 0, "Duration (s)": round(chunk["time_s"].max(), 1), "Energy (kWh)": round(pw.sum()*dt/3600/1000, 3), "Ramp 95pc (kW/s)": round(ramp.quantile(0.95)/1000, 4), "Ramp max (kW/s)": round(ramp.max()/1000, 4)})
    return pd.DataFrame(rows)

def gen_solar(n_steps, dt, pv_kW):
    h = np.arange(n_steps) * dt / 3600
    return np.where((h >= 6) & (h <= 20), pv_kW * np.maximum(0.0, np.sin(np.pi * (h - 6) / 14)), 0.0)

def sim_soc(net_kW, pwr_kW, e_kWh, rte, dt, lo_pct, hi_pct):
    soc, lo, hi = np.zeros(len(net_kW)), lo_pct / 100 * e_kWh, hi_pct / 100 * e_kWh
    soc[0] = hi
    for i in range(1, len(net_kW)):
        d = min(abs(net_kW[i]), pwr_kW) * dt / 3600
        soc[i] = max(soc[i-1] - d / rte, lo) if net_kW[i] > 0 else min(soc[i-1] + d * rte, hi)
    return soc / e_kWh * 100

def show_single_batch(run_df, file_num, grp_val, grp_lbl, dt):
    pw, time = run_df["power_W"], run_df["time_s"]
    ramp = pw.diff().abs() / max(dt, 1e-9)
    peak_kW, mean_kW, idle_kW = pw.max()/1000, pw.mean()/1000, pw.quantile(0.05)/1000
    dur_s, energy_kWh, ramp_p95 = time.max(), pw.sum()*dt/3600/1000, ramp.quantile(0.95)/1000
    micro_cycles_hr = len(np.where(np.diff(np.sign(pw.diff().dropna())))[0]) / (dur_s / 3600) if dur_s > 0 else 0

    st.markdown(f"#### Batch detail — {grp_lbl} = **{grp_val}** | File # {file_num}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Peak", f"{peak_kW:.2f} kW"); c2.metric("Mean", f"{mean_kW:.2f} kW"); c3.metric("Idle", f"{idle_kW:.2f} kW"); c4.metric("Std dev", f"{pw.std()/1000:.3f} kW"); c5.metric("Load factor", f"{(mean_kW/peak_kW*100) if peak_kW>0 else 0:.1f} %")
    c6, c7, c8, c9, c10 = st.columns(5)
    c6.metric("Duration", f"{dur_s:.0f} s"); c7.metric("Energy", f"{energy_kWh:.3f} kWh"); c8.metric("Ramp 95th", f"{ramp_p95:.4f} kW/s"); c9.metric("Micro-cycles/hr", f"{micro_cycles_hr:.0f}"); c10.write("")
    st.divider()
    
    p1a, p1b = st.columns([2, 1])
    with p1a:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time/60, y=pw/1000, mode="lines", name="Power (kW)", line=dict(color="#378ADD", width=1.2)))
        fig.add_hline(y=peak_kW, line_dash="dash", line_color="#BA7517"); fig.add_hline(y=mean_kW, line_dash="dot", line_color="#1D9E75")
        fig.update_layout(title="Power time-series", xaxis_title="Time (min)", yaxis_title="Power (kW)", height=340, margin=dict(l=55, r=15, t=50, b=45))
        st.plotly_chart(fig, use_container_width=True)
    with p1b:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=time/60, y=ramp/1000, mode="lines", name="Ramp (kW/s)", line=dict(color="#f38ba8", width=0.8), fill="tozeroy"))
        fig2.update_layout(title="Ramp rate over time", xaxis_title="Time (min)", yaxis_title="kW/s", height=340, margin=dict(l=55, r=15, t=50, b=45))
        st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ BESS Dashboard")
    st.caption("NLR GenAI — DOI 10.7799/3025227")
    st.divider()

    st.markdown("**Step 1 — Data source**")
    source_mode = st.radio(
        "Choose data source",
        options=["demo", "nrel", "gdrive", "local"],
        format_func=lambda x: {"demo":"🎯 Demo mode (synthetic)", "nrel":"🌐 NREL Database (Auto-fetch)", "gdrive":"☁️  Google Drive", "local":"💾 Local folder"}[x],
        index=["demo", "nrel", "gdrive", "local"].index(st.session_state["source_mode"])
    )
    st.session_state["source_mode"] = source_mode

    if source_mode == "demo":
        st.info("Synthetic profiles. Works instantly.")
        if st.button("▶️  Load demo data", type="primary", use_container_width=True):
            meta, data = generate_demo_data()
            st.session_state["wl_cache"]["demo"] = {"meta": meta, "data": data, "folder": "training", "is_demo": True}
            st.session_state["demo_loaded"] = True

    # -> NEW AUTO FETCH LOGIC HERE <-
    elif source_mode == "nrel":
        st.markdown("**Download directly from NREL**")
        st.info("This will automatically fetch the Kestrel HPC GenAI dataset directly from the NREL servers (~1 GB).")
        if st.button("🌐 Download NREL Data", type="primary", use_container_width=True):
            try:
                base = download_from_url(NREL_DATASET_URL)
                found = scan_workloads(str(base))
                if found:
                    st.session_state["base"], st.session_state["wl_found"], st.session_state["wl_cache"] = str(base), found, {}
                    st.success(f"✅  {len(found)} workload(s) ready")
                else: st.error("❌  Dataset extracted but no workloads found.")
            except Exception as e: st.error(f"❌  Error: {e}")

    elif source_mode == "gdrive":
        if st.button("☁️  Download from Google Drive", type="primary", use_container_width=True):
            base = download_from_gdrive(GDRIVE_FILE_ID)
            found = scan_workloads(str(base))
            if found:
                st.session_state["base"], st.session_state["wl_found"], st.session_state["wl_cache"] = str(base), found, {}
                st.success(f"✅  {len(found)} workload(s) ready")

    else:
        local_path = st.text_input("Folder path", value="")
        if st.button("🔍  Scan folder", type="primary", use_container_width=True):
            found = scan_workloads(local_path.strip())
            if found:
                st.session_state["base"], st.session_state["wl_found"], st.session_state["wl_cache"] = local_path.strip(), found, {}
                st.success(f"✅  {len(found)} workload(s) found")

    wl_found = st.session_state.get("wl_found", {})
    if not (source_mode == "demo" and st.session_state.get("demo_loaded")) and wl_found:
        st.divider()
        st.markdown("**Step 2 — Select workload**")
        folder_name = st.radio("Available workloads", options=list(wl_found.keys()), format_func=lambda k: wl_found[k])
        max_files = st.slider("Max runs to load", 20, 500, 150, 10)
        if st.button("📂  Load workload", type="primary", use_container_width=True):
            meta, data = load_workload(st.session_state["base"], folder_name, max_files)
            if data is not None:
                st.session_state["wl_cache"][folder_name] = {"meta": meta, "data": data, "folder": folder_name, "is_demo": False}
                st.success("✅  Loaded")

    st.divider()
    st.markdown("**Step 3 — Facility & BESS**")
    facility_nodes = st.slider("Total GPU nodes", 10, 500, 156, 10)
    pue            = st.slider("PUE", 1.05, 2.00, 1.20, 0.05)
    solar_frac     = st.slider("Solar PV (% of peak)", 20, 100, 60, 5)
    bess_margin    = st.slider("BESS safety margin (%)", 0, 100, 25, 5) / 100.0
    bess_dur       = st.slider("BESS duration (hours)", 1, 8, 4, 1)
    rte            = st.slider("Round-trip efficiency", 0.80, 0.98, 0.92, 0.01)
    soc_min        = st.slider("Min SoC (%)", 5, 30, 20, 5)
    soc_max        = st.slider("Max SoC (%)", 70, 99, 90, 5)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN DASHBOARD LOGIC
# ─────────────────────────────────────────────────────────────────────────────
wl_cache = st.session_state["wl_cache"]
active_key = "demo" if (source_mode == "demo" and "demo" in wl_cache) else next((k for k in reversed(wl_cache.keys()) if k != "demo"), None)

if not active_key:
    st.title("⚡ BESS Engineering Dashboard")
    st.markdown("### Welcome — NLR GenAI Power Profiles BESS Analysis\nAnalyse AI workload power profiles to compute BESS engineering parameters. **Choose a data source in the sidebar ←**")
    st.stop()

cached = wl_cache[active_key]
meta, data, is_demo, folder = cached["meta"], cached["data"], cached.get("is_demo", False), cached.get("folder", "training")
dt = max(float(data.sort_values(["file_num","time_s"]).groupby("file_num")["time_s"].diff().dropna().median() if not data.empty else 0.2), 0.01)
stats_df = compute_stats(data, dt)
peak_W, idle_W, mean_W = float(data["power_W"].max()), float(data["power_W"].quantile(0.05)), float(data["power_W"].mean())
lf = mean_W / peak_W if peak_W > 0 else 0

scale = facility_nodes / 1
fac_peak_kW, fac_mean_kW, fac_idle_kW = peak_W/1000*scale*pue, mean_W/1000*scale*pue, idle_W/1000*scale*pue
bess_pwr_kW = max(0, (fac_peak_kW - fac_mean_kW) * (1 + bess_margin))
bess_e_kWh = bess_pwr_kW * bess_dur
bess_pwr_MW, bess_e_MWh = bess_pwr_kW / 1000, bess_e_kWh / 1000
c_rate = bess_pwr_kW / max(bess_e_kWh, 1e-9)

n24 = int(24 * 3600 / dt)
t24 = np.arange(n24) * dt / 3600
load_24 = np.full(n24, fac_idle_kW)
job_hrs = [2, 5, 8, 12, 16, 20]
top_idx = int(stats_df.sort_values("Peak (kW)", ascending=False).iloc[0]["File #"])
rl = data[data["file_num"] == top_idx].sort_values("time_s")["power_W"].values / 1000 * scale * pue
for sh in job_hrs:
    si = int(sh * 3600 / dt)
    ei = min(si + len(rl), n24)
    load_24[si:ei] = rl[:ei - si]

solar_kW = gen_solar(n24, dt, fac_peak_kW * solar_frac / 100)
net_kW = load_24 - solar_kW
soc_arr = sim_soc(net_kW, bess_pwr_kW, bess_e_kWh, rte, dt, soc_min, soc_max)

st.title("⚡ BESS Engineering Dashboard")
st.caption(f"{'🎯 Demo' if is_demo else '☁️ Real NLR data'} | dt={dt:.2f}s | {facility_nodes} nodes | Margin {bess_margin*100:.0f}%")
k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("Peak load", f"{fac_peak_kW:.0f} kW"); k2.metric("Mean load", f"{fac_mean_kW:.0f} kW"); k3.metric("Load factor", f"{lf*100:.1f} %")
k4.metric("BESS power", f"{bess_pwr_MW:.2f} MW"); k5.metric("BESS energy", f"{bess_e_MWh:.2f} MWh"); k6.metric("C-rate", f"{c_rate:.2f} C")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔬 Single batch", "🔋 BESS sizing", "⚖️ Comparison"])

with tab1:
    fig_all = go.Figure()
    for g in data["group"].unique():
        rs = data[data["group"] == g].sort_values("time_s")
        fig_all.add_trace(go.Scatter(x=rs["time_s"]/60, y=rs["power_W"]/1000, mode="lines", name=g))
    fig_all.update_layout(title="Power profiles", xaxis_title="Time (min)", yaxis_title="Power (kW)", height=420)
    st.plotly_chart(fig_all, use_container_width=True)
    st.dataframe(stats_df.sort_values(["Group","File #"]), use_container_width=True)

with tab2:
    sel_file = st.selectbox("Select batch to analyze", data["file_num"].unique())
    sel_run = data[data["file_num"] == sel_file].sort_values("time_s")
    if not sel_run.empty: show_single_batch(sel_run, sel_file, sel_run["group"].iloc[0], WORKLOAD_DEFS[folder]["group_lbl"], dt)

with tab3:
    fig_24 = go.Figure()
    fig_24.add_trace(go.Scatter(x=t24, y=load_24/1000, line=dict(color="#E24B4A", width=1.5), name="Facility load (MW)"))
    fig_24.add_trace(go.Scatter(x=t24, y=solar_kW/1000, fill="tozeroy", line=dict(color="#1D9E75", width=1.5), name=f"Solar PV"))
    fig_24.add_trace(go.Scatter(x=t24, y=np.maximum(net_kW,0)/1000, line=dict(color="#534AB7", width=1.5, dash="dot"), name="Net grid demand"))
    fig_24.update_layout(title="Facility 24h Load Profile", xaxis_title="Hour", yaxis_title="Power (MW)", height=380)
    st.plotly_chart(fig_24, use_container_width=True)
    
    fig_soc = go.Figure()
    fig_soc.add_trace(go.Scatter(x=t24, y=soc_arr, fill="tozeroy", line=dict(color="#BA7517", width=2), name="BESS SoC (%)"))
    fig_soc.update_layout(title=f"BESS SoC — {bess_e_MWh:.2f} MWh | {bess_pwr_MW:.2f} MW", yaxis=dict(range=[0,105]), height=360)
    st.plotly_chart(fig_soc, use_container_width=True)

with tab4:
    all_fnums = sorted(data["file_num"].unique().tolist())
    sel_files = st.multiselect("Select batches to compare", all_fnums, default=all_fnums[:min(2, len(all_fnums))])
    if len(sel_files) >= 2:
        fig_cmp = go.Figure()
        for i, fn in enumerate(sel_files):
            run = data[data["file_num"] == fn].sort_values("time_s")
            fig_cmp.add_trace(go.Scatter(x=run["time_s"]/60, y=run["power_W"]/1000, mode="lines", name=f"File {fn}"))
        fig_cmp.update_layout(height=380, title="Batch comparison")
        st.plotly_chart(fig_cmp, use_container_width=True)
