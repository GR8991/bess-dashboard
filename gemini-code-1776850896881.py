"""
=============================================================================
BESS Engineering Dashboard — NLR GenAI Power Profiles Dataset
=============================================================================
Run locally:   streamlit run bess_dashboard.py
Deployed at:   Streamlit Community Cloud

Three data source modes:
  1. Demo mode     — built-in synthetic data, works instantly
  2. Google Drive  — downloads real NLR dataset zip via gdown
  3. Local mode    — paste local folder path (PC use only)

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
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BESS Dashboard — NLR GenAI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = px.colors.qualitative.Plotly
def get_color(idx): return PALETTE[int(idx) % len(PALETTE)]

# ─────────────────────────────────────────────────────────────────────────────
# GOOGLE DRIVE FILE ID
# ─────────────────────────────────────────────────────────────────────────────
GDRIVE_FILE_ID = "1lD6LWo6eKaWutR4q5Gku-bWUGggffM7pi"

# ─────────────────────────────────────────────────────────────────────────────
# WORKLOAD DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────
WORKLOAD_DEFS = {
    "training": {
        "label"     : "🏋️  Training (Llama2 LoRA + Stable Diffusion)",
        "id_col"    : "Unnamed: 0",
        "group_col" : "nodes",
        "group_lbl" : "Nodes",
        "extra_cols": ["model", "repeat"],
    },
    "inference_offline_llama3_70b": {
        "label"     : "📦 Inference — Offline batch (Llama3 70B)",
        "id_col"    : None,
        "group_col" : "batch_size",
        "group_lbl" : "Batch size",
        "extra_cols": ["repeat", "elapsed", "peak_power[W]", "mean_power[W]"],
    },
    "inference_online_rate_llama3_70b": {
        "label"     : "⚡ Inference — Online rate (Llama3 70B)",
        "id_col"    : None,
        "group_col" : "request_rate",
        "group_lbl" : "Request rate (req/s)",
        "extra_cols": ["num-prompts", "burstiness",
                       "peak_power[W]", "mean_power[W]"],
    },
    "inference_online_finite_llama3_70b": {
        "label"     : "🎯 Inference — Online finite (Llama3 70B)",
        "id_col"    : None,
        "group_col" : "num_prompts",
        "group_lbl" : "Num prompts",
        "extra_cols": ["request_rate_y", "duration",
                       "peak_power[W]", "mean_power[W]"],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# DEMO DATA GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def generate_demo_data():
    np.random.seed(42)
    dt      = 0.2
    configs = [
        (2,  "llama2_70b_lora",  0),
        (2,  "llama2_70b_lora",  1),
        (4,  "llama2_70b_lora",  0),
        (4,  "llama2_70b_lora",  1),
        (8,  "llama2_70b_lora",  0),
        (8,  "stable_diffusion", 0),
        (16, "llama2_70b_lora",  0),
        (16, "stable_diffusion", 0),
    ]
    meta_rows, all_runs = [], []

    for idx, (nodes, model, repeat) in enumerate(configs):
        duration_s    = np.random.uniform(4000, 6000)
        n_steps       = int(duration_s / dt)
        time_s        = np.arange(n_steps) * dt
        peak_per_node = np.random.uniform(2800, 3520)
        idle_per_node = np.random.uniform(380,  460)
        peak_total    = peak_per_node * nodes
        idle_total    = idle_per_node * nodes
        ramp_steps    = int(30 / dt)
        plateau_end   = n_steps - int(20 / dt)

        power = np.zeros(n_steps)
        for i in range(n_steps):
            if i < ramp_steps:
                frac     = i / ramp_steps
                power[i] = idle_total + (peak_total - idle_total) * frac
                power[i] += np.random.normal(0, peak_total * 0.01)
            elif i < plateau_end:
                power[i] = peak_total * np.random.uniform(0.94, 0.99)
            else:
                frac     = (i - plateau_end) / max(n_steps - plateau_end, 1)
                power[i] = peak_total * (1 - frac) + idle_total * frac
                power[i] += np.random.normal(0, peak_total * 0.005)

        power = np.clip(power, idle_total * 0.8, peak_total * 1.02)

        df = pd.DataFrame({
            "time_s"   : time_s,
            "power_W"  : power,
            "group"    : str(nodes),
            "file_num" : idx,
            "run_idx"  : idx,
            "nodes"    : nodes,
            "model"    : model,
            "repeat"   : repeat,
        })
        all_runs.append(df)
        meta_rows.append({
            "Unnamed: 0": idx,
            "model"      : model,
            "nodes"      : nodes,
            "repeat"     : repeat,
            "path_save"  : f"training/results/{idx:06d}.parquet",
            "slurmid"    : 10000000 + idx,
        })

    return pd.DataFrame(meta_rows), pd.concat(all_runs, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# GOOGLE DRIVE DOWNLOADER
# ─────────────────────────────────────────────────────────────────────────────
def download_from_gdrive(file_id: str) -> pathlib.Path:
    import gdown
    tmp = pathlib.Path(tempfile.mkdtemp()) / "nlr_dataset"
    tmp.mkdir(parents=True, exist_ok=True)
    zip_path = tmp / "dataset.zip"
    url = f"https://drive.google.com/uc?id={file_id}"

    st.info(
        "⬇️  Downloading dataset from Google Drive (~1 GB).\n\n"
        "This may take 5–15 minutes depending on your connection. "
        "Please do not close this tab."
    )

    try:
        gdown.download(url=url, output=str(zip_path), quiet=False)
    except Exception as e:
        st.error(f"❌  gdown download error: {e}")
        return tmp

    if not zip_path.exists():
        st.error("❌  Download failed — file not found after download.")
        return tmp

    size_mb = zip_path.stat().st_size // (1024 * 1024)
    if size_mb < 10:
        st.error(f"❌  Downloaded file is too small ({size_mb} MB).")
        return tmp

    st.success(f"✅  Download complete — {size_mb} MB")

    with st.spinner("📦  Extracting zip … (1–2 minutes for 1 GB)"):
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp)
            st.success("✅  Extraction complete")
        except zipfile.BadZipFile:
            st.error("❌  File is not a valid zip.")
            return tmp

    candidates = [p for p in tmp.rglob("01_aggregated_datasets") if p.is_dir()]
    return candidates[0].parent if candidates else tmp


# ─────────────────────────────────────────────────────────────────────────────
# SCAN & LOAD WORKLOADS
# ─────────────────────────────────────────────────────────────────────────────
def scan_workloads(base_str: str) -> dict:
    base    = pathlib.Path(base_str)
    agg_dir = base / "01_aggregated_datasets"
    found   = {}
    if not agg_dir.exists(): return found
    for folder in sorted(agg_dir.iterdir()):
        if not folder.is_dir(): continue
        results = folder / "results"
        parquets = list(results.glob("*.parquet")) if results.exists() else []
        if parquets and folder.name in WORKLOAD_DEFS:
            found[folder.name] = WORKLOAD_DEFS[folder.name]["label"]
    return found

def load_workload(base_str: str, folder: str, max_files: int = 150):
    base    = pathlib.Path(base_str)
    wl_dir  = base / "01_aggregated_datasets" / folder
    results = wl_dir / "results"
    wl_def  = WORKLOAD_DEFS[folder]
    id_col  = wl_def["id_col"]
    grp_col = wl_def["group_col"]

    meta = None
    for fname in ["metadata.csv", "metadata", "metadata.xlsx"]:
        mp = wl_dir / fname
        if not mp.exists(): continue
        try:
            meta = (pd.read_excel(mp) if mp.suffix == ".xlsx" else pd.read_csv(mp))
            break
        except Exception: pass
    if meta is None:
        st.error(f"❌  Cannot read metadata in {wl_dir}")
        return None, None

    if id_col and id_col in meta.columns:
        pairs = list(zip(meta[id_col].astype(int).tolist(), [row for _, row in meta.iterrows()]))
    else:
        pairs = list(zip(range(len(meta)), [row for _, row in meta.iterrows()]))

    n_total = len(pairs)
    if n_total > max_files:
        step  = max(1, n_total // max_files)
        pairs = pairs[::step][:max_files]

    runs, missing, errors = [], 0, 0
    bar = st.progress(0, text="Loading parquet files …")

    for i, (file_num, row) in enumerate(pairs):
        fpath = results / f"{int(file_num):06d}.parquet"
        if not fpath.exists():
            missing += 1; continue
        try: raw = pd.read_parquet(fpath)
        except Exception:
            errors += 1; continue

        raw = raw.reset_index()
        raw.columns = ["time_s", "power_W"]
        raw["time_s"]  = pd.to_numeric(raw["time_s"],  errors="coerce")
        raw["power_W"] = pd.to_numeric(raw["power_W"], errors="coerce")
        raw = raw.dropna(subset=["time_s", "power_W"])
        if raw.empty: continue

        raw["group"]    = (str(row[grp_col]) if grp_col in row.index and pd.notna(row.get(grp_col)) else "all")
        raw["file_num"] = int(file_num)
        raw["run_idx"]  = i

        for col in wl_def.get("extra_cols", []):
            if col in row.index and pd.notna(row.get(col)):
                raw[col] = row[col]

        runs.append(raw)
        bar.progress((i + 1) / len(pairs), text=f"Loading … {i+1}/{len(pairs)}")

    bar.empty()
    if not runs: return None, None
    return meta, pd.concat(runs, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_stats(data: pd.DataFrame, dt: float) -> pd.DataFrame:
    rows = []
    for (grp, fidx), chunk in data.groupby(["group", "file_num"]):
        chunk = chunk.sort_values("time_s")
        pw    = chunk["power_W"]
        ramp  = pw.diff().abs() / max(dt, 1e-9)
        rows.append({
            "Group"            : grp,
            "File #"           : int(fidx),
            "Peak (kW)"        : round(pw.max()  / 1000, 2),
            "Mean (kW)"        : round(pw.mean() / 1000, 2),
            "Idle (kW)"        : round(pw.quantile(0.05) / 1000, 2),
            "Std (kW)"         : round(pw.std()  / 1000, 3),
            "LF (%)"           : round(pw.mean() / pw.max() * 100, 1),
            "Duration (s)"     : round(chunk["time_s"].max(), 1),
            "Energy (kWh)"     : round(pw.sum() * dt / 3600 / 1000, 3),
            "Ramp 95pc (kW/s)" : round(ramp.quantile(0.95) / 1000, 4),
            "Ramp max (kW/s)"  : round(ramp.max()           / 1000, 4),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# SOLAR + SoC
# ─────────────────────────────────────────────────────────────────────────────
def gen_solar(n_steps, dt, pv_kW):
    h = np.arange(n_steps) * dt / 3600
    return np.where((h >= 6) & (h <= 20), pv_kW * np.maximum(0.0, np.sin(np.pi * (h - 6) / 14)), 0.0)

def sim_soc(net_kW, pwr_kW, e_kWh, rte, dt, lo_pct, hi_pct):
    soc = np.zeros(len(net_kW))
    lo, hi = lo_pct / 100 * e_kWh, hi_pct / 100 * e_kWh
    soc[0] = hi
    for i in range(1, len(net_kW)):
        d = min(abs(net_kW[i]), pwr_kW) * dt / 3600
        soc[i] = (max(soc[i-1] - d / rte, lo) if net_kW[i] > 0 else min(soc[i-1] + d * rte, hi))
    return soc / e_kWh * 100


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE BATCH DRILL-DOWN
# ─────────────────────────────────────────────────────────────────────────────
def show_single_batch(run_df, file_num, grp_val, grp_lbl, dt, meta_row=None):
    pw   = run_df["power_W"]
    time = run_df["time_s"]
    diff_pw = pw.diff()
    ramp = diff_pw.abs() / max(dt, 1e-9)

    peak_kW    = pw.max()  / 1000
    idle_kW    = pw.quantile(0.05) / 1000
    mean_kW    = pw.mean() / 1000
    std_kW     = pw.std()  / 1000
    lf         = (pw.mean() / pw.max() * 100) if pw.max() > 0 else 0
    dur_s      = time.max()
    energy_kWh = pw.sum() * dt / 3600 / 1000
    ramp_p95   = ramp.quantile(0.95) / 1000
    
    # Micro-cycling Calculation: Zero crossings in the derivative array
    zero_crossings = np.where(np.diff(np.sign(diff_pw.dropna())))[0]
    num_micro_cycles = len(zero_crossings)
    micro_cycles_hr = num_micro_cycles / (dur_s / 3600) if dur_s > 0 else 0

    st.markdown(f"#### Batch detail — {grp_lbl} = **{grp_val}** | File # {file_num}")
    
    # Split metrics into two rows for cleaner UI scaling
    row1_c1, row1_c2, row1_c3, row1_c4, row1_c5 = st.columns(5)
    row1_c1.metric("Peak", f"{peak_kW:.2f} kW")
    row1_c2.metric("Mean", f"{mean_kW:.2f} kW")
    row1_c3.metric("Idle", f"{idle_kW:.2f} kW")
    row1_c4.metric("Std dev", f"{std_kW:.3f} kW")
    row1_c5.metric("Load factor", f"{lf:.1f} %")

    row2_c1, row2_c2, row2_c3, row2_c4, row2_c5 = st.columns(5)
    row2_c1.metric("Duration", f"{dur_s:.0f} s")
    row2_c2.metric("Energy", f"{energy_kWh:.3f} kWh")
    row2_c3.metric("Ramp 95th", f"{ramp_p95:.4f} kW/s")
    row2_c4.metric("Micro-cycles/hr", f"{micro_cycles_hr:.0f}")
    
    st.divider()

    # Row 1: time-series + ramp over time
    p1a, p1b = st.columns([2, 1])
    with p1a:
        win = max(10, int(30 / dt))
        roll_mean = pw.rolling(win, center=True).mean() / 1000
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time/60, y=pw/1000, mode="lines", name="Power (kW)", line=dict(color="#378ADD", width=1.2)))
        fig.add_trace(go.Scatter(x=time/60, y=roll_mean, mode="lines", name=f"Rolling mean ({win*dt:.0f}s)", line=dict(color="#E24B4A", width=2, dash="dot")))
        fig.add_hline(y=peak_kW, line_dash="dash", line_color="#BA7517", annotation_text=f"Peak {peak_kW:.2f} kW", annotation_position="top right")
        fig.add_hline(y=mean_kW, line_dash="dot", line_color="#1D9E75", annotation_text=f"Mean {mean_kW:.2f} kW", annotation_position="bottom right")
        fig.update_layout(title="Power time-series with rolling mean", xaxis_title="Time (min)", yaxis_title="Power (kW)", height=340, hovermode="x unified", margin=dict(l=55, r=15, t=50, b=45))
        st.plotly_chart(fig, use_container_width=True)

    with p1b:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=time/60, y=ramp/1000, mode="lines", name="Ramp (kW/s)", line=dict(color="#f38ba8", width=0.8), fill="tozeroy", fillcolor="rgba(243,139,168,0.10)"))
        fig2.add_hline(y=ramp_p95, line_dash="dash", line_color="red", annotation_text=f"95th {ramp_p95:.4f} kW/s")
        fig2.update_layout(title="Ramp rate over time", xaxis_title="Time (min)", yaxis_title="kW/s", height=340, hovermode="x unified", margin=dict(l=55, r=15, t=50, b=45))
        st.plotly_chart(fig2, use_container_width=True)

    # Row 2: histogram + cumulative energy + phase pie
    p2a, p2b, p2c = st.columns(3)
    with p2a:
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(x=pw/1000, nbinsx=60, marker_color="#378ADD", opacity=0.75))
        fig3.add_vline(x=mean_kW, line_dash="dash", line_color="#1D9E75", annotation_text=f"Mean {mean_kW:.2f} kW")
        fig3.add_vline(x=peak_kW, line_dash="dash", line_color="#E24B4A", annotation_text=f"Peak {peak_kW:.2f} kW")
        fig3.update_layout(title="Power distribution", xaxis_title="Power (kW)", yaxis_title="Count", height=300, margin=dict(l=55, r=15, t=50, b=45))
        st.plotly_chart(fig3, use_container_width=True)

    with p2b:
        cumE = np.cumsum(pw.values) * dt / 3600 / 1000
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=time/60, y=cumE, mode="lines", fill="tozeroy", fillcolor="rgba(29,158,117,0.15)", line=dict(color="#1D9E75", width=1.5)))
        fig4.update_layout(title="Cumulative energy consumed", xaxis_title="Time (min)", yaxis_title="kWh", height=300, hovermode="x unified", margin=dict(l=55, r=15, t=50, b=45))
        st.plotly_chart(fig4, use_container_width=True)

    with p2c:
        idle_thresh = idle_kW * 1000 * 1.3
        plateau_thresh = peak_kW * 1000 * 0.80
        from collections import Counter
        phases = ["Idle" if p <= idle_thresh else "Plateau" if p >= plateau_thresh else "Transition" for p in pw.values]
        counts = Counter(phases)
        labels = ["Idle", "Transition", "Plateau"]
        values = [counts.get(l, 0) * dt for l in labels]
        fig5 = go.Figure(go.Pie(labels=labels, values=values, marker=dict(colors=["#888780","#BA7517","#E24B4A"]), hole=0.4, textinfo="label+percent"))
        fig5.update_layout(title="Time in each power phase", height=300, showlegend=False, margin=dict(l=15, r=15, t=50, b=15))
        st.plotly_chart(fig5, use_container_width=True)

    # Row 3: load duration + ramp up/down split
    p3a, p3b = st.columns(2)
    with p3a:
        sorted_pw = np.sort(pw.values / 1000)[::-1]
        pct = np.arange(1, len(sorted_pw)+1) / len(sorted_pw) * 100
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(x=pct, y=sorted_pw, mode="lines", line=dict(color="#534AB7", width=2), fill="tozeroy", fillcolor="rgba(83,74,183,0.10)"))
        fig6.add_hline(y=mean_kW, line_dash="dot", line_color="#1D9E75", annotation_text=f"Mean {mean_kW:.2f} kW")
        fig6.update_layout(title="Load duration curve", xaxis_title="Duration exceeded (%)", yaxis_title="Power (kW)", height=300, margin=dict(l=55, r=15, t=50, b=45))
        st.plotly_chart(fig6, use_container_width=True)

    with p3b:
        ramp_up   = (ramp / 1000)[diff_pw > 0].dropna()
        ramp_down = (ramp / 1000)[diff_pw < 0].dropna()
        ramp_up   = ramp_up[ramp_up <= ramp_up.quantile(0.999)]
        ramp_down = ramp_down[ramp_down <= ramp_down.quantile(0.999)]
        fig7 = make_subplots(rows=1, cols=2, subplot_titles=["Ramp-up", "Ramp-down"])
        fig7.add_trace(go.Histogram(x=ramp_up, nbinsx=40, marker_color="#E24B4A", opacity=0.75), row=1, col=1)
        fig7.add_trace(go.Histogram(x=ramp_down, nbinsx=40, marker_color="#378ADD", opacity=0.75), row=1, col=2)
        fig7.update_layout(title_text="Ramp-up vs ramp-down", height=300, showlegend=False, margin=dict(l=40, r=15, t=60, b=45))
        st.plotly_chart(fig7, use_container_width=True)

    if meta_row is not None:
        with st.expander("📋  Full metadata for this batch", expanded=False):
            st.dataframe(pd.DataFrame([meta_row]).T.rename(columns={0: "Value"}), use_container_width=True)


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
        options=["demo", "gdrive", "local"],
        format_func=lambda x: {"demo":"🎯 Demo mode (synthetic — instant)", "gdrive":"☁️  Google Drive (real dataset)", "local":"💾 Local folder (your PC only)"}[x],
        index=["demo","gdrive","local"].index(st.session_state["source_mode"])
    )
    st.session_state["source_mode"] = source_mode

    if source_mode == "demo":
        st.info("Synthetic profiles. Works instantly.")
        if st.button("▶️  Load demo data", type="primary", use_container_width=True):
            with st.spinner("Generating synthetic profiles …"):
                meta, data = generate_demo_data()
            st.session_state["wl_cache"]["demo"] = {"meta": meta, "data": data, "folder": "training", "is_demo": True}
            st.session_state["demo_loaded"] = True
            st.success("✅  Demo data ready")

    elif source_mode == "gdrive":
        if st.button("☁️  Download from Google Drive", type="primary", use_container_width=True):
            try:
                base = download_from_gdrive(GDRIVE_FILE_ID)
                found = scan_workloads(str(base))
                if found:
                    st.session_state["base"] = str(base)
                    st.session_state["wl_found"] = found
                    st.session_state["wl_cache"] = {}
                    st.success(f"✅  {len(found)} workload(s) ready")
                else: st.error("❌  No workloads found.")
            except Exception as e: st.error(f"❌  Error: {e}")

    else:
        local_path = st.text_input("Folder path", value=r"C:\Users\Downloads\dataset")
        if st.button("🔍  Scan folder", type="primary", use_container_width=True):
            found = scan_workloads(local_path.strip())
            if found:
                st.session_state["base"] = local_path.strip()
                st.session_state["wl_found"] = found
                st.session_state["wl_cache"] = {}
                st.success(f"✅  {len(found)} workload(s) found")
            else: st.error("❌  No workloads found.")

    wl_found = st.session_state.get("wl_found", {})
    is_demo  = (source_mode == "demo" and st.session_state.get("demo_loaded"))

    if not is_demo and wl_found:
        st.divider()
        st.markdown("**Step 2 — Select workload**")
        folder_name = st.radio("Available workloads", options=list(wl_found.keys()), format_func=lambda k: wl_found[k])
        max_files = st.slider("Max runs to load", 20, 500, 150, 10)
        if st.button("📂  Load workload", type="primary", use_container_width=True):
            with st.spinner("Loading parquet files …"):
                meta, data = load_workload(st.session_state["base"], folder_name, max_files)
            if data is not None:
                st.session_state["wl_cache"][folder_name] = {"meta": meta, "data": data, "folder": folder_name, "is_demo": False}
                st.success("✅  Loaded")

    st.divider()
    st.markdown("**Step 3 — Facility & BESS**")
    facility_nodes = st.slider("Total GPU nodes", 10, 500, 156, 10)
    pue            = st.slider("PUE", 1.05, 2.00, 1.20, 0.05)
    solar_frac     = st.slider("Solar PV (% of peak)", 20, 100, 60, 5)
    bess_margin    = st.slider("BESS safety margin (%)", 0, 100, 25, 5) / 100.0  # NEW: Margin for volatile loads
    bess_dur       = st.slider("BESS duration (hours)", 1, 8, 4, 1)
    rte            = st.slider("Round-trip efficiency", 0.80, 0.98, 0.92, 0.01)
    soc_min        = st.slider("Min SoC (%)", 5, 30, 20, 5)
    soc_max        = st.slider("Max SoC (%)", 70, 99, 90, 5)


# ─────────────────────────────────────────────────────────────────────────────
# ACTIVE DATA & DERIVED QUANTITIES
# ─────────────────────────────────────────────────────────────────────────────
wl_cache = st.session_state["wl_cache"]
active_key = "demo" if (source_mode == "demo" and "demo" in wl_cache) else next((k for k in reversed(wl_cache.keys()) if k != "demo"), None)

if not active_key:
    st.title("⚡ BESS Engineering Dashboard")
    st.markdown("### Welcome — NLR GenAI Power Profiles BESS Analysis\nAnalyse AI workload power profiles to compute BESS engineering parameters. **Choose a data source in the sidebar ←**")
    st.stop()

cached = wl_cache[active_key]
meta, data, is_demo, folder = cached["meta"], cached["data"], cached.get("is_demo", False), cached.get("folder", "training")
wl_def = WORKLOAD_DEFS.get(folder, WORKLOAD_DEFS["training"])
grp_lbl = wl_def["group_lbl"]

dt_vals = data.sort_values(["file_num","time_s"]).groupby("file_num")["time_s"].diff().dropna()
dt = max(float(dt_vals.median()) if len(dt_vals) > 0 else 0.2, 0.01)
groups = sorted(data["group"].unique().tolist())
grp_colors = {g: get_color(i) for i, g in enumerate(groups)}
stats_df = compute_stats(data, dt)

peak_W = float(data["power_W"].max())
idle_W = float(data["power_W"].quantile(0.05))
mean_W = float(data["power_W"].mean())
lf = mean_W / peak_W if peak_W > 0 else 0
ramp_p95 = float(data.sort_values(["file_num","time_s"]).groupby("file_num")["power_W"].diff().abs().div(dt).quantile(0.95))

rep_nodes, top_idx = 1, int(stats_df.sort_values("Peak (kW)", ascending=False).iloc[0]["File #"])
rep_run = data[data["file_num"] == top_idx].sort_values("time_s").reset_index(drop=True)
if "nodes" in rep_run.columns:
    try: rep_nodes = int(rep_run["nodes"].iloc[0])
    except: pass

scale = facility_nodes / max(rep_nodes, 1)
fac_peak_kW = peak_W / 1000 * scale * pue
fac_idle_kW = idle_W / 1000 * scale * pue
fac_mean_kW = mean_W / 1000 * scale * pue

# NEW MATH: Safe buffer calculated from Peak-Mean delta + user-defined margin
bess_pwr_kW = max(0, (fac_peak_kW - fac_mean_kW) * (1 + bess_margin))
bess_e_kWh  = bess_pwr_kW * bess_dur
c_rate      = bess_pwr_kW / max(bess_e_kWh, 1e-9)
bess_pwr_MW = bess_pwr_kW / 1000
bess_e_MWh  = bess_e_kWh / 1000
chem        = "LFP — high power / NMC Hybrid" if c_rate > 1.0 else "LFP — standard"

n24 = int(24 * 3600 / dt)
t24 = np.arange(n24) * dt / 3600
load_24 = np.full(n24, fac_idle_kW)
job_hrs = [2, 5, 8, 12, 16, 20]
rl = rep_run["power_W"].values / 1000 * scale * pue
for sh in job_hrs:
    si = int(sh * 3600 / dt)
    ei = min(si + len(rl), n24)
    load_24[si:ei] = rl[:ei - si]

solar_kW = gen_solar(n24, dt, fac_peak_kW * solar_frac / 100)
net_kW = load_24 - solar_kW
soc_arr = sim_soc(net_kW, bess_pwr_kW, bess_e_kWh, rte, dt, soc_min, soc_max)
deficit_kWh = float(np.sum(net_kW.clip(min=0)) * dt / 3600)
surplus_kWh = float(np.sum((-net_kW).clip(min=0)) * dt / 3600)
total_kWh = float(np.sum(load_24) * dt / 3600)
re_pct = max(0.0, (1 - deficit_kWh / max(total_kWh, 1e-9)) * 100)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER + KPIs
# ─────────────────────────────────────────────────────────────────────────────
st.title("⚡ BESS Engineering Dashboard")
st.caption(f"{'🎯 Demo' if is_demo else '☁️ Real NLR data'} | dt={dt:.2f}s | {facility_nodes} nodes | PUE {pue} | Margin {bess_margin*100:.0f}%")

st.markdown("### 📊 Key metrics")
k1,k2,k3,k4,k5,k6,k7,k8 = st.columns(8)
k1.metric("Peak load", f"{fac_peak_kW:.0f} kW")
k2.metric("Mean load", f"{fac_mean_kW:.0f} kW")
k3.metric("Idle load", f"{fac_idle_kW:.0f} kW")
k4.metric("Load factor", f"{lf*100:.1f} %")
k5.metric("BESS power", f"{bess_pwr_MW:.2f} MW")
k6.metric("BESS energy", f"{bess_e_MWh:.2f} MWh")
k7.metric("C-rate", f"{c_rate:.2f} C")
k8.metric("RE coverage", f"{re_pct:.1f} %")
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔬 Single batch", "🔋 BESS sizing", "⚖️ Comparison"])

with tab1:
    fig_all = go.Figure()
    for g in groups:
        shown = False
        for fidx, run in data[data["group"] == g].groupby("file_num"):
            rs = run.sort_values("time_s")
            fig_all.add_trace(go.Scatter(x=rs["time_s"]/60, y=rs["power_W"]/1000, mode="lines", line=dict(color=grp_colors[g], width=0.9), opacity=0.40, name=f"{grp_lbl}={g}", legendgroup=str(g), showlegend=not shown))
            shown = True
    fig_all.update_layout(title="Power profiles", xaxis_title="Time (min)", yaxis_title="Power (kW)", height=420)
    st.plotly_chart(fig_all, use_container_width=True)
    st.dataframe(stats_df.sort_values(["Group","File #"]), use_container_width=True)

with tab2:
    sel_c1, sel_c2, sel_c3 = st.columns([1, 1, 2])
    with sel_c1: sel_group = st.selectbox(f"Filter by {grp_lbl}", ["All groups"] + [str(g) for g in groups])
    avail = data[["file_num","group"]].drop_duplicates() if sel_group == "All groups" else data[data["group"] == sel_group][["file_num","group"]].drop_duplicates()
    with sel_c2: sel_file = st.selectbox("Select batch", sorted(avail["file_num"].unique().tolist()), format_func=lambda x: f"File {x:06d}")
    
    sel_run = data[data["file_num"] == sel_file].sort_values("time_s").reset_index(drop=True)
    if not sel_run.empty:
        show_single_batch(sel_run, sel_file, sel_run["group"].iloc[0], grp_lbl, dt)

with tab3:
    fig_24 = go.Figure()
    fig_24.add_trace(go.Scatter(x=t24, y=load_24/1000, line=dict(color="#E24B4A", width=1.5), name="Facility load (MW)"))
    fig_24.add_trace(go.Scatter(x=t24, y=solar_kW/1000, fill="tozeroy", line=dict(color="#1D9E75", width=1.5), name=f"Solar PV"))
    fig_24.add_trace(go.Scatter(x=t24, y=np.maximum(net_kW,0)/1000, line=dict(color="#534AB7", width=1.5, dash="dot"), name="Net grid demand"))
    fig_24.update_layout(title="Facility 24h Load Profile", xaxis_title="Hour", yaxis_title="Power (MW)", height=380)
    st.plotly_chart(fig_24, use_container_width=True)

    c_soc, c_spec = st.columns([2, 1])
    with c_soc:
        fig_soc = go.Figure()
        fig_soc.add_trace(go.Scatter(x=t24, y=soc_arr, fill="tozeroy", line=dict(color="#BA7517", width=2), name="BESS SoC (%)"))
        fig_soc.update_layout(title=f"BESS SoC — {bess_e_MWh:.2f} MWh | {bess_pwr_MW:.2f} MW", yaxis=dict(range=[0,105]), height=360)
        st.plotly_chart(fig_soc, use_container_width=True)
    with c_spec:
        st.markdown("**BESS specification**")
        st.write(f"**Power rating:** {bess_pwr_MW:.2f} MW")
        st.write(f"**Energy capacity:** {bess_e_MWh:.2f} MWh")
        st.write(f"**C-rate:** {c_rate:.2f} C")
        st.write(f"**Chemistry:** {chem}")
        st.write(f"**Safety Margin:** {bess_margin*100:.0f} %")
        st.write(f"**Ramp 95th:** {ramp_p95/1000:.3f} kW/s")

with tab4:
    all_fnums = sorted(data["file_num"].unique().tolist())
    sel_files = st.multiselect("Select batches", all_fnums, default=all_fnums[:min(2, len(all_fnums))])
    if len(sel_files) >= 2:
        fig_cmp = go.Figure()
        for i, fn in enumerate(sel_files):
            run = data[data["file_num"] == fn].sort_values("time_s")
            fig_cmp.add_trace(go.Scatter(x=run["time_s"]/60, y=run["power_W"]/1000, mode="lines", name=f"File {fn}"))
        fig_cmp.update_layout(height=380)
        st.plotly_chart(fig_cmp, use_container_width=True)