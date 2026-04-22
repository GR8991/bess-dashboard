"""
=============================================================================
BESS Engineering Dashboard — NLR GenAI Power Profiles Dataset
=============================================================================
Run locally:   streamlit run bess_dashboard.py
Deployed at:   Streamlit Community Cloud

Three data source modes:
  1. Demo mode   — built-in synthetic data, works instantly
  2. NLR URL     — downloads real dataset from official NLR server
  3. Local mode  — paste local folder path (PC use only)

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
import requests
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
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
# OFFICIAL NLR DATASET URL — public, no authentication needed
# ─────────────────────────────────────────────────────────────────────────────
NLR_URL = "https://data.nlr.gov/system/files/312/1774982010-dataset.zip"

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
        "extra_cols": ["repeat", "elapsed",
                       "peak_power[W]", "mean_power[W]"],
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
        peak_total    = np.random.uniform(2800, 3520) * nodes
        idle_total    = np.random.uniform(380,  460)  * nodes
        ramp_steps    = int(30 / dt)
        plateau_end   = n_steps - int(20 / dt)
        power = np.zeros(n_steps)
        for i in range(n_steps):
            if i < ramp_steps:
                frac     = i / ramp_steps
                power[i] = idle_total + (peak_total-idle_total)*frac
                power[i] += np.random.normal(0, peak_total*0.01)
            elif i < plateau_end:
                power[i] = peak_total * np.random.uniform(0.94, 0.99)
            else:
                frac     = (i-plateau_end)/max(n_steps-plateau_end,1)
                power[i] = peak_total*(1-frac)+idle_total*frac
                power[i] += np.random.normal(0, peak_total*0.005)
        power = np.clip(power, idle_total*0.8, peak_total*1.02)
        df = pd.DataFrame({
            "time_s"  : time_s, "power_W": power,
            "group"   : str(nodes), "file_num": idx,
            "run_idx" : idx, "nodes": nodes,
            "model"   : model, "repeat": repeat,
        })
        all_runs.append(df)
        meta_rows.append({
            "Unnamed: 0": idx, "model": model,
            "nodes": nodes, "repeat": repeat,
            "path_save": f"training/results/{idx:06d}.parquet",
            "slurmid": 10000000+idx,
        })
    return pd.DataFrame(meta_rows), pd.concat(all_runs, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# NLR URL DOWNLOADER — direct download, no authentication needed
# ─────────────────────────────────────────────────────────────────────────────
def download_from_url(url: str) -> pathlib.Path:
    """
    Download zip from any public URL using requests.
    Works with the official NLR data portal — no authentication.
    """
    tmp = pathlib.Path(tempfile.mkdtemp()) / "nlr_dataset"
    tmp.mkdir(parents=True, exist_ok=True)
    zip_path = tmp / "dataset.zip"

    prog = st.progress(0, text="⬇️  Connecting to NLR data server …")

    try:
        resp = requests.get(
            url,
            stream=True,
            timeout=300,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        done  = 0

        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = min(done / total, 1.0)
                    mb  = done  // (1024 * 1024)
                    tot = total // (1024 * 1024)
                    prog.progress(
                        pct,
                        text=f"⬇️  Downloading … {mb} MB / {tot} MB"
                    )
                else:
                    mb = done // (1024 * 1024)
                    prog.progress(
                        0,
                        text=f"⬇️  Downloading … {mb} MB"
                    )

        prog.empty()

    except requests.exceptions.RequestException as e:
        prog.empty()
        st.error(f"❌  Download failed: {e}")
        return tmp

    # Validate
    if not zip_path.exists():
        st.error("❌  Download failed — file not saved.")
        return tmp

    size_mb = zip_path.stat().st_size // (1024 * 1024)
    if size_mb < 10:
        st.error(
            f"❌  Downloaded file is too small ({size_mb} MB).\n\n"
            "The server may have returned an error page instead of "
            "the actual file. Please try again."
        )
        return tmp

    st.success(f"✅  Download complete — {size_mb} MB")

    # Extract
    with st.spinner("📦  Extracting zip … (1–2 minutes for 1 GB)"):
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp)
            st.success("✅  Extraction complete")
        except zipfile.BadZipFile:
            st.error("❌  File is not a valid zip. Please try again.")
            return tmp

    candidates = [p for p in tmp.rglob("01_aggregated_datasets")
                  if p.is_dir()]
    return candidates[0].parent if candidates else tmp


# ─────────────────────────────────────────────────────────────────────────────
# SCAN WORKLOADS
# ─────────────────────────────────────────────────────────────────────────────
def scan_workloads(base_str: str) -> dict:
    base    = pathlib.Path(base_str)
    agg_dir = base / "01_aggregated_datasets"
    found   = {}
    if not agg_dir.exists():
        return found
    for folder in sorted(agg_dir.iterdir()):
        if not folder.is_dir():
            continue
        results  = folder / "results"
        parquets = list(results.glob("*.parquet")) if results.exists() else []
        if parquets and folder.name in WORKLOAD_DEFS:
            found[folder.name] = WORKLOAD_DEFS[folder.name]["label"]
    return found


# ─────────────────────────────────────────────────────────────────────────────
# LOAD ONE WORKLOAD
# ─────────────────────────────────────────────────────────────────────────────
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
        if not mp.exists():
            continue
        try:
            meta = (pd.read_excel(mp) if mp.suffix == ".xlsx"
                    else pd.read_csv(mp))
            break
        except Exception as e:
            st.warning(f"metadata error ({fname}): {e}")
    if meta is None:
        st.error(f"❌  Cannot read metadata in {wl_dir}")
        return None, None

    if id_col and id_col in meta.columns:
        pairs = list(zip(meta[id_col].astype(int).tolist(),
                         [row for _, row in meta.iterrows()]))
    else:
        pairs = list(zip(range(len(meta)),
                         [row for _, row in meta.iterrows()]))

    n_total = len(pairs)
    if n_total > max_files:
        step  = max(1, n_total // max_files)
        pairs = pairs[::step][:max_files]
        st.info(
            f"ℹ️  {n_total} runs available — loading {len(pairs)} "
            f"(every {step}th). Raise 'Max runs' slider for more."
        )

    runs    = []
    missing = 0
    errors  = 0
    bar     = st.progress(0, text="Loading parquet files …")

    for i, (file_num, row) in enumerate(pairs):
        fpath = results / f"{int(file_num):06d}.parquet"
        if not fpath.exists():
            missing += 1
            continue
        try:
            raw = pd.read_parquet(fpath)
        except Exception:
            errors += 1
            continue

        raw          = raw.reset_index()
        raw.columns  = ["time_s", "power_W"]
        raw["time_s"]  = pd.to_numeric(raw["time_s"],  errors="coerce")
        raw["power_W"] = pd.to_numeric(raw["power_W"], errors="coerce")
        raw = raw.dropna(subset=["time_s", "power_W"])
        if raw.empty:
            continue

        raw["group"]    = (str(row[grp_col])
                           if grp_col in row.index
                           and pd.notna(row.get(grp_col))
                           else "all")
        raw["file_num"] = int(file_num)
        raw["run_idx"]  = i

        for col in wl_def.get("extra_cols", []):
            if col in row.index and pd.notna(row.get(col)):
                raw[col] = row[col]

        runs.append(raw)
        bar.progress((i + 1) / len(pairs),
                     text=f"Loading … {i+1}/{len(pairs)}")

    bar.empty()
    if missing: st.warning(f"⚠️  {missing} file(s) not found")
    if errors:  st.warning(f"⚠️  {errors} file(s) unreadable")
    if not runs:
        st.error("❌  No data loaded.")
        return None, None

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
    return np.where(
        (h >= 6) & (h <= 20),
        pv_kW * np.maximum(0.0, np.sin(np.pi*(h-6)/14)), 0.0)

def sim_soc(net_kW, pwr_kW, e_kWh, rte, dt, lo_pct, hi_pct):
    soc    = np.zeros(len(net_kW))
    lo, hi = lo_pct/100*e_kWh, hi_pct/100*e_kWh
    soc[0] = hi
    for i in range(1, len(net_kW)):
        d = min(abs(net_kW[i]), pwr_kW) * dt / 3600
        soc[i] = (max(soc[i-1]-d/rte, lo) if net_kW[i] > 0
                  else min(soc[i-1]+d*rte, hi))
    return soc / e_kWh * 100


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE BATCH DRILL-DOWN
# ─────────────────────────────────────────────────────────────────────────────
def show_single_batch(run_df, file_num, grp_val, grp_lbl, dt, meta_row=None):
    pw   = run_df["power_W"]
    time = run_df["time_s"]
    ramp = pw.diff().abs() / max(dt, 1e-9)

    peak_kW    = pw.max()  / 1000
    idle_kW    = pw.quantile(0.05) / 1000
    mean_kW    = pw.mean() / 1000
    std_kW     = pw.std()  / 1000
    lf         = pw.mean() / pw.max() * 100
    dur_s      = time.max()
    energy_kWh = pw.sum() * dt / 3600 / 1000
    ramp_p95   = ramp.quantile(0.95) / 1000

    st.markdown(
        f"#### Batch detail — {grp_lbl} = **{grp_val}** | File # {file_num}"
    )
    k1,k2,k3,k4,k5,k6,k7,k8 = st.columns(8)
    for col, lbl, v in [
        (k1, "Peak",        f"{peak_kW:.2f} kW"),
        (k2, "Mean",        f"{mean_kW:.2f} kW"),
        (k3, "Idle",        f"{idle_kW:.2f} kW"),
        (k4, "Std dev",     f"{std_kW:.3f} kW"),
        (k5, "Load factor", f"{lf:.1f} %"),
        (k6, "Duration",    f"{dur_s:.0f} s"),
        (k7, "Energy",      f"{energy_kWh:.3f} kWh"),
        (k8, "Ramp 95th",   f"{ramp_p95:.4f} kW/s"),
    ]:
        col.metric(lbl, v)

    st.divider()

    p1a, p1b = st.columns([2, 1])
    with p1a:
        win       = max(10, int(30/dt))
        roll_mean = pw.rolling(win, center=True).mean() / 1000
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time/60, y=pw/1000, mode="lines",
            name="Power (kW)",
            line=dict(color="#378ADD", width=1.2),
        ))
        fig.add_trace(go.Scatter(
            x=time/60, y=roll_mean, mode="lines",
            name=f"Rolling mean ({win*dt:.0f}s)",
            line=dict(color="#E24B4A", width=2, dash="dot"),
        ))
        fig.add_hline(y=peak_kW, line_dash="dash",
                      line_color="#BA7517",
                      annotation_text=f"Peak {peak_kW:.2f} kW",
                      annotation_position="top right")
        fig.add_hline(y=mean_kW, line_dash="dot",
                      line_color="#1D9E75",
                      annotation_text=f"Mean {mean_kW:.2f} kW",
                      annotation_position="bottom right")
        fig.update_layout(
            title="Power time-series with rolling mean",
            xaxis_title="Time (min)", yaxis_title="Power (kW)",
            height=340, hovermode="x unified",
            margin=dict(l=55,r=15,t=50,b=45),
        )
        st.plotly_chart(fig, use_container_width=True)

    with p1b:
        ramp_kws = ramp / 1000
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=time/60, y=ramp_kws, mode="lines",
            name="Ramp (kW/s)",
            line=dict(color="#f38ba8", width=0.8),
            fill="tozeroy",
            fillcolor="rgba(243,139,168,0.10)",
        ))
        fig2.add_hline(y=ramp_p95, line_dash="dash",
                       line_color="red",
                       annotation_text=f"95th {ramp_p95:.4f} kW/s")
        fig2.update_layout(
            title="Ramp rate over time",
            xaxis_title="Time (min)", yaxis_title="kW/s",
            height=340, hovermode="x unified",
            margin=dict(l=55,r=15,t=50,b=45),
        )
        st.plotly_chart(fig2, use_container_width=True)

    p2a, p2b, p2c = st.columns(3)
    with p2a:
        fig3 = go.Figure()
        fig3.add_trace(go.Histogram(
            x=pw/1000, nbinsx=60,
            marker_color="#378ADD", opacity=0.75,
        ))
        fig3.add_vline(x=mean_kW, line_dash="dash",
                       line_color="#1D9E75",
                       annotation_text=f"Mean {mean_kW:.2f} kW")
        fig3.add_vline(x=peak_kW, line_dash="dash",
                       line_color="#E24B4A",
                       annotation_text=f"Peak {peak_kW:.2f} kW")
        fig3.update_layout(
            title="Power distribution",
            xaxis_title="Power (kW)", yaxis_title="Count",
            height=300, margin=dict(l=55,r=15,t=50,b=45),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with p2b:
        cumE = np.cumsum(pw.values) * dt / 3600 / 1000
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=time/60, y=cumE, mode="lines",
            fill="tozeroy",
            fillcolor="rgba(29,158,117,0.15)",
            line=dict(color="#1D9E75", width=1.5),
        ))
        fig4.update_layout(
            title="Cumulative energy consumed",
            xaxis_title="Time (min)", yaxis_title="kWh",
            height=300, hovermode="x unified",
            margin=dict(l=55,r=15,t=50,b=45),
        )
        st.plotly_chart(fig4, use_container_width=True)

    with p2c:
        idle_thresh    = idle_kW * 1000 * 1.3
        plateau_thresh = peak_kW * 1000 * 0.80
        from collections import Counter
        phases = ["Idle" if p <= idle_thresh
                  else "Plateau" if p >= plateau_thresh
                  else "Transition"
                  for p in pw.values]
        counts = Counter(phases)
        labels = ["Idle", "Transition", "Plateau"]
        values = [counts.get(l,0)*dt for l in labels]
        fig5 = go.Figure(go.Pie(
            labels=labels, values=values,
            marker=dict(colors=["#888780","#BA7517","#E24B4A"]),
            hole=0.4, textinfo="label+percent",
            hovertemplate="%{label}: %{value:.0f} s<extra></extra>",
        ))
        fig5.update_layout(
            title="Time in each power phase",
            height=300, showlegend=False,
            margin=dict(l=15,r=15,t=50,b=15),
        )
        st.plotly_chart(fig5, use_container_width=True)

    p3a, p3b = st.columns(2)
    with p3a:
        sorted_pw = np.sort(pw.values/1000)[::-1]
        pct       = np.arange(1, len(sorted_pw)+1)/len(sorted_pw)*100
        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(
            x=pct, y=sorted_pw, mode="lines",
            line=dict(color="#534AB7", width=2),
            fill="tozeroy",
            fillcolor="rgba(83,74,183,0.10)",
        ))
        fig6.add_hline(y=mean_kW, line_dash="dot",
                       line_color="#1D9E75",
                       annotation_text=f"Mean {mean_kW:.2f} kW")
        fig6.update_layout(
            title="Load duration curve (this batch)",
            xaxis_title="Duration exceeded (%)",
            yaxis_title="Power (kW)", height=300,
            margin=dict(l=55,r=15,t=50,b=45),
        )
        st.plotly_chart(fig6, use_container_width=True)

    with p3b:
        ramp_kws_s = ramp / 1000
        diff_pw    = pw.diff()
        ramp_up    = ramp_kws_s[diff_pw > 0].dropna()
        ramp_down  = ramp_kws_s[diff_pw < 0].dropna()
        ramp_up    = ramp_up[ramp_up   <= ramp_up.quantile(0.999)]
        ramp_down  = ramp_down[ramp_down <= ramp_down.quantile(0.999)]
        fig7 = make_subplots(rows=1, cols=2,
                             subplot_titles=["Ramp-up","Ramp-down"])
        fig7.add_trace(go.Histogram(
            x=ramp_up, nbinsx=40,
            marker_color="#E24B4A", opacity=0.75),
            row=1, col=1)
        fig7.add_trace(go.Histogram(
            x=ramp_down, nbinsx=40,
            marker_color="#378ADD", opacity=0.75),
            row=1, col=2)
        fig7.update_layout(
            title_text="Ramp-up vs ramp-down",
            height=300, showlegend=False,
            margin=dict(l=40,r=15,t=60,b=45),
        )
        fig7.update_xaxes(title_text="kW/s")
        fig7.update_yaxes(title_text="Count", col=1)
        st.plotly_chart(fig7, use_container_width=True)

    if meta_row is not None:
        with st.expander("📋  Full metadata for this batch",
                         expanded=False):
            st.dataframe(
                pd.DataFrame([meta_row]).T.rename(columns={0:"Value"}),
                use_container_width=True,
            )


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
        options=["demo", "nlr", "local"],
        format_func=lambda x: {
            "demo" : "🎯 Demo mode (synthetic — instant)",
            "nlr"  : "🌐 NLR dataset (official download ~1 GB)",
            "local": "💾 Local folder (your PC only)",
        }[x],
        index=["demo","nlr","local"].index(
            st.session_state["source_mode"]),
    )
    st.session_state["source_mode"] = source_mode

    # ── Demo ──────────────────────────────────────────────────────────────────
    if source_mode == "demo":
        st.info(
            "Synthetic H100 power profiles that mirror real "
            "NLR measurements. Works instantly."
        )
        if st.button("▶️  Load demo data", type="primary",
                     use_container_width=True):
            with st.spinner("Generating …"):
                meta, data = generate_demo_data()
            st.session_state["wl_cache"]["demo"] = {
                "meta": meta, "data": data,
                "folder": "training", "is_demo": True,
            }
            st.session_state["demo_loaded"] = True
            st.success("✅  Demo data ready")

    # ── NLR official URL ──────────────────────────────────────────────────────
    elif source_mode == "nlr":
        st.markdown("**Official NLR dataset**")
        st.info(
            "Downloads directly from the NLR data portal.\n\n"
            "**No authentication needed** — publicly available.\n\n"
            "⚠️  File size ~1 GB. Download takes 5–20 minutes "
            "depending on connection speed."
        )
        custom_url = st.text_input(
            "Download URL (pre-filled with official NLR URL)",
            value=NLR_URL,
        )
        if st.button("⬇️  Download NLR dataset",
                     type="primary", use_container_width=True):
            try:
                base  = download_from_url(custom_url.strip())
                found = scan_workloads(str(base))
                if found:
                    st.session_state["base"]     = str(base)
                    st.session_state["wl_found"] = found
                    st.session_state["wl_cache"] = {}
                    st.success(
                        f"✅  {len(found)} workload(s) found. "
                        "Select one below."
                    )
                else:
                    st.error(
                        "❌  No workloads found after extraction.\n\n"
                        "The zip may not contain "
                        "`01_aggregated_datasets/` folder."
                    )
            except Exception as e:
                st.error(f"❌  Error: {e}")

    # ── Local ─────────────────────────────────────────────────────────────────
    else:
        st.markdown("**Local folder path**")
        local_path = st.text_input(
            "Folder path",
            value=r"C:\Users\gangaraju.pilly\Downloads\dataset",
        )
        if st.button("🔍  Scan folder", type="primary",
                     use_container_width=True):
            found = scan_workloads(local_path.strip())
            if found:
                st.session_state["base"]     = local_path.strip()
                st.session_state["wl_found"] = found
                st.session_state["wl_cache"] = {}
                st.success(f"✅  {len(found)} workload(s) found")
            else:
                st.error("❌  No workloads found — check path")

    # ── Workload selector ─────────────────────────────────────────────────────
    wl_found = st.session_state.get("wl_found", {})
    is_demo  = (source_mode == "demo"
                and st.session_state.get("demo_loaded"))

    if not is_demo and wl_found:
        st.divider()
        st.markdown("**Step 2 — Select workload**")
        folder_name = st.radio(
            "Available workloads",
            options=list(wl_found.keys()),
            format_func=lambda k: wl_found[k],
        )
        max_files = st.slider("Max runs to load", 20, 500, 150, 10)
        if st.button("📂  Load workload", type="primary",
                     use_container_width=True):
            with st.spinner("Loading parquet files …"):
                meta, data = load_workload(
                    st.session_state["base"],
                    folder_name, max_files)
            if data is not None:
                st.session_state["wl_cache"][folder_name] = {
                    "meta": meta, "data": data,
                    "folder": folder_name, "is_demo": False,
                }
                st.success("✅  Loaded")
    elif not is_demo and not wl_found:
        if source_mode != "demo":
            st.info("Download / scan data first using the button above.")

    # ── BESS parameters ───────────────────────────────────────────────────────
    st.divider()
    st.markdown("**Step 3 — Facility & BESS**")
    facility_nodes = st.slider("Total GPU nodes",        10,  500, 156, 10)
    pue            = st.slider("PUE",                  1.05, 2.00, 1.20, 0.05)
    solar_frac     = st.slider("Solar PV (% of peak)",   20,  100,  60,   5)
    bess_dur       = st.slider("BESS duration (hours)",    1,    8,   4,   1)
    rte            = st.slider("Round-trip efficiency", 0.80, 0.98, 0.92, 0.01)
    soc_min        = st.slider("Min SoC (%)",   5,  30, 20, 5)
    soc_max        = st.slider("Max SoC (%)", 70,  99, 90, 5)


# ─────────────────────────────────────────────────────────────────────────────
# ACTIVE DATA
# ─────────────────────────────────────────────────────────────────────────────
wl_cache   = st.session_state["wl_cache"]
active_key = None

if source_mode == "demo" and "demo" in wl_cache:
    active_key = "demo"
elif wl_found:
    for k in reversed(list(wl_cache.keys())):
        if k != "demo":
            active_key = k
            break

if not active_key:
    st.title("⚡ BESS Engineering Dashboard")
    st.markdown("""
### Welcome — NLR GenAI Power Profiles BESS Analysis

Analyse AI workload power profiles from the **NLR Kestrel HPC**
dataset (DOI: 10.7799/3025227) and compute BESS engineering
parameters for grid integration with renewables.

**Get started — choose a data source in the sidebar ←**

| Mode | Description | Speed |
|---|---|---|
| 🎯 Demo | Synthetic H100 profiles | Instant |
| 🌐 NLR dataset | Official public download (~1 GB) | 5–20 min |
| 💾 Local | Your extracted folder (PC only) | Instant |
    """)
    st.stop()

cached  = wl_cache[active_key]
meta    = cached["meta"]
data    = cached["data"]
is_demo = cached.get("is_demo", False)
folder  = cached.get("folder", "training")
wl_def  = WORKLOAD_DEFS.get(folder, WORKLOAD_DEFS["training"])
grp_lbl = wl_def["group_lbl"]

# ─────────────────────────────────────────────────────────────────────────────
# DERIVED QUANTITIES
# ─────────────────────────────────────────────────────────────────────────────
dt_vals = (data.sort_values(["file_num","time_s"])
               .groupby("file_num")["time_s"].diff().dropna())
dt = float(dt_vals.median()) if len(dt_vals) > 0 else 0.2
dt = max(dt, 0.01)

groups     = sorted(data["group"].unique().tolist())
grp_colors = {g: get_color(i) for i, g in enumerate(groups)}
stats_df   = compute_stats(data, dt)

peak_W   = float(data["power_W"].max())
idle_W   = float(data["power_W"].quantile(0.05))
mean_W   = float(data["power_W"].mean())
lf       = mean_W / peak_W if peak_W > 0 else 0
ramp_p95 = float(
    data.sort_values(["file_num","time_s"])
        .groupby("file_num")["power_W"]
        .diff().abs().div(dt).quantile(0.95)
)

rep_nodes = 1
top_idx   = int(stats_df.sort_values("Peak (kW)", ascending=False)
                         .iloc[0]["File #"])
rep_run   = (data[data["file_num"] == top_idx]
             .sort_values("time_s").reset_index(drop=True))
if "nodes" in rep_run.columns:
    try: rep_nodes = int(rep_run["nodes"].iloc[0])
    except: pass

scale        = facility_nodes / max(rep_nodes, 1)
fac_peak_kW  = peak_W / 1000 * scale * pue
fac_idle_kW  = idle_W / 1000 * scale * pue
fac_mean_kW  = mean_W / 1000 * scale * pue
bess_pwr_kW  = (fac_peak_kW - fac_idle_kW) * 0.80
bess_e_kWh   = bess_pwr_kW * bess_dur
c_rate       = bess_pwr_kW / max(bess_e_kWh, 1e-9)
bess_pwr_MW  = bess_pwr_kW / 1000
bess_e_MWh   = bess_e_kWh  / 1000
chem         = "LFP — high power" if c_rate > 0.5 else "LFP — standard"

n24     = int(24 * 3600 / dt)
t24     = np.arange(n24) * dt / 3600
load_24 = np.full(n24, fac_idle_kW)
job_hrs = [2, 5, 8, 12, 16, 20]
rl      = rep_run["power_W"].values / 1000 * scale * pue
for sh in job_hrs:
    si = int(sh * 3600 / dt)
    ei = min(si + len(rl), n24)
    load_24[si:ei] = rl[:ei-si]

solar_kW    = gen_solar(n24, dt, fac_peak_kW * solar_frac / 100)
net_kW      = load_24 - solar_kW
soc_arr     = sim_soc(net_kW, bess_pwr_kW, bess_e_kWh,
                      rte, dt, soc_min, soc_max)
deficit_kWh = float(np.sum(net_kW.clip(min=0)) * dt / 3600)
surplus_kWh = float(np.sum((-net_kW).clip(min=0)) * dt / 3600)
total_kWh   = float(np.sum(load_24) * dt / 3600)
re_pct      = max(0.0, (1 - deficit_kWh/max(total_kWh,1e-9))*100)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER + KPIs
# ─────────────────────────────────────────────────────────────────────────────
mode_badge = "🎯 Demo" if is_demo else "🌐 Real NLR data"
st.title("⚡ BESS Engineering Dashboard")
st.caption(
    f"{mode_badge}  |  {data['file_num'].nunique()} runs  |  "
    f"dt={dt:.2f}s  |  {facility_nodes} nodes | "
    f"PUE {pue} | {solar_frac}% solar"
)

if is_demo:
    st.info(
        "🎯 **Demo mode** — synthetic data. "
        "Switch to **NLR dataset** in the sidebar for real data."
    )

with st.expander("📋  Metadata (first 10 rows)", expanded=False):
    st.dataframe(meta.head(10), use_container_width=True, height=180)

st.markdown("### 📊 Key metrics")
k1,k2,k3,k4,k5,k6,k7,k8 = st.columns(8)
for col, lbl, v in [
    (k1, "Peak load",     f"{fac_peak_kW:.0f} kW"),
    (k2, "Idle baseline", f"{fac_idle_kW:.0f} kW"),
    (k3, "Mean load",     f"{fac_mean_kW:.0f} kW"),
    (k4, "Load factor",   f"{lf*100:.1f} %"),
    (k5, "BESS power",    f"{bess_pwr_MW:.2f} MW"),
    (k6, "BESS energy",   f"{bess_e_MWh:.2f} MWh"),
    (k7, "C-rate",        f"{c_rate:.2f} C"),
    (k8, "RE coverage",   f"{re_pct:.1f} %"),
]:
    col.metric(lbl, v)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 All batches overview",
    "🔬 Single batch drill-down",
    "🔋 BESS sizing",
    "⚖️  Batch comparison",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ALL BATCHES OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Power profiles — all batches")
    fig_all = go.Figure()
    for g in groups:
        col_g = grp_colors[g]
        shown = False
        for fidx, run in data[data["group"]==g].groupby("file_num"):
            rs = run.sort_values("time_s")
            fig_all.add_trace(go.Scatter(
                x=rs["time_s"]/60, y=rs["power_W"]/1000,
                mode="lines", line=dict(color=col_g, width=0.9),
                opacity=0.40, name=f"{grp_lbl}={g}",
                legendgroup=str(g), showlegend=not shown,
                hovertemplate=(
                    f"{grp_lbl}={g} | t=%{{x:.1f}} min | "
                    "%{y:.2f} kW<extra></extra>"),
            ))
            shown = True
    fig_all.update_layout(
        xaxis_title="Time (min)", yaxis_title="Power (kW)",
        height=420, hovermode="x unified",
        legend=dict(title=grp_lbl, font=dict(size=10)),
        margin=dict(l=55,r=15,t=20,b=45),
    )
    st.plotly_chart(fig_all, use_container_width=True)

    r2a, r2b = st.columns(2)
    with r2a:
        fig_ldc = go.Figure()
        for g in groups:
            pwr = np.sort(
                data[data["group"]==g]["power_W"].values/1000)[::-1]
            pct = np.arange(1,len(pwr)+1)/len(pwr)*100
            fig_ldc.add_trace(go.Scatter(
                x=pct, y=pwr, mode="lines",
                name=f"{grp_lbl}={g}",
                line=dict(color=grp_colors[g], width=1.8),
            ))
        fig_ldc.update_layout(
            title="Load duration curve — all batches",
            xaxis_title="Time power exceeded (%)",
            yaxis_title="Power (kW)", height=350,
            margin=dict(l=55,r=15,t=50,b=45),
        )
        st.plotly_chart(fig_ldc, use_container_width=True)

    with r2b:
        fig_ramp = go.Figure()
        for g in groups:
            ramps = (data[data["group"]==g]
                     .groupby("file_num")["power_W"]
                     .diff().abs().div(dt).dropna()/1000)
            ramps = ramps[ramps <= ramps.quantile(0.999)]
            fig_ramp.add_trace(go.Histogram(
                x=ramps, name=f"{grp_lbl}={g}",
                marker_color=grp_colors[g],
                opacity=0.60, nbinsx=60,
            ))
        fig_ramp.add_vline(
            x=ramp_p95/1000, line_dash="dash", line_color="red",
            annotation_text=f"95th {ramp_p95/1000:.3f} kW/s",
            annotation_position="top right",
        )
        fig_ramp.update_layout(
            title="Ramp rate distribution — all batches",
            xaxis_title="kW/s", yaxis_title="Count",
            barmode="overlay", height=350,
            margin=dict(l=55,r=15,t=50,b=45),
        )
        st.plotly_chart(fig_ramp, use_container_width=True)

    st.markdown(f"### Statistics by {grp_lbl}")
    r3a, r3b, r3c = st.columns(3)
    with r3a:
        fig_box = go.Figure()
        for g in groups:
            fig_box.add_trace(go.Box(
                y=data[data["group"]==g]["power_W"]/1000,
                name=str(g), marker_color=grp_colors[g],
                boxmean=True, showlegend=False,
            ))
        fig_box.update_layout(
            title=f"Power distribution by {grp_lbl}",
            xaxis_title=grp_lbl, yaxis_title="Power (kW)",
            height=320, margin=dict(l=55,r=15,t=50,b=45),
        )
        st.plotly_chart(fig_box, use_container_width=True)

    with r3b:
        agg = (stats_df.groupby("Group")[
                   ["Peak (kW)","Mean (kW)","Idle (kW)"]]
               .mean().reset_index())
        fig_bar = go.Figure()
        for m in ["Peak (kW)","Mean (kW)","Idle (kW)"]:
            fig_bar.add_trace(go.Bar(
                x=agg["Group"].astype(str),
                y=agg[m].round(2), name=m))
        fig_bar.update_layout(
            title=f"Peak / mean / idle by {grp_lbl}",
            xaxis_title=grp_lbl, yaxis_title="Power (kW)",
            barmode="group", height=320,
            margin=dict(l=55,r=15,t=50,b=45),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with r3c:
        lf_agg = (stats_df.groupby("Group")["LF (%)"]
                  .agg(["mean","std"]).reset_index())
        fig_lf = go.Figure()
        fig_lf.add_trace(go.Bar(
            x=lf_agg["Group"].astype(str),
            y=lf_agg["mean"].round(1),
            error_y=dict(type="data",
                         array=lf_agg["std"].round(1)),
            marker_color=[grp_colors[g]
                          for g in lf_agg["Group"]],
            text=lf_agg["mean"].round(1).astype(str)+"%",
            textposition="outside",
        ))
        fig_lf.update_layout(
            title=f"Load factor by {grp_lbl}",
            xaxis_title=grp_lbl, yaxis_title="LF (%)",
            yaxis=dict(range=[0,115]), height=320,
            margin=dict(l=55,r=15,t=50,b=45),
        )
        st.plotly_chart(fig_lf, use_container_width=True)

    st.markdown("### Full statistics table")
    st.dataframe(
        stats_df.sort_values(["Group","File #"]),
        use_container_width=True, height=350,
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SINGLE BATCH DRILL-DOWN
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🔬 Single batch drill-down")
    st.caption(
        "Select any batch to see full detailed analysis — "
        "time-series, ramp rate, histogram, cumulative energy, "
        "power phases, load duration curve."
    )

    sel_c1, sel_c2, sel_c3 = st.columns([1, 1, 2])
    with sel_c1:
        grp_options = ["All groups"] + [str(g) for g in groups]
        sel_group   = st.selectbox(f"Filter by {grp_lbl}",
                                   grp_options)

    avail = (data[["file_num","group"]].drop_duplicates()
             if sel_group == "All groups"
             else data[data["group"] == sel_group][
                 ["file_num","group"]].drop_duplicates())
    file_nums = sorted(avail["file_num"].unique().tolist())

    with sel_c2:
        sel_file = st.selectbox(
            "Select batch (file #)",
            options=file_nums,
            format_func=lambda x: f"File {x:06d}",
        )

    with sel_c3:
        row_info = stats_df[stats_df["File #"] == sel_file]
        if not row_info.empty:
            r = row_info.iloc[0]
            st.info(
                f"**{grp_lbl}** = {r['Group']}  |  "
                f"Peak = {r['Peak (kW)']} kW  |  "
                f"Mean = {r['Mean (kW)']} kW  |  "
                f"Duration = {r['Duration (s)']} s  |  "
                f"Energy = {r['Energy (kWh)']} kWh"
            )

    st.divider()
    sel_run = (data[data["file_num"] == sel_file]
               .sort_values("time_s").reset_index(drop=True))
    sel_grp_val = (sel_run["group"].iloc[0]
                   if len(sel_run) > 0 else "?")

    meta_row = None
    try:
        wl_id_col = wl_def.get("id_col")
        if wl_id_col and wl_id_col in meta.columns:
            mm = meta[meta[wl_id_col].astype(int) == sel_file]
        else:
            mm = (meta.iloc[[sel_file]]
                  if sel_file < len(meta) else None)
        if mm is not None and len(mm) > 0:
            meta_row = mm.iloc[0]
    except Exception:
        meta_row = None

    if len(sel_run) == 0:
        st.warning("No data for selected batch.")
    else:
        show_single_batch(sel_run, sel_file, sel_grp_val,
                          grp_lbl, dt, meta_row)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BESS SIZING
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        "### 🔋 Facility-level BESS sizing & renewable integration"
    )

    fig_24 = go.Figure()
    fig_24.add_trace(go.Scatter(
        x=t24, y=load_24/1000,
        fill="tozeroy", fillcolor="rgba(226,75,74,0.15)",
        line=dict(color="#E24B4A", width=1.5),
        name="Facility load (MW)",
    ))
    fig_24.add_trace(go.Scatter(
        x=t24, y=solar_kW/1000,
        fill="tozeroy", fillcolor="rgba(29,158,117,0.20)",
        line=dict(color="#1D9E75", width=1.5),
        name=f"Solar PV — {solar_frac}% of peak",
    ))
    fig_24.add_trace(go.Scatter(
        x=t24, y=np.maximum(net_kW,0)/1000,
        line=dict(color="#534AB7", width=1.5, dash="dot"),
        name="Net grid demand",
    ))
    fig_24.add_trace(go.Scatter(
        x=t24, y=np.minimum(net_kW,0)/1000,
        fill="tozeroy", fillcolor="rgba(83,74,183,0.10)",
        line=dict(color="#9399b2", width=0.8),
        name="RE surplus → BESS charges",
    ))
    for sh in job_hrs:
        fig_24.add_vline(x=sh, line_dash="dot",
                         line_color="gray",
                         line_width=0.8, opacity=0.5)
    fig_24.update_layout(
        title=(f"RE coverage: {re_pct:.1f}%  |  "
               f"Firming gap: {deficit_kWh:.0f} kWh  |  "
               f"RE surplus: {surplus_kWh:.0f} kWh"),
        xaxis_title="Hour of day",
        yaxis_title="Power (MW)",
        height=380, hovermode="x unified",
        xaxis=dict(range=[0,24], dtick=2),
        margin=dict(l=55,r=15,t=55,b=45),
    )
    st.plotly_chart(fig_24, use_container_width=True)

    c_soc, c_spec = st.columns([2, 1])
    with c_soc:
        fig_soc = go.Figure()
        fig_soc.add_hrect(
            y0=soc_min, y1=soc_max,
            fillcolor="rgba(166,227,161,0.07)", line_width=0)
        fig_soc.add_trace(go.Scatter(
            x=t24, y=soc_arr,
            fill="tozeroy",
            fillcolor="rgba(186,117,23,0.20)",
            line=dict(color="#BA7517", width=2),
            name="BESS SoC (%)",
            hovertemplate=(
                "Hour %{x:.1f} | SoC %{y:.1f}%<extra></extra>"),
        ))
        fig_soc.add_hline(
            y=soc_min, line_dash="dash", line_color="#f38ba8",
            annotation_text=f"Min {soc_min}%",
            annotation_position="bottom right")
        fig_soc.add_hline(
            y=soc_max, line_dash="dash", line_color="#a6e3a1",
            annotation_text=f"Max {soc_max}%",
            annotation_position="top right")
        for sh in job_hrs:
            fig_soc.add_vline(x=sh, line_dash="dot",
                              line_color="gray",
                              line_width=0.8, opacity=0.5)
        fig_soc.update_layout(
            title=(f"BESS SoC — {bess_e_MWh:.2f} MWh | "
                   f"{bess_pwr_MW:.2f} MW | RTE {rte:.0%}"),
            xaxis_title="Hour of day",
            yaxis_title="SoC (%)",
            yaxis=dict(range=[0,105]),
            xaxis=dict(range=[0,24], dtick=2),
            height=360, hovermode="x unified",
            margin=dict(l=55,r=15,t=55,b=45),
        )
        st.plotly_chart(fig_soc, use_container_width=True)

    with c_spec:
        st.markdown("**BESS specification**")
        spec = {
            "Power rating"   : f"{bess_pwr_MW:.2f} MW",
            "Energy capacity": f"{bess_e_MWh:.2f} MWh",
            "C-rate"         : f"{c_rate:.2f} C",
            "Chemistry"      : chem,
            "SoC window"     : f"{soc_min} – {soc_max} %",
            "RTE"            : f"{rte:.0%}",
            "Ramp 95th"      : f"{ramp_p95/1000:.3f} kW/s",
            "DIV1"           : None,
            "Facility nodes" : str(facility_nodes),
            "PUE"            : str(pue),
            "Peak demand"    : f"{fac_peak_kW:.0f} kW",
            "Idle baseline"  : f"{fac_idle_kW:.0f} kW",
            "Mean demand"    : f"{fac_mean_kW:.0f} kW",
            "Load factor"    : f"{lf*100:.1f} %",
            "DIV2"           : None,
            "Solar capacity" : f"{fac_peak_kW*solar_frac/100:.0f} kW",
            "RE coverage"    : f"{re_pct:.1f} %",
            "Firming gap"    : f"{deficit_kWh:.0f} kWh/day",
            "RE surplus"     : f"{surplus_kWh:.0f} kWh/day",
        }
        for k, v in spec.items():
            if v is None:
                st.markdown(
                    "<hr style='margin:5px 0;border:none;"
                    "border-top:1px solid #444'>",
                    unsafe_allow_html=True)
            else:
                ca, cb = st.columns([3, 2])
                ca.markdown(
                    f"<span style='color:gray;"
                    f"font-size:12px'>{k}</span>",
                    unsafe_allow_html=True)
                cb.markdown(f"**{v}**")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BATCH COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### ⚖️  Side-by-side batch comparison")
    st.caption("Select 2–5 specific batches to compare directly.")

    all_fnums = sorted(data["file_num"].unique().tolist())
    sel_files = st.multiselect(
        "Select batches to compare",
        options=all_fnums,
        default=all_fnums[:min(3, len(all_fnums))],
        format_func=lambda x: (
            f"File {x:06d} | {grp_lbl}="
            f"{data[data['file_num']==x]['group'].iloc[0]} | "
            f"Peak="
            f"{stats_df[stats_df['File #']==x]['Peak (kW)'].values[0] if len(stats_df[stats_df['File #']==x])>0 else '?'} kW"
        ),
    )

    if len(sel_files) < 2:
        st.info("Please select at least 2 batches.")
    else:
        fig_cmp = go.Figure()
        for i, fn in enumerate(sel_files):
            run = data[data["file_num"]==fn].sort_values("time_s")
            grp = run["group"].iloc[0]
            fig_cmp.add_trace(go.Scatter(
                x=run["time_s"]/60, y=run["power_W"]/1000,
                mode="lines",
                name=f"File {fn} | {grp_lbl}={grp}",
                line=dict(color=get_color(i), width=1.5),
                hovertemplate=(
                    f"File {fn} | {grp_lbl}={grp}<br>"
                    "t=%{x:.1f} min | "
                    "%{y:.2f} kW<extra></extra>"),
            ))
        fig_cmp.update_layout(
            title="Power time-series — selected batches overlaid",
            xaxis_title="Time (min)", yaxis_title="Power (kW)",
            height=380, hovermode="x unified",
            margin=dict(l=55,r=15,t=50,b=45),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        cmp_a, cmp_b = st.columns(2)
        with cmp_a:
            fig_ldc2 = go.Figure()
            for i, fn in enumerate(sel_files):
                run = data[data["file_num"]==fn]
                grp = run["group"].iloc[0]
                pwr = np.sort(run["power_W"].values/1000)[::-1]
                pct = np.arange(1,len(pwr)+1)/len(pwr)*100
                fig_ldc2.add_trace(go.Scatter(
                    x=pct, y=pwr, mode="lines",
                    name=f"File {fn} | {grp_lbl}={grp}",
                    line=dict(color=get_color(i), width=2),
                ))
            fig_ldc2.update_layout(
                title="Load duration curves",
                xaxis_title="Duration exceeded (%)",
                yaxis_title="Power (kW)", height=340,
                margin=dict(l=55,r=15,t=50,b=45),
            )
            st.plotly_chart(fig_ldc2, use_container_width=True)

        with cmp_b:
            cmp_stats = stats_df[
                stats_df["File #"].isin(sel_files)].copy()
            metrics  = ["Peak (kW)","Mean (kW)",
                        "LF (%)","Ramp 95pc (kW/s)"]
            fig_cmp2 = go.Figure()
            for i, row in cmp_stats.iterrows():
                fn = int(row["File #"])
                fig_cmp2.add_trace(go.Bar(
                    name=f"File {fn} | {grp_lbl}={row['Group']}",
                    x=metrics,
                    y=[row[m] for m in metrics],
                    marker_color=get_color(sel_files.index(fn)),
                ))
            fig_cmp2.update_layout(
                title="Key metrics comparison",
                barmode="group", height=340,
                yaxis_title="Value",
                margin=dict(l=55,r=15,t=50,b=45),
            )
            st.plotly_chart(fig_cmp2, use_container_width=True)

        st.markdown("**Comparison statistics table**")
        st.dataframe(
            cmp_stats.set_index("File #"),
            use_container_width=True,
        )

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Dataset: Vercellino et al. (2026) — NLR/OT-2C00-99122 — "
    "DOI 10.7799/3025227  |  "
    "Solar: synthetic (sinusoidal) — replace with NSRDB for "
    "Golden, CO  |  "
    "BESS sizing indicative — full power system study required"
)
