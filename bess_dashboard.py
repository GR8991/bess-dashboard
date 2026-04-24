"""
=============================================================================
BESS Engineering Dashboard — NLR GenAI Power Profiles
=============================================================================
Run:  streamlit run bess_dashboard.py --server.maxMessageSize 1000

Tabs: All batches | Single batch | Node deep-dive | BESS sizing | Comparison
Data: Demo (instant) | NLR URL (~1 GB) | Local folder
Dataset: DOI 10.7799/3025227 — NLR Kestrel HPC
=============================================================================
"""

# ── standard library ─────────────────────────────────────────────────────────
import io
import pathlib
import tempfile
import warnings
import zipfile
from collections import Counter

warnings.filterwarnings("ignore")

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from plotly.subplots import make_subplots

# ── streamlit (must come before any st.* call) ───────────────────────────────
import streamlit as st

st.set_page_config(
    page_title="BESS Dashboard — NLR GenAI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE — initialise every key with a safe default
# ─────────────────────────────────────────────────────────────────────────────
for _k, _v in {
    "src"        : "demo",
    "base"       : "",
    "wl_found"   : {},
    "wl_cache"   : {},
    "demo_ready" : False,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
PAL     = px.colors.qualitative.Plotly
NLR_URL = "https://data.nlr.gov/system/files/312/1774982010-dataset.zip"

WL = {
    "training": {
        "label"   : "🏋️  Training (Llama2 LoRA + Stable Diffusion)",
        "id_col"  : "Unnamed: 0",
        "grp_col" : "nodes",
        "grp_lbl" : "Nodes",
        "extras"  : ["model", "repeat"],
    },
    "inference_offline_llama3_70b": {
        "label"   : "📦 Inference — Offline batch (Llama3 70B)",
        "id_col"  : None,
        "grp_col" : "batch_size",
        "grp_lbl" : "Batch size",
        "extras"  : ["repeat", "elapsed", "peak_power[W]", "mean_power[W]"],
    },
    "inference_online_rate_llama3_70b": {
        "label"   : "⚡ Inference — Online rate (Llama3 70B)",
        "id_col"  : None,
        "grp_col" : "request_rate",
        "grp_lbl" : "Req/s",
        "extras"  : ["num-prompts", "burstiness", "peak_power[W]", "mean_power[W]"],
    },
    "inference_online_finite_llama3_70b": {
        "label"   : "🎯 Inference — Online finite (Llama3 70B)",
        "id_col"  : None,
        "grp_col" : "num_prompts",
        "grp_lbl" : "Num prompts",
        "extras"  : ["request_rate_y", "duration", "peak_power[W]", "mean_power[W]"],
    },
}

def clr(i):
    return PAL[int(i) % len(PAL)]

# ─────────────────────────────────────────────────────────────────────────────
# DOWNSAMPLE — keeps charts fast and under the 200 MB browser limit
# ─────────────────────────────────────────────────────────────────────────────
def _ds(df_or_arr, n=400):
    """Return at most n evenly-spaced rows / elements."""
    if isinstance(df_or_arr, pd.DataFrame):
        if len(df_or_arr) <= n:
            return df_or_arr
        step = max(1, len(df_or_arr) // n)
        return df_or_arr.iloc[::step]
    else:                                   # numpy array or Series
        arr = np.asarray(df_or_arr)
        if len(arr) <= n:
            return arr
        step = max(1, len(arr) // n)
        return arr[::step]

# ─────────────────────────────────────────────────────────────────────────────
# DEMO DATA GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
def _demo():
    rng = np.random.default_rng(42)
    dt  = 0.2
    cfgs = [
        (2,  "llama2_70b_lora",  0),
        (2,  "llama2_70b_lora",  1),
        (4,  "llama2_70b_lora",  0),
        (4,  "llama2_70b_lora",  1),
        (8,  "llama2_70b_lora",  0),
        (8,  "stable_diffusion", 0),
        (16, "llama2_70b_lora",  0),
        (16, "stable_diffusion", 0),
    ]
    meta_rows, runs = [], []
    for idx, (nodes, model, repeat) in enumerate(cfgs):
        n      = int(rng.uniform(4000, 6000) / dt)
        t      = np.arange(n) * dt
        peak   = rng.uniform(2800, 3520) * nodes
        idle   = rng.uniform(380,  460)  * nodes
        rn     = int(30 / dt)
        dend   = n - int(15 / dt)
        pw     = np.zeros(n)
        for i in range(n):
            if i < rn:
                pw[i] = idle + (peak - idle) * (i / rn)
                pw[i] += rng.normal(0, peak * 0.01)
            elif i < dend:
                ph    = (i % int(10/dt)) / int(10/dt)
                pw[i] = peak * (0.92 + 0.05 * np.sin(ph * 2 * np.pi))
                pw[i] += rng.normal(0, peak * 0.01)
            else:
                f     = (i - dend) / max(n - dend, 1)
                pw[i] = peak * (1 - f) + idle * f
                pw[i] += rng.normal(0, peak * 0.005)
        pw = np.clip(pw, idle * 0.8, peak * 1.02)
        runs.append(pd.DataFrame({
            "time_s" : t,   "power_W": pw,
            "group"  : str(nodes), "file_num": idx,
            "nodes"  : nodes, "model": model, "repeat": repeat,
        }))
        meta_rows.append({"Unnamed: 0": idx, "model": model,
                          "nodes": nodes, "repeat": repeat})
    return pd.DataFrame(meta_rows), pd.concat(runs, ignore_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────
def _download(url):
    tmp = pathlib.Path(tempfile.mkdtemp()) / "nlr"
    tmp.mkdir(parents=True, exist_ok=True)
    zp  = tmp / "dataset.zip"
    bar = st.progress(0, text="⬇️  Connecting …")
    try:
        r = requests.get(url, stream=True, timeout=300,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        total, done = int(r.headers.get("content-length", 0)), 0
        with open(zp, "wb") as f:
            for chunk in r.iter_content(4 * 1024 * 1024):
                if chunk:
                    f.write(chunk); done += len(chunk)
                    bar.progress(min(done/total,1.) if total else 0,
                                 text=f"⬇️  {done//1024//1024} MB")
        bar.empty()
    except Exception as e:
        bar.empty(); st.error(f"❌  {e}"); return tmp
    mb = zp.stat().st_size // (1024*1024)
    if mb < 10:
        st.error(f"❌  File too small ({mb} MB)"); return tmp
    st.success(f"✅  Downloaded {mb} MB")
    with st.spinner("📦  Extracting …"):
        try:
            with zipfile.ZipFile(zp) as z: z.extractall(tmp)
            st.success("✅  Extracted")
        except Exception as e:
            st.error(f"❌  {e}"); return tmp
    hits = [p for p in tmp.rglob("01_aggregated_datasets") if p.is_dir()]
    return hits[0].parent if hits else tmp

# ─────────────────────────────────────────────────────────────────────────────
# SCAN / LOAD
# ─────────────────────────────────────────────────────────────────────────────
def _scan(base):
    agg = pathlib.Path(base) / "01_aggregated_datasets"
    if not agg.exists(): return {}
    return {f.name: WL[f.name]["label"]
            for f in sorted(agg.iterdir())
            if f.is_dir() and f.name in WL
            and list((f/"results").glob("*.parquet"))}

def _load(base, folder, maxn=50):
    wd  = pathlib.Path(base) / "01_aggregated_datasets" / folder
    res = wd / "results"
    w   = WL[folder]
    ic, gc = w["id_col"], w["grp_col"]

    meta = None
    for fn in ["metadata.csv", "metadata", "metadata.xlsx"]:
        mp = wd / fn
        if mp.exists():
            try:
                meta = pd.read_excel(mp) if mp.suffix==".xlsx" else pd.read_csv(mp)
                break
            except Exception: pass
    if meta is None:
        st.error(f"❌  No metadata in {wd}"); return None, None

    pairs = (list(zip(meta[ic].astype(int), [r for _,r in meta.iterrows()]))
             if ic and ic in meta.columns
             else list(zip(range(len(meta)), [r for _,r in meta.iterrows()])))

    if len(pairs) > maxn:
        step  = max(1, len(pairs)//maxn)
        pairs = pairs[::step][:maxn]
        st.info(f"ℹ️  Loading {len(pairs)} of {len(meta)} runs (every {step}th).")

    runs, miss, err = [], 0, 0
    bar = st.progress(0, text="Loading …")
    for i, (fn, row) in enumerate(pairs):
        fp = res / f"{int(fn):06d}.parquet"
        if not fp.exists(): miss += 1; continue
        try: raw = pd.read_parquet(fp)
        except Exception: err += 1; continue
        raw = raw.reset_index()
        raw.columns = ["time_s", "power_W"]
        raw["time_s"]  = pd.to_numeric(raw["time_s"],  errors="coerce")
        raw["power_W"] = pd.to_numeric(raw["power_W"], errors="coerce")
        raw = raw.dropna(subset=["time_s","power_W"])
        if raw.empty: continue
        raw["group"]    = str(row[gc]) if gc in row.index and pd.notna(row.get(gc)) else "all"
        raw["file_num"] = int(fn)
        for col in w.get("extras", []):
            if col in row.index and pd.notna(row.get(col)): raw[col] = row[col]
        runs.append(raw)
        bar.progress((i+1)/len(pairs), text=f"Loading … {i+1}/{len(pairs)}")
    bar.empty()
    if miss: st.warning(f"⚠️  {miss} missing")
    if err:  st.warning(f"⚠️  {err} unreadable")
    if not runs: st.error("❌  No data loaded."); return None, None
    return meta, pd.concat(runs, ignore_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────────────────────────────────────
def _stats(data, dt):
    rows = []
    for (g, fid), chunk in data.groupby(["group","file_num"]):
        chunk = chunk.sort_values("time_s")
        pw    = chunk["power_W"]
        ramp  = pw.diff().abs() / max(dt, 1e-9)
        # detect node count — group value IS the node count for training
        try:
            nn = int(float(g))
        except Exception:
            nn = 1
        nn = max(nn, 1)
        # also check nodes column
        if "nodes" in chunk.columns:
            try: nn = int(chunk["nodes"].iloc[0])
            except: pass
        pw_node  = pw / nn
        rmp_node = ramp / nn
        rows.append({
            "Nodes"         : nn,
            "File #"        : int(fid),
            "Peak kW/node"  : round(pw_node.max() /1000, 2),
            "Mean kW/node"  : round(pw_node.mean()/1000, 2),
            "Idle kW/node"  : round(pw_node.quantile(.05)/1000, 2),
            "LF %"          : round(pw_node.mean()/pw_node.max()*100, 1),
            "Dur s"         : round(chunk["time_s"].max(), 1),
            "kWh/node"      : round(pw_node.sum()*dt/3600/1000, 3),
            "R95 kW/s/node" : round(rmp_node.quantile(.95)/1000, 4),
            "Peak kW total" : round(pw.max()/1000, 2),
            "Model"         : chunk["model"].iloc[0] if "model" in chunk.columns else "?",
        })
    return pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────────────────────────
# SOLAR + SOC
# ─────────────────────────────────────────────────────────────────────────────
def _solar(n, dt, kW):
    h = np.arange(n) * dt / 3600
    return np.where((h>=6)&(h<=20), kW*np.maximum(0.,np.sin(np.pi*(h-6)/14)), 0.)

def _soc(net, pwr, cap, rte, dt, lo, hi):
    s = np.zeros(len(net)); lo_e=lo/100*cap; hi_e=hi/100*cap; s[0]=hi_e
    for i in range(1,len(net)):
        d=min(abs(net[i]),pwr)*dt/3600
        s[i]=(max(s[i-1]-d/rte,lo_e) if net[i]>0 else min(s[i-1]+d*rte,hi_e))
    return s/cap*100

# ─────────────────────────────────────────────────────────────────────────────
# PHASE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def _phases(pw_s, dt):
    pw = pw_s.values; mn,mx = pw.min(), pw.max(); rng=max(mx-mn,1.)
    lo,hi = mn+rng*.15, mn+rng*.85; n=len(pw)
    ri  = next((i for i in range(n)    if pw[i]>lo),  0)
    re  = next((i for i in range(ri,n) if pw[i]>hi),  ri)
    pe  = next((i for i in range(n-1,re,-1) if pw[i]>hi), re)
    rde = next((i for i in range(pe,n)      if pw[i]<lo), n-1)
    return ri, re, pe, rde

# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-BATCH PANEL
# ─────────────────────────────────────────────────────────────────────────────
def _batch_panel(run, fnum, gval, glbl, dt, mrow=None):
    pw=run["power_W"]; t=run["time_s"]
    ramp=pw.diff().abs()/max(dt,1e-9)
    pk=pw.max()/1000; mn=pw.mean()/1000; idl=pw.quantile(.05)/1000
    lf=pw.mean()/pw.max()*100; r95=ramp.quantile(.95)/1000
    en=pw.sum()*dt/3600/1000

    st.markdown(f"#### {glbl} = **{gval}** | File {fnum:06d}")
    cols=st.columns(8)
    for c,l,v in zip(cols,["Peak","Mean","Idle","LF %","Duration","Energy","R95","Std"],
        [f"{pk:.2f} kW",f"{mn:.2f} kW",f"{idl:.2f} kW",f"{lf:.1f} %",
         f"{t.max():.0f} s",f"{en:.3f} kWh",f"{r95:.4f} kW/s",
         f"{pw.std()/1000:.3f} kW"]):
        c.metric(l,v)
    st.divider()

    c1,c2=st.columns([2,1])
    with c1:
        win=max(10,int(30/dt))
        sl=_ds(run.sort_values("time_s"))
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=sl["time_s"],y=sl["power_W"]/1000,
            mode="lines",name="Power",line=dict(color="#378ADD",width=1.2)))
        fig.add_trace(go.Scatter(
            x=_ds(t),
            y=_ds(pw.rolling(win,center=True).mean())/1000,
            mode="lines",name=f"Mean({win*dt:.0f}s)",
            line=dict(color="#E24B4A",width=2,dash="dot")))
        fig.add_hline(y=pk, line_dash="dash",line_color="#BA7517",
                      annotation_text=f"Peak {pk:.2f}",annotation_position="top right")
        fig.add_hline(y=mn, line_dash="dot", line_color="#1D9E75",
                      annotation_text=f"Mean {mn:.2f}",annotation_position="bottom right")
        fig.update_layout(title="Power time-series",xaxis_title="Time (s)",
            yaxis_title="kW",height=320,hovermode="x unified",
            margin=dict(l=55,r=120,t=45,b=45))
        st.plotly_chart(fig,use_container_width=True)
    with c2:
        sl2=_ds(run.sort_values("time_s"))
        rmp2=sl2["power_W"].diff().abs()/max(dt,1e-9)/1000
        fig2=go.Figure()
        fig2.add_trace(go.Scatter(x=sl2["time_s"],y=rmp2,mode="lines",
            line=dict(color="#f38ba8",width=0.8),
            fill="tozeroy",fillcolor="rgba(243,139,168,.10)"))
        fig2.add_hline(y=r95,line_dash="dash",line_color="red",
                       annotation_text=f"95th {r95:.4f}")
        fig2.update_layout(title="Ramp rate dP/dt",xaxis_title="Time (s)",
            yaxis_title="kW/s",height=320,hovermode="x unified",
            margin=dict(l=55,r=100,t=45,b=45))
        st.plotly_chart(fig2,use_container_width=True)

    c3,c4,c5=st.columns(3)
    with c3:
        fig3=go.Figure()
        fig3.add_trace(go.Histogram(x=pw/1000,nbinsx=50,
            marker_color="#378ADD",opacity=.75))
        fig3.add_vline(x=mn,line_dash="dash",line_color="#1D9E75",
                       annotation_text=f"{mn:.2f}")
        fig3.add_vline(x=pk,line_dash="dash",line_color="#E24B4A",
                       annotation_text=f"{pk:.2f}")
        fig3.update_layout(title="Histogram",xaxis_title="kW",
            yaxis_title="Count",height=260,margin=dict(l=45,r=15,t=45,b=45))
        st.plotly_chart(fig3,use_container_width=True)
    with c4:
        cumE=np.cumsum(pw.values)*dt/3600/1000
        sl4=_ds(run.sort_values("time_s"))
        cumE4=_ds(pd.Series(cumE))
        fig4=go.Figure()
        fig4.add_trace(go.Scatter(x=sl4["time_s"],y=cumE4,mode="lines",
            fill="tozeroy",fillcolor="rgba(29,158,117,.15)",
            line=dict(color="#1D9E75",width=1.5)))
        fig4.update_layout(title="Cumulative energy",xaxis_title="Time (s)",
            yaxis_title="kWh/node",height=260,hovermode="x unified",
            margin=dict(l=55,r=15,t=45,b=45))
        st.plotly_chart(fig4,use_container_width=True)
    with c5:
        it=idl*1000*1.3; pt=pk*1000*.80
        ph=["Idle" if v<=it else "Plateau" if v>=pt else "Trans" for v in pw.values]
        cnt=Counter(ph)
        fig5=go.Figure(go.Pie(labels=["Idle","Trans","Plateau"],
            values=[cnt.get(l,0)*dt for l in ["Idle","Trans","Plateau"]],
            marker=dict(colors=["#888780","#BA7517","#E24B4A"]),
            hole=.4,textinfo="label+percent"))
        fig5.update_layout(title="Phase time",height=260,
            showlegend=False,margin=dict(l=15,r=15,t=45,b=15))
        st.plotly_chart(fig5,use_container_width=True)

    c6,c7=st.columns(2)
    with c6:
        sp=np.sort(pw.values/1000)[::-1]
        pct=np.arange(1,len(sp)+1)/len(sp)*100
        skip=max(1,len(sp)//400)
        fig6=go.Figure()
        fig6.add_trace(go.Scatter(x=pct[::skip],y=sp[::skip],mode="lines",
            line=dict(color="#534AB7",width=2),
            fill="tozeroy",fillcolor="rgba(83,74,183,.10)"))
        fig6.add_hline(y=mn,line_dash="dot",line_color="#1D9E75",
                       annotation_text=f"Mean {mn:.2f}")
        fig6.update_layout(title="Load duration curve",
            xaxis_title="% exceeded",yaxis_title="kW",
            height=260,margin=dict(l=55,r=100,t=45,b=45))
        st.plotly_chart(fig6,use_container_width=True)
    with c7:
        dp=pw.diff()
        ru=(ramp/1000)[dp>0].dropna(); rd=(ramp/1000)[dp<0].dropna()
        ru=ru[ru<=ru.quantile(.999)]; rd=rd[rd<=rd.quantile(.999)]
        fig7=make_subplots(rows=1,cols=2,subplot_titles=["Ramp-up","Ramp-down"])
        fig7.add_trace(go.Histogram(x=ru,nbinsx=40,
            marker_color="#E24B4A",opacity=.75),row=1,col=1)
        fig7.add_trace(go.Histogram(x=rd,nbinsx=40,
            marker_color="#378ADD",opacity=.75),row=1,col=2)
        fig7.update_layout(title_text="Up vs Down",height=260,
            showlegend=False,margin=dict(l=40,r=15,t=55,b=45))
        fig7.update_xaxes(title_text="kW/s")
        fig7.update_yaxes(title_text="Count",col=1)
        st.plotly_chart(fig7,use_container_width=True)

    if mrow is not None:
        with st.expander("📋  Metadata",expanded=False):
            st.dataframe(pd.DataFrame([mrow]).T.rename(columns={0:"Value"}),
                         use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# NODE DEEP-DIVE PANEL
# ─────────────────────────────────────────────────────────────────────────────
def _node_panel(run, nn, fnum, model, dt):
    df  = run.sort_values("time_s").reset_index(drop=True)
    pw  = df["power_W"] / max(nn,1)
    t   = df["time_s"]
    dpdt= pw.diff() / max(dt,1e-9)

    ri,re,pe,rde = _phases(pw,dt)
    pk  = float(pw.max())
    idl = float(pw.iloc[:max(ri,1)].mean()) if ri>0 else float(pw.quantile(.05))
    mn  = float(pw.iloc[re:pe].mean())  if pe>re else float(pw.mean())
    std = float(pw.iloc[re:pe].std())   if pe>re else 0.
    en  = float(pw.sum()*dt/3600/1000)
    ru_d= (re -ri )*dt;  rd_d=(rde-pe)*dt
    ru_r= (pk-idl)/max(ru_d,dt); rd_r=(pk-idl)/max(rd_d,dt)

    st.markdown(f"#### 🔬 Node deep-dive — **{nn}-node** | File {fnum:06d} | {model}")
    st.caption(f"Per-node = combined ÷ {nn}  |  dt={dt:.2f}s  |  {len(df):,} samples")

    cols=st.columns(9)
    for c,l,v in zip(cols,
        ["Peak/node","Idle/node","Mean plat","LF %",
         "Ramp↑ time","Ramp↓ time","dP/dt↑","dP/dt↓","Energy/node"],
        [f"{pk/1000:.2f} kW",f"{idl/1000:.2f} kW",f"{mn/1000:.2f} kW",
         f"{mn/pk*100:.1f} %",f"{ru_d:.0f} s",f"{rd_d:.0f} s",
         f"{ru_r:.0f} W/s",f"{rd_r:.0f} W/s",f"{en:.3f} kWh"]):
        c.metric(l,v)

    st.info(f"⚡ Ramp-up = **{ru_r:.0f} W/s/node** | "
            f"Ramp-down = **{rd_r:.0f} W/s/node** | "
            f"Ramp-down is **{rd_r/max(ru_r,1):.1f}×** faster → "
            f"size BESS inverter for the ramp-DOWN speed.")
    st.divider()

    # ── A: full profile ───────────────────────────────────────────────────────
    st.markdown("##### A — Full per-node power profile")
    SEG=[("Idle pre","#6b7280",0,ri),("Ramp-up","#ef4444",ri,re),
         ("Plateau","#f59e0b",re,pe),("Ramp-down","#3b82f6",pe,rde),
         ("Idle post","#6b7280",rde,len(df))]
    figA=go.Figure()
    for lbl,col,s,e in SEG:
        if e<=s: continue
        sl=_ds(df.iloc[s:e])
        figA.add_trace(go.Scatter(x=sl["time_s"],
            y=pw.iloc[s:e].iloc[::max(1,(e-s)//400)]/1000,
            mode="lines",name=lbl,line=dict(color=col,width=1.5)))
    for level,lbl,col in [(idl,f"Idle {idl/1000:.2f}","#6b7280"),
                           (pk, f"Peak {pk/1000:.2f}", "#ef4444"),
                           (mn, f"Mean {mn/1000:.2f}", "#f59e0b")]:
        figA.add_hline(y=level/1000,line_dash="dot",line_color=col,
                       annotation_text=lbl,annotation_position="right")
    figA.update_layout(title=f"{nn}-node {model} training — per node",
        xaxis_title="Time (s)",yaxis_title="kW/node",height=340,
        hovermode="x unified",
        legend=dict(orientation="h",y=1.1,font=dict(size=10)),
        margin=dict(l=55,r=150,t=55,b=45))
    st.plotly_chart(figA,use_container_width=True)
    st.divider()

    # ── B: ramp-up ────────────────────────────────────────────────────────────
    st.markdown("##### B — Ramp-up")
    bs=max(0,ri-int(15/dt)); be=min(len(df),re+int(20/dt))
    slB=_ds(df.iloc[bs:be]); pwB=_ds(pw.iloc[bs:be]); dpB=_ds(dpdt.iloc[bs:be])

    b1,b2=st.columns(2)
    with b1:
        figB=go.Figure()
        figB.add_trace(go.Scatter(x=slB["time_s"],y=pwB/1000,mode="lines",
            line=dict(color="#ef4444",width=1.5),
            fill="tozeroy",fillcolor="rgba(239,68,68,.08)"))
        figB.add_hline(y=idl/1000,line_dash="dot",line_color="#6b7280",
                       annotation_text=f"Idle {idl/1000:.2f}",annotation_position="right")
        figB.add_hline(y=pk/1000, line_dash="dash",line_color="#ef4444",
                       annotation_text=f"Peak {pk/1000:.2f}",annotation_position="right")
        figB.update_layout(title=f"Ramp-up | {ru_d:.0f}s | {ru_r:.0f} W/s",
            xaxis_title="s",yaxis_title="kW/node",height=280,
            hovermode="x unified",margin=dict(l=55,r=120,t=50,b=45))
        st.plotly_chart(figB,use_container_width=True)
    with b2:
        figBd=go.Figure()
        figBd.add_trace(go.Scatter(x=slB["time_s"],y=dpB,mode="lines",
            line=dict(color="#f59e0b",width=1.2),
            fill="tozeroy",fillcolor="rgba(245,158,11,.08)"))
        figBd.update_layout(title="dP/dt ramp-up (W/s)",
            xaxis_title="s",yaxis_title="W/s",height=280,
            hovermode="x unified",margin=dict(l=55,r=60,t=50,b=45))
        st.plotly_chart(figBd,use_container_width=True)

    with st.expander("📖  What am I seeing?",expanded=False):
        st.markdown(f"""
**Why ~{ru_d:.0f} s?** All {nn} nodes start simultaneously. GPUs load model
weights into HBM memory, then CUDA kernels initialise. Mean rate = **{ru_r:.0f} W/s**.

**BESS action:** Discharge at **{(pk-idl)*.80/1000:.2f} kW/node** instantly.
Grid ramps slowly. Data centre gets full power immediately.
        """)
    st.divider()

    # ── C: ramp-down ─────────────────────────────────────────────────────────
    st.markdown("##### C — Ramp-down  ⚠️  size your inverter for this")
    cs=max(0,pe-int(10/dt)); ce=min(len(df),rde+int(30/dt))
    slC=_ds(df.iloc[cs:ce]); pwC=_ds(pw.iloc[cs:ce]); dpC=_ds(dpdt.iloc[cs:ce])

    c1,c2=st.columns(2)
    with c1:
        figC=go.Figure()
        figC.add_trace(go.Scatter(x=slC["time_s"],y=pwC/1000,mode="lines",
            line=dict(color="#3b82f6",width=1.5),
            fill="tozeroy",fillcolor="rgba(59,130,246,.08)"))
        figC.add_hline(y=idl/1000,line_dash="dot",line_color="#6b7280",
                       annotation_text=f"Idle {idl/1000:.2f}",annotation_position="right")
        figC.update_layout(title=f"Ramp-down | {rd_d:.0f}s | {rd_r:.0f} W/s",
            xaxis_title="s",yaxis_title="kW/node",height=280,
            hovermode="x unified",margin=dict(l=55,r=120,t=50,b=45))
        st.plotly_chart(figC,use_container_width=True)
    with c2:
        figCd=go.Figure()
        figCd.add_trace(go.Scatter(x=slC["time_s"],y=dpC,mode="lines",
            line=dict(color="#f38ba8",width=1.2),
            fill="tozeroy",fillcolor="rgba(243,139,168,.08)"))
        figCd.update_layout(title="dP/dt ramp-down — NEGATIVE = falling fast",
            xaxis_title="s",yaxis_title="W/s",height=280,
            hovermode="x unified",margin=dict(l=55,r=60,t=50,b=45))
        st.plotly_chart(figCd,use_container_width=True)

    # normalised overlay
    if re>ri and rde>pe:
        rus=pw.iloc[ri:re].values; rds=pw.iloc[pe:rde].values
        ru_n=(rus-rus[0])/max(rus[-1]-rus[0],1.)
        rd_n=(rds[0]-rds)/max(rds[0]-rds[-1],1.)
        figOv=go.Figure()
        figOv.add_trace(go.Scatter(x=np.arange(len(ru_n))*dt,y=ru_n,mode="lines",
            name=f"Ramp-up ({ru_d:.0f}s)",line=dict(color="#ef4444",width=2)))
        figOv.add_trace(go.Scatter(x=np.arange(len(rd_n))*dt,y=rd_n,mode="lines",
            name=f"Ramp-down ({rd_d:.0f}s)",line=dict(color="#3b82f6",width=2)))
        figOv.update_layout(
            title=f"Normalised overlay — ramp-down is {rd_r/max(ru_r,1):.1f}× faster",
            xaxis_title="s from event",yaxis_title="0=idle → 1=peak",
            yaxis=dict(range=[-.05,1.1]),height=260,
            legend=dict(orientation="h",y=1.12,font=dict(size=10)),
            margin=dict(l=55,r=15,t=55,b=45))
        st.plotly_chart(figOv,use_container_width=True)

    with st.expander("📖  Why ramp-down is more dangerous",expanded=False):
        st.markdown(f"""
CUDA kernels stop instantly when job finishes. Power drops {pk/1000:.2f}→{idl/1000:.2f} kW
in **{rd_d:.0f}s**. Generators that were supplying the load now have excess energy →
frequency **rises**. Above **50.5 Hz** protection trips.

**BESS action:** Reverse to charge mode in **<100 ms** (4-quadrant inverter).
Facility-level rate = **{rd_r*nn*1.2/1000:.1f} kW/s** ({nn} nodes × PUE 1.2).
        """)
    st.divider()

    # ── D: plateau ────────────────────────────────────────────────────────────
    st.markdown("##### D — Plateau micro-cycling")
    plat_s=(pe-re)*dt
    if plat_s>5:
        z_opts=sorted(set([z for z in [10,30,60,120,300] if z<plat_s]+[int(min(plat_s,300))]))
        d1,d2=st.columns([2,1])
        with d1:
            zoom=st.select_slider("Zoom window (s)",options=z_opts,
                                  value=min(60,z_opts[-1]),key="pz")
        with d2:
            rw=st.slider("Rolling avg (steps)",1,100,25,key="rw")
            st.caption(f"= {rw*dt:.1f}s")
        zn=int(zoom/dt)
        slD=_ds(df.iloc[re:re+zn],n=500)
        pwD=_ds(pw.iloc[re:re+zn],n=500)
        rl =_ds(pw.iloc[re:re+zn].rolling(rw,center=True).mean(),n=500)
        dpD=_ds(dpdt.iloc[re:re+zn],n=500)

        figD=go.Figure()
        figD.add_trace(go.Scatter(x=slD["time_s"],y=pwD/1000,mode="lines",
            name="Raw",line=dict(color="rgba(59,130,246,.5)",width=0.8)))
        figD.add_trace(go.Scatter(x=slD["time_s"],y=rl/1000,mode="lines",
            name=f"Mean ({rw*dt:.1f}s)",line=dict(color="#f59e0b",width=2.5)))
        figD.add_hline(y=mn/1000,line_dash="dot",line_color="#1D9E75",
                       annotation_text=f"Mean {mn/1000:.2f}",annotation_position="right")
        figD.update_layout(
            title=f"Plateau {zoom}s | mean={mn/1000:.2f} kW | std=±{std/1000:.3f} kW",
            xaxis_title="s",yaxis_title="kW/node",height=280,
            hovermode="x unified",
            legend=dict(orientation="h",y=1.12,font=dict(size=10)),
            margin=dict(l=55,r=120,t=55,b=45))
        st.plotly_chart(figD,use_container_width=True)

        pd1,pd2=st.columns(2)
        with pd1:
            figDd=go.Figure()
            figDd.add_trace(go.Scatter(x=slD["time_s"],y=dpD,mode="lines",
                line=dict(color="#8b5cf6",width=0.8),
                fill="tozeroy",fillcolor="rgba(139,92,246,.08)"))
            figDd.update_layout(title="dP/dt plateau (micro-cycling)",
                xaxis_title="s",yaxis_title="W/s",height=240,
                hovermode="x unified",margin=dict(l=55,r=60,t=50,b=45))
            st.plotly_chart(figDd,use_container_width=True)
        with pd2:
            pp=pw.iloc[re:pe]
            if len(pp)>1:
                sp=np.sort(pp.values/1000)[::-1]
                pct=np.arange(1,len(sp)+1)/len(sp)*100
                sk=max(1,len(sp)//400)
                figL=go.Figure()
                figL.add_trace(go.Scatter(x=pct[::sk],y=sp[::sk],mode="lines",
                    line=dict(color="#10b981",width=1.5),
                    fill="tozeroy",fillcolor="rgba(16,185,129,.08)"))
                figL.add_hline(y=mn/1000,line_dash="dot",line_color="#f59e0b",
                               annotation_text=f"Mean {mn/1000:.2f}")
                figL.update_layout(title="LDC plateau only",
                    xaxis_title="% exceeded",yaxis_title="kW/node",
                    height=240,margin=dict(l=55,r=100,t=50,b=45))
                st.plotly_chart(figL,use_container_width=True)
    else:
        st.info("Plateau not detected in this run.")
    st.divider()

    # ── E: freq + BESS spec ───────────────────────────────────────────────────
    st.markdown("##### E — Grid frequency impact & BESS specification")
    bpwr=(pk-idl)*.80/1000; be=bpwr*.25; bcr=bpwr/max(be,1e-6)

    fe1,fe2=st.columns(2)
    with fe1:
        rng2=np.random.default_rng(42); ns=300; ts=np.arange(ns)*.5
        fn_no =np.where(ts<5,50.,50.-.28*(1-np.exp(-(ts-5)/8))*np.exp(-(ts-5)/40)+rng2.normal(0,.003,ns))
        rng3=np.random.default_rng(99)
        fn_yes=np.where(ts<5,50.,50.-.012*(1-np.exp(-(ts-5)/3))*np.exp(-(ts-5)/12)+rng3.normal(0,.0004,ns))
        figF=go.Figure()
        figF.add_trace(go.Scatter(x=ts,y=fn_no, mode="lines",name="Without BESS",
            line=dict(color="#ef4444",width=1.5)))
        figF.add_trace(go.Scatter(x=ts,y=fn_yes,mode="lines",name="With BESS",
            line=dict(color="#10b981",width=1.5)))
        figF.add_hline(y=50.,line_dash="dot",line_color="rgba(200,200,200,.3)",
                       annotation_text="50 Hz")
        figF.add_hrect(y0=49.5,y1=49.8,fillcolor="rgba(245,158,11,.07)",line_width=0,
                       annotation_text="Warning",annotation_position="right")
        figF.update_layout(title="Grid freq during ramp-up (illustrative)",
            xaxis_title="s",yaxis_title="Hz",yaxis=dict(range=[49.2,50.1]),
            height=300,hovermode="x unified",
            legend=dict(orientation="h",y=1.12,font=dict(size=10)),
            margin=dict(l=55,r=60,t=55,b=45))
        st.plotly_chart(figF,use_container_width=True)
    with fe2:
        st.markdown("**Per-node BESS spec**")
        for k,v in [
            ("Peak/node",         f"{pk/1000:.2f} kW"),
            ("Idle/node",         f"{idl/1000:.2f} kW"),
            ("Ramp-up rate",      f"{ru_r:.0f} W/s"),
            ("Ramp-down rate",    f"**{rd_r:.0f} W/s ← size for this**"),
            (None,None),
            ("BESS power/node",   f"{bpwr:.2f} kW"),
            ("BESS energy/node",  f"{be:.3f} kWh"),
            ("C-rate",            f"{bcr:.2f} C"),
            ("Chemistry",         "LFP high-power" if bcr>.5 else "LFP standard"),
            (None,None),
            (f"× {nn} nodes",     ""),
            ("BESS power total",  f"{bpwr*nn:.2f} kW"),
            ("BESS energy total", f"{be*nn:.3f} kWh"),
            ("× PUE 1.2",         ""),
            ("Grid BESS power",   f"{bpwr*nn*1.2:.2f} kW"),
        ]:
            if k is None:
                st.markdown("<hr style='margin:4px 0;border:none;"
                            "border-top:1px solid #333'>",unsafe_allow_html=True)
            elif v=="":
                st.markdown(f"<span style='color:#888;font-size:11px;"
                            f"font-style:italic'>{k}</span>",unsafe_allow_html=True)
            else:
                ca,cb=st.columns([3,2])
                ca.markdown(f"<span style='color:gray;font-size:12px'>{k}</span>",
                            unsafe_allow_html=True)
                cb.markdown(f"**{v}**")


    st.divider()

    # ── F: FREQUENCY ANALYSIS ─────────────────────────────────────────────────
    st.markdown("##### F — Frequency analysis of power signal (per phase)")
    st.caption(
        "FFT of the power signal in each phase. "
        "Frequency content = how fast your BESS inverter must respond."
    )

    with st.expander("📖  What does frequency mean here?", expanded=True):
        st.markdown("""
**Frequency of power signal** = how many times per second the power level changes.

| Frequency | Period | What causes it | BESS implication |
|---|---|---|---|
| 0.001 Hz | 1000 s | Whole job (idle→peak→idle) | Energy capacity |
| 0.01 Hz | 100 s | Ramp-up / ramp-down | Power rating |
| 0.1 Hz | 10 s | One training step cycle | Inverter speed |
| 1 Hz | 1 s | GPU micro-fluctuations | Inverter bandwidth |
| 2.5 Hz | 0.4 s | Nyquist limit (dt=0.2 s) | Measurement limit |

**Rule:** BESS inverter bandwidth ≥ 10× the highest significant power frequency.
        """)

    def _fft_phase(sig_w, dt_s, n_pts=800):
        n = len(sig_w)
        if n < 8: return None
        sig   = sig_w - sig_w.mean()
        fvals = np.abs(np.fft.rfft(sig)) / n * 2
        freqs = np.fft.rfftfreq(n, d=dt_s)
        freqs, fvals = freqs[1:], fvals[1:]
        step  = max(1, len(freqs) // n_pts)
        dom   = int(np.argmax(fvals))
        df_hz = float(freqs[dom]) if len(freqs) > 0 else 0.
        dT    = 1./df_hz if df_hz > 0 else float("inf")
        return freqs[::step], fvals[::step], df_hz, dT

    ph_segs = [
        ("Idle (pre-job)",      pw.iloc[:max(ri,2)].values,           "#6b7280"),
        ("Ramp-up",             pw.iloc[ri:max(re,ri+2)].values,      "#ef4444"),
        ("Plateau (operating)", pw.iloc[re:max(pe,re+2)].values,      "#f59e0b"),
        ("Ramp-down",           pw.iloc[pe:max(rde,pe+2)].values,     "#3b82f6"),
        ("Idle (post-job)",     pw.iloc[rde:].values,                  "#6b7280"),
    ]

    st.markdown("**Power spectrum (FFT) — all phases**")
    figFFT = go.Figure()
    dom_rows = []
    for lbl, seg, col in ph_segs:
        if len(seg) < 8: continue
        res = _fft_phase(seg, dt)
        if res is None: continue
        fq, am, df_hz, dT = res
        figFFT.add_trace(go.Scatter(x=fq, y=am/1000, mode="lines", name=lbl,
            line=dict(color=col, width=1.5),
            hovertemplate=f"{lbl}<br>f=%{{x:.4f}} Hz|A=%{{y:.3f}} kW<extra></extra>"))
        dom_rows.append({
            "Phase"           : lbl,
            "Duration (s)"    : round(len(seg)*dt, 1),
            "Dom freq (Hz)"   : round(df_hz, 5),
            "Dom period (s)"  : round(dT, 1) if dT < 1e6 else "DC",
            "Peak amp (kW)"   : round(float(am.max())/1000, 3),
            "RMS fluct (kW)"  : round(float(np.std(seg))/1000, 3),
        })
    for fref, flbl in [(0.01,"Ramp ~0.01Hz"),(0.1,"Step ~0.1Hz"),(1.0,"GPU ~1Hz")]:
        figFFT.add_vline(x=fref, line_dash="dot",
                         line_color="rgba(200,200,200,0.3)",
                         annotation_text=flbl, annotation_font_size=9,
                         annotation_position="top right")
    figFFT.update_layout(
        title="Power spectrum — all phases overlaid",
        xaxis_title="Frequency (Hz)", yaxis_title="Amplitude (kW)",
        xaxis_type="log", xaxis=dict(range=[-3, 0.4]),
        height=360, hovermode="x unified",
        legend=dict(orientation="h", y=1.12, font=dict(size=10)),
        margin=dict(l=55, r=15, t=55, b=45))
    st.plotly_chart(figFFT, use_container_width=True)

    if dom_rows:
        st.markdown("**Dominant frequency table — BESS bandwidth requirements**")
        st.dataframe(pd.DataFrame(dom_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("**dP/dt per phase — BESS ramp rate specification**")

    ph_list = [
        ("Idle (pre)",     0,   ri,      "#6b7280"),
        ("Ramp-up",        ri,  re,      "#ef4444"),
        ("Plateau/Token",  re,  pe,      "#f59e0b"),
        ("Ramp-down",      pe,  rde,     "#3b82f6"),
        ("Idle (post)",    rde, len(df), "#6b7280"),
    ]
    fp1, fp2 = st.columns(2)
    with fp1:
        pnames, pmx, prms, pcols = [], [], [], []
        for plbl, s, e, col in ph_list:
            seg_dp = dpdt.iloc[s:e].abs().dropna()
            if len(seg_dp) < 2: continue
            pnames.append(plbl)
            pmx.append(round(float(seg_dp.quantile(0.95))/1000, 3))
            prms.append(round(float(seg_dp.mean())/1000, 3))
            pcols.append(col)
        figBar = go.Figure()
        figBar.add_trace(go.Bar(x=pnames, y=pmx, name="95th |dP/dt|",
            marker_color=pcols, opacity=0.85,
            text=[f"{v}" for v in pmx], textposition="outside"))
        figBar.add_trace(go.Bar(x=pnames, y=prms, name="Mean |dP/dt|",
            marker_color=pcols, opacity=0.45))
        figBar.update_layout(title="|dP/dt| per phase → BESS ramp rate spec",
            xaxis_title="Phase", yaxis_title="kW/s", barmode="group",
            height=320, legend=dict(orientation="h", y=1.12, font=dict(size=10)),
            margin=dict(l=55, r=15, t=55, b=80))
        st.plotly_chart(figBar, use_container_width=True)
    with fp2:
        figHist = go.Figure()
        for plbl, s, e, col in ph_list:
            sdp = dpdt.iloc[s:e].dropna()/1000
            sdp = sdp[sdp.abs() <= sdp.abs().quantile(0.999)]
            if len(sdp) < 5: continue
            figHist.add_trace(go.Histogram(x=sdp, name=plbl,
                marker_color=col, opacity=0.55, nbinsx=60))
        figHist.update_layout(title="dP/dt distribution — all phases",
            xaxis_title="dP/dt (kW/s)", yaxis_title="Count",
            barmode="overlay", height=320,
            legend=dict(orientation="h", y=1.12, font=dict(size=10)),
            margin=dict(l=55, r=15, t=55, b=45))
        st.plotly_chart(figHist, use_container_width=True)

    st.divider()
    st.markdown("**Token generation — power behaviour during each Llama2 training step**")
    st.caption("Each training step = forward pass + backward pass + weight update = one 'token batch'.")

    if pe > re + int(60/dt):
        tk_s = re + int(30/dt)
        tk_e = min(tk_s + int(120/dt), pe)
        sl_tk = df.iloc[tk_s:tk_e]
        pw_tk = pw.iloc[tk_s:tk_e]
        dp_tk = dpdt.iloc[tk_s:tk_e]
        t_rel = sl_tk["time_s"] - sl_tk["time_s"].iloc[0]

        figTk = make_subplots(rows=3, cols=1,
            subplot_titles=[
                "Power per node (120 s window during plateau)",
                "dP/dt — rate of change every 0.2 s",
                "FFT — frequency content during token generation",
            ], vertical_spacing=0.12)

        sl2 = _ds(pd.DataFrame({"t": t_rel.values, "pw": pw_tk.values}), n=600)
        figTk.add_trace(go.Scatter(x=sl2["t"], y=sl2["pw"]/1000, mode="lines",
            line=dict(color="#f59e0b", width=1.2)), row=1, col=1)
        figTk.add_hline(y=float(pw_tk.mean())/1000, line_dash="dot",
                        line_color="#1D9E75",
                        annotation_text=f"Mean {pw_tk.mean()/1000:.2f} kW",
                        row=1, col=1)

        dp2 = _ds(dp_tk, n=600); t2 = _ds(pd.Series(t_rel.values), n=600)
        figTk.add_trace(go.Scatter(x=t2, y=dp2/1000, mode="lines",
            line=dict(color="#8b5cf6", width=0.8),
            fill="tozeroy", fillcolor="rgba(139,92,246,0.08)"), row=2, col=1)
        figTk.add_hline(y= float(dp_tk.abs().quantile(0.95))/1000,
                        line_dash="dash", line_color="#ef4444",
                        annotation_text="95th pct", row=2, col=1)
        figTk.add_hline(y=-float(dp_tk.abs().quantile(0.95))/1000,
                        line_dash="dash", line_color="#3b82f6", row=2, col=1)

        res_tk = _fft_phase(pw_tk.values, dt)
        if res_tk:
            fq_tk, am_tk, df_tk, dT_tk = res_tk
            figTk.add_trace(go.Scatter(x=fq_tk, y=am_tk/1000, mode="lines",
                line=dict(color="#8b5cf6", width=1.5)), row=3, col=1)
            figTk.add_vline(x=df_tk, line_dash="dash", line_color="#ef4444",
                annotation_text=f"Peak {df_tk:.4f} Hz = {dT_tk:.1f}s/cycle",
                row=3, col=1)
            st.info(
                f"**Token generation frequency = {df_tk:.4f} Hz "
                f"(one cycle every {dT_tk:.1f} seconds).**  "
                f"This is the forward+backward pass rhythm of Llama2 training.  "
                f"BESS must track ±{float(dp_tk.abs().quantile(0.95))/1000:.3f} kW/s "
                f"fluctuations at this frequency continuously."
            )

        figTk.update_xaxes(title_text="Time (s)", row=1, col=1)
        figTk.update_xaxes(title_text="Time (s)", row=2, col=1)
        figTk.update_xaxes(title_text="Frequency (Hz)", type="log", row=3, col=1)
        figTk.update_yaxes(title_text="kW/node",  row=1, col=1)
        figTk.update_yaxes(title_text="kW/s",     row=2, col=1)
        figTk.update_yaxes(title_text="Amp (kW)", row=3, col=1)
        figTk.update_layout(height=680, showlegend=False,
            margin=dict(l=55, r=15, t=60, b=45))
        st.plotly_chart(figTk, use_container_width=True)

    st.divider()
    st.markdown("**BESS inverter bandwidth summary**")
    dp_plat = float(dpdt.iloc[re:pe].abs().quantile(0.95)) if pe>re else 0.
    dp_up   = float(dpdt.iloc[ri:re].abs().quantile(0.95)) if re>ri else 0.
    dp_dn   = float(dpdt.iloc[pe:rde].abs().quantile(0.95)) if rde>pe else 0.
    dp_max  = max(dp_plat, dp_up, dp_dn)

    bw1,bw2,bw3,bw4 = st.columns(4)
    bw1.metric("Plateau dP/dt 95th", f"{dp_plat/1000:.3f} kW/s")
    bw2.metric("Ramp-up dP/dt 95th", f"{dp_up/1000:.3f} kW/s")
    bw3.metric("Ramp-down dP/dt 95th",f"{dp_dn/1000:.3f} kW/s",
               delta="worst case ↑", delta_color="inverse")
    bw4.metric("Min inverter bandwidth",f"≥ {dp_max/1000*10:.1f} kW/s",
               help="10× max dP/dt — standard control engineering rule")

    st.success(
        f"**{nn}-node {model} summary:**  "
        f"Idle = 0 W/s  |  "
        f"Ramp-up = +{dp_up/1000:.3f} kW/s  |  "
        f"Plateau (token gen) = ±{dp_plat/1000:.3f} kW/s  |  "
        f"Ramp-down = -{dp_dn/1000:.3f} kW/s  ← worst case  |  "
        f"Facility-level min bandwidth = "
        f"{dp_max/1000*10*nn*1.2:.1f} kW/s ({nn} nodes × PUE 1.2)"
    )

# ─────────────────────────────────────────────────────────────────────────────
# ── SIDEBAR ──────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ BESS Dashboard")
    st.caption("NLR GenAI — DOI 10.7799/3025227")
    st.divider()

    st.markdown("**Step 1 — Data source**")
    _modes=["demo","nlr","local"]
    _lbl={"demo":"🎯 Demo (instant)","nlr":"🌐 NLR dataset (~1 GB)","local":"💾 Local folder"}
    _cur=st.session_state.get("src","demo")
    _cur=_cur if _cur in _modes else "demo"
    src=st.radio("Source",_modes,format_func=lambda x:_lbl[x],index=_modes.index(_cur))
    st.session_state["src"]=src

    if src=="demo":
        st.info("Synthetic H100 profiles — instant.")
        if st.button("▶️  Load demo data",type="primary",use_container_width=True):
            with st.spinner("Generating …"):
                _m,_d=_demo()
            st.session_state["wl_cache"]["demo"]={"meta":_m,"data":_d,
                "folder":"training","is_demo":True}
            st.session_state["demo_ready"]=True
            st.success("✅  Ready")

    elif src=="nlr":
        st.info("Official public URL — no login. ~1 GB, 5–20 min.")
        _url=st.text_input("URL",value=NLR_URL)
        if st.button("⬇️  Download",type="primary",use_container_width=True):
            try:
                _base=_download(_url.strip())
                _found=_scan(str(_base))
                if _found:
                    st.session_state.update({"base":str(_base),"wl_found":_found,"wl_cache":{}})
                    st.success(f"✅  {len(_found)} workload(s)")
                else:
                    st.error("❌  No workloads found.")
            except Exception as e:
                st.error(f"❌  {e}")
    else:
        _lp=st.text_input("Folder path", placeholder="e.g. C:/Users/you/Downloads/dataset", value="")
        if st.button("🔍  Scan",type="primary",use_container_width=True):
            _found=_scan(_lp.strip())
            if _found:
                st.session_state.update({"base":_lp.strip(),"wl_found":_found,"wl_cache":{}})
                st.success(f"✅  {len(_found)} workload(s)")
            else:
                st.error("❌  Not found")

    _wlf=st.session_state.get("wl_found",{})
    _demo_ok=src=="demo" and st.session_state.get("demo_ready",False)

    if not _demo_ok and _wlf:
        st.divider()
        st.markdown("**Step 2 — Select workload**")
        fn=st.radio("Workloads",list(_wlf.keys()),format_func=lambda k:_wlf[k])
        mx=st.slider("Max runs",10,200,30,10)
        st.caption("Keep ≤ 50 to avoid browser memory errors.")
        if st.button("📂  Load",type="primary",use_container_width=True):
            with st.spinner("Loading …"):
                _m,_d=_load(st.session_state["base"],fn,mx)
            if _d is not None:
                st.session_state["wl_cache"][fn]={"meta":_m,"data":_d,
                    "folder":fn,"is_demo":False}
                st.success("✅  Loaded")
    elif not _demo_ok:
        st.info("Load data above first.")

    st.divider()
    st.markdown("**Step 3 — Facility & BESS**")
    fac_n  =st.slider("Total GPU nodes",       10, 500,156,10)
    pue    =st.slider("PUE",                 1.05,2.00,1.20,.05)
    sol_f  =st.slider("Solar PV (% peak)",     20, 100, 60,  5)
    b_dur  =st.slider("BESS duration (h)",       1,   8,  4,  1)
    rte    =st.slider("Round-trip eff.",       .80, .98,.92,.01)
    soc_lo =st.slider("Min SoC %",   5, 30,20, 5)
    soc_hi =st.slider("Max SoC %",  70, 99,90, 5)

# ─────────────────────────────────────────────────────────────────────────────
# RESOLVE ACTIVE DATA — completely safe, no st.stop()
# ─────────────────────────────────────────────────────────────────────────────
_cache  = st.session_state.get("wl_cache",{})
_src    = st.session_state.get("src","demo")
_active = None

if _src=="demo" and "demo" in _cache:
    _active="demo"
else:
    for _k in reversed(list(_cache.keys())):
        if _k!="demo": _active=_k; break

# check data is genuinely usable
_ok = (
    _active is not None
    and _active in _cache
    and isinstance(_cache[_active].get("data"), pd.DataFrame)
    and not _cache[_active]["data"].empty
    and "file_num" in _cache[_active]["data"].columns
)

# ─────────────────────────────────────────────────────────────────────────────
# WELCOME SCREEN — shown if no data loaded
# ─────────────────────────────────────────────────────────────────────────────
if not _ok:
    st.title("⚡ BESS Engineering Dashboard")
    st.markdown("### Welcome — NLR GenAI Power Profiles")
    st.info(
        "👈 **Get started in the sidebar:**\n\n"
        "1. **🎯 Demo mode** → click **▶️ Load demo data** — instant\n"
        "2. **🌐 NLR dataset** → click **⬇️ Download** — real data, ~1 GB\n"
        "3. **💾 Local folder** → paste your dataset path → **🔍 Scan** → **📂 Load**"
    )

# ─────────────────────────────────────────────────────────────────────────────
# MAIN DASHBOARD — only runs when _ok is True
# ─────────────────────────────────────────────────────────────────────────────
if _ok:
    meta    = _cache[_active]["meta"]
    data    = _cache[_active]["data"]
    is_demo = _cache[_active].get("is_demo",False)
    folder  = _cache[_active].get("folder","training")
    wdef    = WL.get(folder, WL["training"])
    glbl    = wdef["grp_lbl"]

    # derived quantities
    dt_v = (data.sort_values(["file_num","time_s"])
                .groupby("file_num")["time_s"].diff().dropna())
    dt   = float(dt_v.median()) if len(dt_v)>0 else 0.2
    dt   = max(dt, 0.01)

    groups = sorted(data["group"].unique().tolist())
    gcols  = {g:clr(i) for i,g in enumerate(groups)}
    # also add integer keys so node count lookups never fail
    for _g in list(gcols.keys()):
        try: gcols[int(float(_g))] = gcols[_g]
        except: pass
    sdf    = _stats(data,dt)

    pk_W  = float(data["power_W"].max())
    idl_W = float(data["power_W"].quantile(.05))
    mn_W  = float(data["power_W"].mean())
    lf    = mn_W/pk_W if pk_W>0 else 0
    r95   = float(data.sort_values(["file_num","time_s"])
                      .groupby("file_num")["power_W"]
                      .diff().abs().div(dt).quantile(.95))

    top   = int(sdf.sort_values("Peak kW/node",ascending=False).iloc[0]["File #"])
    rep   = data[data["file_num"]==top].sort_values("time_s").reset_index(drop=True)
    rn    = 1
    if "nodes" in rep.columns:
        try: rn=int(rep["nodes"].iloc[0])
        except: pass

    sc      = fac_n/max(rn,1)
    f_pk    = pk_W /1000*sc*pue
    f_idl   = idl_W/1000*sc*pue
    f_mn    = mn_W /1000*sc*pue
    bp_kW   = (f_pk-f_idl)*.80
    be_kWh  = bp_kW*b_dur
    cr      = bp_kW/max(be_kWh,1e-9)
    bp_MW   = bp_kW/1000; be_MWh=be_kWh/1000
    chem    = "LFP high-power" if cr>.5 else "LFP standard"

    n24     = int(24*3600/dt); t24=np.arange(n24)*dt/3600
    L24     = np.full(n24,f_idl)
    rl      = rep["power_W"].values/1000*sc*pue
    for sh in [2,5,8,12,16,20]:
        si=int(sh*3600/dt); ei=min(si+len(rl),n24); L24[si:ei]=rl[:ei-si]
    sol     = _solar(n24,dt,f_pk*sol_f/100)
    net     = L24-sol
    soc     = _soc(net,bp_kW,be_kWh,rte,dt,soc_lo,soc_hi)
    def_kWh = float(np.sum(net.clip(min=0))*dt/3600)
    sur_kWh = float(np.sum((-net).clip(min=0))*dt/3600)
    tot_kWh = float(np.sum(L24)*dt/3600)
    re_pct  = max(0.,(1-def_kWh/max(tot_kWh,1e-9))*100)

    # ── header ───────────────────────────────────────────────────────────────
    st.title("⚡ BESS Engineering Dashboard")
    st.caption(
        f"{'🎯 Demo' if is_demo else '🌐 Real NLR'}  |  "
        f"{data['file_num'].nunique()} runs  |  dt={dt:.2f}s  |  "
        f"{fac_n} nodes | PUE {pue} | {sol_f}% solar"
    )
    if is_demo:
        st.info("🎯 Demo mode — synthetic data. Switch to NLR dataset for real measurements.")
    if meta is not None:
        with st.expander("📋  Metadata",expanded=False):
            st.dataframe(meta.head(10),use_container_width=True,height=180)

    st.markdown("### 📊 Key metrics")
    kcols=st.columns(8)
    for c,l,v in zip(kcols,
        ["Peak load","Idle baseline","Mean load","Load factor",
         "BESS power","BESS energy","C-rate","RE coverage"],
        [f"{f_pk:.0f} kW",f"{f_idl:.0f} kW",f"{f_mn:.0f} kW",f"{lf*100:.1f} %",
         f"{bp_MW:.2f} MW",f"{be_MWh:.2f} MWh",f"{cr:.2f} C",f"{re_pct:.1f} %"]):
        c.metric(l,v)
    st.divider()

    # ── tabs ─────────────────────────────────────────────────────────────────
    t0,t1,t2,t3,t4,t5=st.tabs([
        "🖥️ Node overview","📊 All runs","🔬 Single run","🔍 Node deep-dive",
        "🔋 BESS sizing","⚖️ Comparison"])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 0 — NODE OVERVIEW  (primary engineering view)
    # ════════════════════════════════════════════════════════════════════════
    with t0:
        st.markdown("### Node overview — per-node power analysis")
        st.caption(
            "This is the correct engineering view. "
            "All power values are divided by node count so you see "
            "**one node behaving** — regardless of how many nodes ran the job. "
            "This is what you scale up to design your facility BESS."
        )

        # ── node count selector ───────────────────────────────────────────
        all_nodes = sorted(sdf["Nodes"].unique().tolist()) if "Nodes" in sdf.columns else []
        node_filter = st.radio(
            "Filter by node count",
            options=["All"] + [str(n) for n in all_nodes],
            horizontal=True,
        )

        if node_filter == "All":
            sdf_f = sdf.copy()
            data_f = data.copy()
        else:
            nn_sel = int(node_filter)
            sdf_f  = sdf[sdf["Nodes"] == nn_sel].copy()
            data_f = data[data["group"] == node_filter].copy() if node_filter in data["group"].values else data[data["group"].astype(str) == node_filter].copy()

        if sdf_f.empty:
            st.warning("No data for selected node count.")
        else:
            # ── KPI row — per node ────────────────────────────────────────
            pk_node  = sdf_f["Peak kW/node"].max()
            mn_node  = sdf_f["Mean kW/node"].mean()
            idl_node = sdf_f["Idle kW/node"].mean()
            lf_node  = sdf_f["LF %"].mean()
            r95_node = sdf_f["R95 kW/s/node"].quantile(0.95)
            en_node  = sdf_f["kWh/node"].mean()

            nk1,nk2,nk3,nk4,nk5,nk6 = st.columns(6)
            for c,l,v in zip([nk1,nk2,nk3,nk4,nk5,nk6],
                ["Peak/node","Mean/node","Idle/node","Load factor","Ramp 95th/node","Energy/node"],
                [f"{pk_node:.2f} kW", f"{mn_node:.2f} kW", f"{idl_node:.2f} kW",
                 f"{lf_node:.1f} %", f"{r95_node:.4f} kW/s", f"{en_node:.3f} kWh"]):
                c.metric(l, v)

            st.info(
                f"**Scaling to full facility (156 nodes × PUE 1.2):**  "
                f"Peak = {pk_node * 156 * 1.2:.0f} kW  |  "
                f"BESS power = {(pk_node - idl_node) * 0.80 * 156 * 1.2:.0f} kW  |  "
                f"Ramp rate = {r95_node * 156 * 1.2:.1f} kW/s"
            )
            st.divider()

            # ── per-node power overlay — all runs ─────────────────────────
            st.markdown("**Per-node power profiles — all runs overlaid**")
            st.caption(
                "Each line = one training run divided by its node count. "
                "If all lines overlap, one node always behaves the same — "
                "you only need to design BESS for ONE node and multiply."
            )
            figN = go.Figure()
            node_counts = sdf_f["Nodes"].unique() if "Nodes" in sdf_f.columns else [1]
            for nidx, nc in enumerate(sorted(node_counts)):
                col_nc = clr(nidx)
                fids   = sdf_f[sdf_f["Nodes"]==nc]["File #"].tolist() if "Nodes" in sdf_f.columns else sdf_f["File #"].tolist()
                shown  = False
                for fid in fids:
                    run = data[data["file_num"]==fid].sort_values("time_s")
                    if run.empty: continue
                    nn_run = nc
                    rs  = _ds(run)
                    pw_per_node = rs["power_W"] / max(nn_run, 1)
                    figN.add_trace(go.Scatter(
                        x=rs["time_s"], y=pw_per_node/1000,
                        mode="lines",
                        name=f"{nc} nodes",
                        legendgroup=str(nc),
                        showlegend=not shown,
                        line=dict(color=col_nc, width=1.0),
                        opacity=0.50,
                        hovertemplate=f"{nc}-node run | File {fid}<br>"
                                      f"t=%{{x:.1f}} min | %{{y:.2f}} kW/node<extra></extra>",
                    ))
                    shown = True
            figN.add_hline(y=pk_node,  line_dash="dash", line_color="#ef4444",
                           annotation_text=f"Peak {pk_node:.2f} kW/node",
                           annotation_position="top right")
            figN.add_hline(y=idl_node, line_dash="dot",  line_color="#6b7280",
                           annotation_text=f"Idle {idl_node:.2f} kW/node",
                           annotation_position="bottom right")
            figN.update_layout(
                xaxis_title="Time (s)", yaxis_title="Power per node (kW)",
                height=380, hovermode="x unified",
                legend=dict(title="Node count", orientation="h", y=1.1, font=dict(size=10)),
                margin=dict(l=55, r=150, t=20, b=45),
            )
            st.plotly_chart(figN, use_container_width=True)

            # ── per-node comparison table ─────────────────────────────────
            nd1, nd2 = st.columns(2)
            with nd1:
                st.markdown("**Per-node stats grouped by node count**")
                if "Nodes" in sdf_f.columns:
                    grp_tbl = sdf_f.groupby("Nodes").agg(
                        Runs         = ("File #",         "count"),
                        Peak_kW_node = ("Peak kW/node",   "mean"),
                        Mean_kW_node = ("Mean kW/node",   "mean"),
                        Idle_kW_node = ("Idle kW/node",   "mean"),
                        LF_pct       = ("LF %",           "mean"),
                        R95_kWs_node = ("R95 kW/s/node",  "mean"),
                        kWh_node     = ("kWh/node",       "mean"),
                    ).round(3).reset_index()
                    grp_tbl.columns = ["Nodes","Runs","Peak kW/n","Mean kW/n",
                                       "Idle kW/n","LF %","R95 kW/s/n","kWh/n"]
                    st.dataframe(grp_tbl, use_container_width=True, hide_index=True)
                    st.caption(
                        "Key check: if Peak kW/node is the same across 2, 4, 8, 16 nodes "
                        "→ one node behaviour is universal → scale linearly to 156 nodes."
                    )
            with nd2:
                st.markdown("**Peak kW per node — by node count**")
                if "Nodes" in sdf_f.columns:
                    figCmp = go.Figure()
                    for nidx, row in grp_tbl.iterrows():
                        figCmp.add_trace(go.Bar(
                            x=[f"{int(row['Nodes'])} nodes"],
                            y=[row["Peak kW/n"]],
                            name=f"{int(row['Nodes'])} nodes",
                            marker_color=clr(nidx),
                            text=[f"{row['Peak kW/n']:.2f}"],
                            textposition="outside",
                        ))
                    figCmp.add_hline(y=3.52, line_dash="dash", line_color="#ef4444",
                                     annotation_text="H100 theoretical max 3.52 kW",
                                     annotation_position="top right")
                    figCmp.update_layout(
                        title="Is per-node power the same regardless of node count?",
                        xaxis_title="Node count", yaxis_title="Peak kW per node",
                        yaxis=dict(range=[0, 4.2]),
                        showlegend=False, height=300,
                        margin=dict(l=55, r=130, t=55, b=45),
                    )
                    st.plotly_chart(figCmp, use_container_width=True)

            # ── load duration curve per-node ───────────────────────────────
            st.divider()
            st.markdown("**Load duration curve — per node**")
            st.caption(
                "LDC shows % of time node power exceeded each level. "
                "If curves from 2-node and 16-node jobs overlap → same node behaviour."
            )
            figLDC = go.Figure()
            if "Nodes" in sdf_f.columns:
                for nidx, nc in enumerate(sorted(sdf_f["Nodes"].unique())):
                    fids = sdf_f[sdf_f["Nodes"]==nc]["File #"].tolist()
                    pw_all = []
                    for fid in fids:
                        run = data[data["file_num"]==fid]
                        if run.empty: continue
                        pw_all.extend((run["power_W"].values / nc).tolist())
                    if not pw_all: continue
                    pw_arr = np.sort(np.array(pw_all)/1000)[::-1]
                    pct    = np.arange(1, len(pw_arr)+1)/len(pw_arr)*100
                    skip   = max(1, len(pw_arr)//400)
                    figLDC.add_trace(go.Scatter(
                        x=pct[::skip], y=pw_arr[::skip],
                        mode="lines", name=f"{nc} nodes",
                        line=dict(color=clr(nidx), width=2),
                    ))
            figLDC.update_layout(
                xaxis_title="% time exceeded", yaxis_title="Power per node (kW)",
                height=300,
                legend=dict(title="Node count", orientation="h", y=1.12, font=dict(size=10)),
                margin=dict(l=55, r=15, t=30, b=45),
            )
            st.plotly_chart(figLDC, use_container_width=True)

            st.success(
                "**How to read these charts as a BESS engineer:**  "
                "If the LDC curves from 2-node, 4-node, 8-node and 16-node jobs all overlap "
                "→ one node always behaves the same way regardless of job size "
                "→ your BESS design is simply: (one node numbers) × 156 × PUE.  "
                "If they do NOT overlap → larger jobs push each node harder "
                "→ you must design for the worst-case node count."
            )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — ALL BATCHES
    # ════════════════════════════════════════════════════════════════════════
    with t1:
        st.markdown("### Power profiles — all batches")
        figAll=go.Figure()
        for g in groups:  
            cg=gcols.get(str(g), gcols.get(g, "#378ADD")); shown=False
            for fid,run in data[data["group"]==g].groupby("file_num"):
                rs=_ds(run.sort_values("time_s"))
                figAll.add_trace(go.Scatter(
                    x=rs["time_s"],y=rs["power_W"]/1000,mode="lines",
                    line=dict(color=cg,width=0.9),opacity=.40,
                    name=f"{glbl}={g}",legendgroup=str(g),showlegend=not shown,
                    hovertemplate=f"{glbl}={g}|t=%{{x:.1f}}m|%{{y:.2f}}kW<extra></extra>",
                )); shown=True
        figAll.update_layout(xaxis_title="Time (s)",yaxis_title="kW",height=380,
            hovermode="x unified",legend=dict(title=glbl,font=dict(size=10)),
            margin=dict(l=55,r=15,t=20,b=45))
        st.plotly_chart(figAll,use_container_width=True)

        r2a,r2b=st.columns(2)
        with r2a:
            figL=go.Figure()
            for g in groups:  
                pw=np.sort(data[data["group"]==g]["power_W"].values/1000)[::-1]
                pc=np.arange(1,len(pw)+1)/len(pw)*100
                sk=max(1,len(pw)//400)
                figL.add_trace(go.Scatter(x=pc[::sk],y=pw[::sk],mode="lines",
                    name=f"{glbl}={g}",line=dict(color=gcols.get(str(g), gcols.get(g, "#378ADD")),width=1.8)))
            figL.update_layout(title="Load duration curve",
                xaxis_title="% exceeded",yaxis_title="kW",
                height=300,margin=dict(l=55,r=15,t=50,b=45))
            st.plotly_chart(figL,use_container_width=True)
        with r2b:
            figR=go.Figure()
            for g in groups:  
                rm=(data[data["group"]==g].groupby("file_num")["power_W"]
                    .diff().abs().div(dt).dropna()/1000)
                rm=rm[rm<=rm.quantile(.999)]
                figR.add_trace(go.Histogram(x=rm,name=f"{glbl}={g}",
                    marker_color=gcols.get(str(g), gcols.get(g, "#378ADD")),opacity=.60,nbinsx=50))
            figR.add_vline(x=r95/1000,line_dash="dash",line_color="red",
                annotation_text=f"95th {r95/1000:.3f}",annotation_position="top right")
            figR.update_layout(title="Ramp rate distribution",
                xaxis_title="kW/s",yaxis_title="Count",barmode="overlay",
                height=300,margin=dict(l=55,r=15,t=50,b=45))
            st.plotly_chart(figR,use_container_width=True)

        st.markdown(f"### By {glbl}")
        b1,b2,b3=st.columns(3)
        with b1:
            figBx=go.Figure()
            for g in groups:  
                figBx.add_trace(go.Box(y=data[data["group"]==g]["power_W"]/1000,
                    name=str(g),marker_color=gcols.get(str(g), gcols.get(g, "#378ADD")),boxmean=True,showlegend=False))
            figBx.update_layout(title=f"Distribution by {glbl}",
                yaxis_title="kW",height=280,margin=dict(l=55,r=15,t=50,b=45))
            st.plotly_chart(figBx,use_container_width=True)
        with b2:
            agg=sdf.groupby("Nodes")[["Peak kW/node","Mean kW/node","Idle kW/node"]].mean().reset_index()
            figBr=go.Figure()
            for m in ["Peak kW/node","Mean kW/node","Idle kW/node"]:
                figBr.add_trace(go.Bar(x=agg["Nodes"].astype(str),y=agg[m].round(2),name=m))
            figBr.update_layout(title="Peak/mean/idle",barmode="group",
                yaxis_title="kW",height=280,margin=dict(l=55,r=15,t=50,b=45))
            st.plotly_chart(figBr,use_container_width=True)
        with b3:
            lfa=sdf.groupby("Nodes")["LF %"].agg(["mean","std"]).reset_index()
            figLF=go.Figure()
            figLF.add_trace(go.Bar(x=lfa["Nodes"].astype(str),y=lfa["mean"].round(1),
                error_y=dict(type="data",array=lfa["std"].round(1)),
                marker_color=[gcols.get(str(g), gcols.get(g, "#378ADD")) for g in lfa["Nodes"]],
                text=lfa["mean"].round(1).astype(str)+"%",textposition="outside"))
            figLF.update_layout(title="Load factor",yaxis_title="LF %",
                yaxis=dict(range=[0,115]),height=280,margin=dict(l=55,r=15,t=50,b=45))
            st.plotly_chart(figLF,use_container_width=True)

        st.markdown("### Statistics table")
        st.dataframe(sdf.sort_values(["Nodes","File #"]),
                     use_container_width=True,height=350)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — SINGLE BATCH
    # ════════════════════════════════════════════════════════════════════════
    with t2:
        st.markdown("### 🔬 Single batch drill-down")
        s1,s2,s3=st.columns([1,1,2])
        with s1:
            sg=st.selectbox(f"Filter {glbl}",["All"]+[str(g) for g in groups])
        av=(data[["file_num","group"]].drop_duplicates() if sg=="All"
            else data[data["group"]==sg][["file_num","group"]].drop_duplicates())
        fns=sorted(av["file_num"].unique().tolist())
        with s2:
            sf=st.selectbox("Batch",fns,format_func=lambda x:f"File {x:06d}")
        with s3:
            ri2=sdf[sdf["File #"]==sf]
            if not ri2.empty:
                r=ri2.iloc[0]
                st.info(f"**{glbl}**={str(int(r['Nodes'])) if 'Nodes' in r.index else r.get('Group','?')} | Peak={r.get('Peak kW/node', r.get('Peak kW','?'))} kW | "
                        f"Mean={r.get('Mean kW/node', r.get('Mean kW','?'))} kW | Dur={r.get('Dur s','?')} s")
        st.divider()
        sr=data[data["file_num"]==sf].sort_values("time_s").reset_index(drop=True)
        sgv=sr["group"].iloc[0] if len(sr)>0 else "?"
        mr=None
        try:
            ic=wdef.get("id_col")
            mm=(meta[meta[ic].astype(int)==sf] if ic and ic in meta.columns
                else meta.iloc[[sf]] if sf<len(meta) else None)
            if mm is not None and len(mm)>0: mr=mm.iloc[0]
        except Exception: mr=None
        if len(sr)==0: st.warning("No data.")
        else: _batch_panel(sr,sf,sgv,glbl,dt,mr)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — NODE DEEP-DIVE
    # ════════════════════════════════════════════════════════════════════════
    with t3:
        st.markdown("### 🔍 Node deep-dive — single node behaviour")
        st.caption("Power ÷ node count = per-node signal. Select any run below.")
        n1,n2,n3=st.columns([1,1,2])
        with n1:
            ng=st.selectbox(f"Filter {glbl}",["All"]+[str(g) for g in groups],key="ndg")
        nav=(data[["file_num","group"]].drop_duplicates() if ng=="All"
             else data[data["group"]==ng][["file_num","group"]].drop_duplicates())
        nfns=sorted(nav["file_num"].unique().tolist())
        with n2:
            nf=st.selectbox("Batch",nfns,
                format_func=lambda x:(
                    f"File {x:06d} [{glbl}={data[data['file_num']==x]['group'].iloc[0]}]"
                    if len(data[data['file_num']==x])>0 else f"File {x:06d}"),
                key="ndf")
        with n3:
            nr=sdf[sdf["File #"]==nf]
            if not nr.empty:
                r=nr.iloc[0]
                st.info(f"**{glbl}**={str(int(r['Nodes'])) if 'Nodes' in r.index else r.get('Group','?')} | Peak={r.get('Peak kW/node', r.get('Peak kW','?'))} kW | "
                        f"Energy={r['kWh']} kWh | Dur={r.get('Dur s','?')} s")
        st.divider()
        nd=data[data["file_num"]==nf].sort_values("time_s").reset_index(drop=True)
        if len(nd)==0:
            st.warning("No data.")
        else:
            nn=1
            if "nodes" in nd.columns:
                try: nn=int(nd["nodes"].iloc[0])
                except: pass
            else:
                try: nn=int(float(nd["group"].iloc[0]))
                except: pass
            mdl=nd["model"].iloc[0] if "model" in nd.columns else "unknown"
            st.success(f"✅  File {nf:06d} | **{nn} nodes** | {mdl} | {len(nd):,} samples")
            _node_panel(nd,nn,nf,mdl,dt)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 — BESS SIZING
    # ════════════════════════════════════════════════════════════════════════
    with t4:
        st.markdown("### 🔋 Facility BESS sizing & renewable integration")
        jhs=[2,5,8,12,16,20]
        fig24=go.Figure()
        fig24.add_trace(go.Scatter(x=_ds(t24),y=_ds(L24)/1000,fill="tozeroy",
            fillcolor="rgba(226,75,74,.15)",line=dict(color="#E24B4A",width=1.5),
            name="Load (MW)"))
        fig24.add_trace(go.Scatter(x=_ds(t24),y=_ds(sol)/1000,fill="tozeroy",
            fillcolor="rgba(29,158,117,.20)",line=dict(color="#1D9E75",width=1.5),
            name=f"Solar {sol_f}%"))
        fig24.add_trace(go.Scatter(x=_ds(t24),y=_ds(np.maximum(net,0))/1000,
            line=dict(color="#534AB7",width=1.5,dash="dot"),name="Net demand"))
        fig24.add_trace(go.Scatter(x=_ds(t24),y=_ds(np.minimum(net,0))/1000,
            fill="tozeroy",fillcolor="rgba(83,74,183,.10)",
            line=dict(color="#9399b2",width=0.8),name="RE surplus"))
        for sh in jhs:
            fig24.add_vline(x=sh,line_dash="dot",line_color="gray",
                            line_width=0.8,opacity=.5)
        fig24.update_layout(
            title=f"RE {re_pct:.1f}% | Gap {def_kWh:.0f} kWh | Surplus {sur_kWh:.0f} kWh",
            xaxis_title="Hour",yaxis_title="MW",height=340,
            hovermode="x unified",xaxis=dict(range=[0,24],dtick=2),
            margin=dict(l=55,r=15,t=55,b=45))
        st.plotly_chart(fig24,use_container_width=True)

        sc1,sc2=st.columns([2,1])
        with sc1:
            figS=go.Figure()
            figS.add_hrect(y0=soc_lo,y1=soc_hi,
                           fillcolor="rgba(166,227,161,.07)",line_width=0)
            figS.add_trace(go.Scatter(x=_ds(t24),y=_ds(soc),fill="tozeroy",
                fillcolor="rgba(186,117,23,.20)",line=dict(color="#BA7517",width=2),
                name="SoC %",
                hovertemplate="Hour %{x:.1f} | SoC %{y:.1f}%<extra></extra>"))
            figS.add_hline(y=soc_lo,line_dash="dash",line_color="#f38ba8",
                           annotation_text=f"Min {soc_lo}%",
                           annotation_position="bottom right")
            figS.add_hline(y=soc_hi,line_dash="dash",line_color="#a6e3a1",
                           annotation_text=f"Max {soc_hi}%",
                           annotation_position="top right")
            for sh in jhs:
                figS.add_vline(x=sh,line_dash="dot",line_color="gray",
                               line_width=0.8,opacity=.5)
            figS.update_layout(
                title=f"BESS SoC — {be_MWh:.2f} MWh | {bp_MW:.2f} MW | RTE {rte:.0%}",
                xaxis_title="Hour",yaxis_title="SoC %",
                yaxis=dict(range=[0,105]),xaxis=dict(range=[0,24],dtick=2),
                height=320,hovermode="x unified",margin=dict(l=55,r=15,t=55,b=45))
            st.plotly_chart(figS,use_container_width=True)
        with sc2:
            st.markdown("**BESS specification**")
            for k,v in [
                ("Power rating",   f"{bp_MW:.2f} MW"),
                ("Energy capacity",f"{be_MWh:.2f} MWh"),
                ("C-rate",         f"{cr:.2f} C"),
                ("Chemistry",      chem),
                ("SoC window",     f"{soc_lo}–{soc_hi} %"),
                ("RTE",            f"{rte:.0%}"),
                ("Ramp 95th",      f"{r95/1000:.3f} kW/s"),
                (None,None),
                ("Facility nodes", str(fac_n)),
                ("PUE",            str(pue)),
                ("Peak demand",    f"{f_pk:.0f} kW"),
                ("Mean demand",    f"{f_mn:.0f} kW"),
                (None,None),
                ("Solar capacity", f"{f_pk*sol_f/100:.0f} kW"),
                ("RE coverage",    f"{re_pct:.1f} %"),
                ("Firming gap",    f"{def_kWh:.0f} kWh/day"),
            ]:
                if k is None:
                    st.markdown("<hr style='margin:4px 0;border:none;"
                                "border-top:1px solid #333'>",unsafe_allow_html=True)
                else:
                    ca,cb=st.columns([3,2])
                    ca.markdown(f"<span style='color:gray;font-size:12px'>{k}</span>",
                                unsafe_allow_html=True)
                    cb.markdown(f"**{v}**")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 5 — COMPARISON
    # ════════════════════════════════════════════════════════════════════════
    with t5:
        st.markdown("### ⚖️  Batch comparison")
        afns=sorted(data["file_num"].unique().tolist())
        sel=st.multiselect("Select batches (2–5)",options=afns,
            default=afns[:min(3,len(afns))],
            format_func=lambda x:(
                f"File {x:06d}|{glbl}={data[data['file_num']==x]['group'].iloc[0]}"
                f"|Pk={sdf[sdf['File #']==x]['Peak kW/node'].values[0] if len(sdf[sdf['File #']==x])>0 else '?'}kW"
            ))
        if len(sel)<2:
            st.info("Select ≥ 2 batches.")
        else:
            figC=go.Figure()
            for i,fn in enumerate(sel):
                run=data[data["file_num"]==fn]
                grp=run["group"].iloc[0]
                rs=_ds(run.sort_values("time_s"))
                figC.add_trace(go.Scatter(x=rs["time_s"],y=rs["power_W"]/1000,
                    mode="lines",name=f"File {fn}|{glbl}={grp}",
                    line=dict(color=clr(i),width=1.5),
                    hovertemplate=f"File {fn}<br>t=%{{x:.1f}}m|%{{y:.2f}}kW<extra></extra>"))
            figC.update_layout(title="Overlay",xaxis_title="Time (s)",yaxis_title="kW",
                height=340,hovermode="x unified",margin=dict(l=55,r=15,t=50,b=45))
            st.plotly_chart(figC,use_container_width=True)

            e1,e2=st.columns(2)
            with e1:
                figL2=go.Figure()
                for i,fn in enumerate(sel):
                    run=data[data["file_num"]==fn]; grp=run["group"].iloc[0]
                    pw=np.sort(run["power_W"].values/1000)[::-1]
                    pc=np.arange(1,len(pw)+1)/len(pw)*100
                    sk=max(1,len(pw)//400)
                    figL2.add_trace(go.Scatter(x=pc[::sk],y=pw[::sk],mode="lines",
                        name=f"File {fn}|{grp}",line=dict(color=clr(i),width=2)))
                figL2.update_layout(title="LDC",xaxis_title="% exceeded",
                    yaxis_title="kW",height=300,margin=dict(l=55,r=15,t=50,b=45))
                st.plotly_chart(figL2,use_container_width=True)
            with e2:
                cs=sdf[sdf["File #"].isin(sel)].copy()
                mets=["Peak kW/node","Mean kW/node","LF %","R95 kW/s/node"]
                figM=go.Figure()
                for i,(_,row) in enumerate(cs.iterrows()):
                    fn=int(row["File #"])
                    figM.add_trace(go.Bar(name=f"File {fn}|{row['Group']}",
                        x=mets,y=[row[m] for m in mets],marker_color=clr(sel.index(fn))))
                figM.update_layout(title="Key metrics",barmode="group",
                    yaxis_title="Value",height=300,margin=dict(l=55,r=15,t=50,b=45))
                st.plotly_chart(figM,use_container_width=True)
            st.dataframe(cs.set_index("File #"),use_container_width=True)

    # ── footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "Dataset: Vercellino et al. (2026) — NLR/OT-2C00-99122 — DOI 10.7799/3025227  |  "
        "Solar: synthetic — replace with NSRDB for final design  |  "
        "BESS sizing is indicative — full power system study required"
    )
