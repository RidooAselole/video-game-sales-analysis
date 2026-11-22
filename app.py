"""
app.py -- Dashboard Video Game Sales (versi tervalidasi)

Sumber angka:
  * Data penjualan  : MySQL (VGS_DB_URL) -> jika gagal, file dump SQL (VGS_SQL). Tidak ada angka bawaan.
  * Model & segmen  : hasil pipeline di folder `hasil/` (VGS_RESULTS), dibuat oleh
                      `python rq_pipeline.py --sql video_game_sales.sql --out hasil`
                      atau dengan menjalankan notebook video_game_sales_final.ipynb.
Letakkan app.py, rq_pipeline.py, dan folder hasil/ di folder yang sama, lalu: streamlit run app.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rq_pipeline as rq  # loader + koreksi 2 penunjuk genre (idempoten)

DB_URL = os.environ.get("VGS_DB_URL", "mysql+pymysql://root:@localhost:3306/video_game_sales")
SQL_FILE = os.environ.get("VGS_SQL", os.path.join(HERE, "video_game_sales.sql"))
RESULTS_DIR = os.environ.get("VGS_RESULTS", os.path.join(HERE, "hasil"))

st.set_page_config(page_title="Video Game Sales Analytics", layout="wide", page_icon="🎮",
                   initial_sidebar_state="collapsed")

# ─────────────────────────────────────────────────────────────────────────────
# TAMPILAN
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background:#0f0f1a; color:#e8e8f0; }
[data-testid="stHeader"] { background:transparent; }
.hero { background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);
        border:1px solid #e94560; border-radius:16px; padding:1.6rem 2.2rem; margin-bottom:1.2rem; }
.hero h1 { font-size:2.1rem; font-weight:800; color:#fff; margin:0; }
.hero p  { color:#a0a8c0; margin:.4rem 0 0; font-size:1rem; }
.kpi-row { display:flex; gap:1rem; flex-wrap:wrap; margin-bottom:1.2rem; }
.kpi-card { flex:1; min-width:150px; background:linear-gradient(145deg,#1e1e30,#252540);
            border:1px solid #2e2e50; border-radius:12px; padding:1rem 1.2rem; text-align:center; }
.kpi-card .val { font-size:1.7rem; font-weight:700; color:#e94560; }
.kpi-card .lbl { font-size:.78rem; color:#8888aa; margin-top:.2rem; }
.card { background:#1a1a2e; border:1px solid #2a2a45; border-radius:14px; padding:1.2rem; margin-bottom:1rem; }
.card h3 { margin-top:0; color:#e94560; font-size:1.1rem; }
.note { background:#12122a; border-left:4px solid #4fc3f7; border-radius:8px; padding:.7rem 1rem;
        color:#a0a8c0; font-size:.9rem; margin:.5rem 0; }
.warn { background:#2d2010; border-left:4px solid #f5a623; border-radius:8px; padding:.7rem 1rem;
        color:#e8d3a8; font-size:.9rem; margin:.5rem 0; }
[data-testid="stTabs"] button { font-size:1rem !important; font-weight:600 !important; }
</style>
""", unsafe_allow_html=True)

CLR_PRIMARY, CLR_ACCENT, CLR_SUCCESS, CLR_WARNING = "#e94560", "#4fc3f7", "#4caf50", "#f5a623"
CLR_BG, CLR_CARD, CLR_GRID = "#0f0f1a", "#1a1a2e", "#2a2a45"
REGION_COLORS = {"NA": CLR_PRIMARY, "EU": CLR_ACCENT, "JP": CLR_WARNING}
SEG_COLORS = [CLR_WARNING, CLR_ACCENT, CLR_SUCCESS, CLR_PRIMARY, "#7b61ff", "#66bb6a", "#ff8a65", "#90a4ae"]
PLOTLY_LAYOUT = dict(paper_bgcolor=CLR_CARD, plot_bgcolor=CLR_CARD,
                     font=dict(color="#e8e8f0", family="Inter, sans-serif"),
                     margin=dict(l=16, r=16, t=40, b=16))
AXIS = dict(gridcolor=CLR_GRID, zerolinecolor=CLR_GRID)
REG = ["NA_Sales", "EU_Sales", "JP_Sales"]


def card_open(title):
    st.markdown(f'<div class="card"><h3>{title}</h3>', unsafe_allow_html=True)


def card_close():
    st.markdown("</div>", unsafe_allow_html=True)


def chart(fig, key):
    st.plotly_chart(fig, use_container_width=True, key=key)


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=600, show_spinner="Memuat data…")
def load_data():
    """MySQL dulu; jika tidak tersedia, pakai dump SQL. Mengembalikan (df, info)."""
    err_db, T, source = None, None, None
    try:
        T = rq.load_tables(mysql=DB_URL)
        source = "MySQL"
    except Exception as e:  # noqa: BLE001
        err_db = str(e).splitlines()[0][:160]
    if T is None and os.path.exists(SQL_FILE):
        T = rq.load_tables(sql=SQL_FILE)
        source = f"dump SQL ({os.path.basename(SQL_FILE)})"
    if T is None:
        return None, {"error": f"MySQL gagal ({err_db}) dan file dump tidak ditemukan di {SQL_FILE}"}
    df, nfix = rq.build_flat(T)
    info = {
        "source": source, "db_error": err_db, "pointer_fixes": int(nfix),
        "n_game_rows": int(len(T["game"])), "n_publisher_rows": int(len(T["publisher"])),
        "year_null": int(df["Year"].isna().sum()),
        "pub_missing": int(df["Publisher"].isin(["N/A", "Unknown"]).sum()),
        "dup_name_platform": int(df.duplicated(["Name", "Platform"]).sum()),
    }
    return df, info


@st.cache_data(ttl=600, show_spinner="Membaca hasil pipeline…")
def load_results():
    """Baca hasil/ dari rq_pipeline.py. Mengembalikan dict atau None bila belum ada."""
    p = os.path.join(RESULTS_DIR, "results.json")
    if not os.path.exists(p):
        return None
    R = json.load(open(p, encoding="utf8"))
    prof = os.path.join(RESULTS_DIR, "rq3_profile.csv")
    ksel = os.path.join(RESULTS_DIR, "rq3_k_selection.csv")
    asg = os.path.join(RESULTS_DIR, "rq3_assignments.csv")
    R["_profile"] = pd.read_csv(prof) if os.path.exists(prof) else None
    R["_ksel"] = pd.read_csv(ksel) if os.path.exists(ksel) else None
    R["_assign"] = pd.read_csv(asg) if os.path.exists(asg) else None
    return R


df, info = load_data()
if df is None:
    st.error(info["error"])
    st.stop()
R = load_results()

reg_totals = df[REG + ["Other_Sales"]].sum()
total_sales = float(reg_totals.sum())

# ─────────────────────────────────────────────────────────────────────────────
# HEADER + KPI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero"><h1>🎮 Video Game Sales Analytics</h1>
<p>Perbandingan pasar regional · prediksi penjualan dari atribut produk · segmentasi K-Means
&nbsp;|&nbsp; sumber data: {info['source']}</p></div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="kpi-row">
  <div class="kpi-card"><div class="val">{total_sales:,.0f}M</div><div class="lbl">Total Penjualan Global</div></div>
  <div class="kpi-card"><div class="val">{len(df):,}</div><div class="lbl">Record (judul × platform)</div></div>
  <div class="kpi-card"><div class="val">{df['Name'].nunique():,}</div><div class="lbl">Judul Unik</div></div>
  <div class="kpi-card"><div class="val">{df['Platform'].nunique()}</div><div class="lbl">Platform</div></div>
  <div class="kpi-card"><div class="val">{df['Genre'].nunique()}</div><div class="lbl">Genre</div></div>
  <div class="kpi-card"><div class="val">{df['Publisher'].nunique():,}</div><div class="lbl">Publisher</div></div>
</div>
""", unsafe_allow_html=True)

if R is None:
    st.markdown(f'<div class="warn">⚠️ Hasil pipeline belum ditemukan di <code>{RESULTS_DIR}</code>. '
                'Tab RQ2 dan RQ3 memerlukan: <code>python rq_pipeline.py --sql video_game_sales.sql --out hasil</code> '
                '(≈4 menit) atau jalankan notebook <code>video_game_sales_final.ipynb</code>.</div>',
                unsafe_allow_html=True)
elif R.get("n_records") != len(df):
    st.markdown(f'<div class="warn">⚠️ Hasil pipeline dibuat dari {R.get("n_records"):,} record, sedangkan data sekarang '
                f'{len(df):,}. Jalankan ulang pipeline agar RQ2/RQ3 sinkron.</div>', unsafe_allow_html=True)

tab_ov, tab1, tab2, tab3, tab4 = st.tabs(["📊 Ringkasan", "🌍 RQ1 · Pasar Regional", "🔮 RQ2 · Prediksi",
                                          "🎯 RQ3 · Segmentasi", "✅ Validasi Data"])

# ─────────────────────────────────────────────────────────────────────────────
# RINGKASAN
# ─────────────────────────────────────────────────────────────────────────────
with tab_ov:
    c1, c2 = st.columns(2, gap="large")
    with c1:
        card_open("🌍 Pangsa Pasar Regional")
        fig = go.Figure(go.Pie(labels=["North America", "Europe", "Japan", "Other"], values=reg_totals.values,
                               hole=0.5, marker=dict(colors=[CLR_PRIMARY, CLR_ACCENT, CLR_WARNING, CLR_SUCCESS],
                                                     line=dict(color=CLR_BG, width=3)),
                               textinfo="label+percent"))
        fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False)
        chart(fig, "ov_region")
        st.caption("Persentase dihitung dari total empat wilayah (termasuk Other).")
        card_close()
    with c2:
        card_open("🎭 Total Penjualan per Genre")
        g = df.groupby("Genre")["Global_Sales"].sum().sort_values()
        fig = go.Figure(go.Bar(x=g.values, y=g.index, orientation="h",
                               marker=dict(color=g.values, colorscale=[[0, "#1a1a3e"], [0.5, CLR_ACCENT], [1, CLR_PRIMARY]]),
                               text=[f"{v:,.0f}M" for v in g.values], textposition="outside"))
        fig.update_layout(**PLOTLY_LAYOUT, height=320, xaxis=dict(title="Juta kopi", **AXIS), yaxis=dict(**AXIS))
        chart(fig, "ov_genre")
        card_close()

    c3, c4 = st.columns(2, gap="large")
    with c3:
        card_open("📅 Tren Rilis per Tahun")
        y = df.dropna(subset=["Year"]).groupby(df["Year"].astype("Int64"))["Name"].count()
        fig = go.Figure(go.Scatter(x=y.index.astype(int), y=y.values, mode="lines", fill="tozeroy",
                                   line=dict(color=CLR_PRIMARY, width=2.5), fillcolor="rgba(233,69,96,0.12)"))
        pk = y.idxmax()
        fig.add_trace(go.Scatter(x=[int(pk)], y=[int(y.max())], mode="markers+text", showlegend=False,
                                 marker=dict(color=CLR_WARNING, size=10), text=[f"  Puncak: {int(y.max()):,} ({int(pk)})"],
                                 textposition="top right"))
        fig.update_layout(**PLOTLY_LAYOUT, height=320, xaxis=dict(title="Tahun", **AXIS),
                          yaxis=dict(title="Jumlah record", **AXIS), showlegend=False)
        chart(fig, "ov_year")
        st.caption(f"{info['year_null']:,} record tanpa tahun rilis tidak ikut dihitung di grafik ini.")
        card_close()
    with c4:
        card_open("🕹️ Penjualan per Platform (Top 12)")
        p = df.groupby("Platform")["Global_Sales"].sum().sort_values(ascending=False).head(12)
        fig = go.Figure(go.Bar(x=p.index, y=p.values, marker=dict(color=p.values, colorscale=[[0, "#1a1a3e"], [1, CLR_ACCENT]]),
                               text=[f"{v:,.0f}M" for v in p.values], textposition="outside"))
        fig.update_layout(**PLOTLY_LAYOUT, height=320, xaxis=dict(title="Platform", **AXIS),
                          yaxis=dict(title="Juta kopi", **AXIS))
        chart(fig, "ov_platform")
        card_close()

    card_open("🏆 Top 10 Publisher")
    pb = df.groupby("Publisher")["Global_Sales"].sum().sort_values().tail(10)
    fig = go.Figure(go.Bar(x=pb.values, y=pb.index, orientation="h",
                           marker=dict(color=pb.values, colorscale=[[0, "#1a2a1a"], [1, CLR_SUCCESS]]),
                           text=[f"{v:,.0f}M" for v in pb.values], textposition="outside"))
    fig.update_layout(**PLOTLY_LAYOUT, height=340, xaxis=dict(title="Juta kopi", **AXIS), yaxis=dict(**AXIS))
    chart(fig, "ov_pub")
    card_close()

# ─────────────────────────────────────────────────────────────────────────────
# RQ1
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    sh4 = reg_totals / reg_totals.sum() * 100
    sh3 = df[REG].sum() / df[REG].sum().sum() * 100
    st.markdown(f'<div class="note"><b>RQ1.</b> Bagaimana pola penjualan menurut genre dan platform berbeda antara '
                f'Amerika Utara, Eropa, dan Jepang? &nbsp;·&nbsp; Pangsa dari 4 wilayah: NA {sh4["NA_Sales"]:.1f}%, '
                f'EU {sh4["EU_Sales"]:.1f}%, JP {sh4["JP_Sales"]:.1f}%, Other {sh4["Other_Sales"]:.1f}%. '
                f'Jika hanya NA+EU+JP dihitung: {sh3["NA_Sales"]:.1f}% / {sh3["EU_Sales"]:.1f}% / {sh3["JP_Sales"]:.1f}%.</div>',
                unsafe_allow_html=True)

    def share_by_region(dim, top=None):
        g = df.groupby(dim)[REG].sum()
        s = g / g.sum() * 100
        s.columns = ["NA", "EU", "JP"]
        if top:
            s = s.loc[s.max(axis=1).sort_values(ascending=False).head(top).index]
        return s

    a, b = st.columns(2, gap="large")
    with a:
        card_open("🎭 Komposisi Genre per Wilayah (% dari penjualan wilayah)")
        s = share_by_region("Genre")
        s = s.loc[s.mean(axis=1).sort_values(ascending=False).index]
        fig = go.Figure([go.Bar(name=r, x=s.index, y=s[r], marker_color=REGION_COLORS[r]) for r in s.columns])
        fig.update_layout(**PLOTLY_LAYOUT, barmode="group", height=360, yaxis=dict(title="% penjualan wilayah", **AXIS),
                          xaxis=dict(tickangle=-30, **AXIS), legend=dict(orientation="h", y=1.12))
        chart(fig, "rq1_genre")
        card_close()
    with b:
        card_open("🕹️ Komposisi Platform per Wilayah (10 platform terbesar)")
        s = share_by_region("Platform", top=10)
        fig = go.Figure([go.Bar(name=r, x=s.index, y=s[r], marker_color=REGION_COLORS[r]) for r in s.columns])
        fig.update_layout(**PLOTLY_LAYOUT, barmode="group", height=360, yaxis=dict(title="% penjualan wilayah", **AXIS),
                          xaxis=dict(**AXIS), legend=dict(orientation="h", y=1.12))
        chart(fig, "rq1_platform")
        card_close()

    a, b = st.columns([0.9, 1.1], gap="large")
    with a:
        card_open("🔗 Korelasi Spearman antar Wilayah")
        cm = df[REG].corr(method="spearman").round(3)
        lab = ["NA", "EU", "JP"]
        fig = go.Figure(go.Heatmap(z=cm.values, x=lab, y=lab, zmin=-1, zmax=1, colorscale="RdBu", text=cm.values,
                                   texttemplate="%{text}"))
        fig.update_layout(**PLOTLY_LAYOUT, height=320)
        chart(fig, "rq1_corr")
        st.caption("Per record judul-platform. Korelasi JP dengan NA/EU negatif: judul yang laris di Barat cenderung "
                   "tidak laris di Jepang, dan sebaliknya.")
        card_close()
    with b:
        card_open("📐 Seberapa Berbeda Komposisi Genre antar Wilayah?")
        if R:
            js = R["rq1"]["by_dimension"]["Genre"]["js"]
            ci = R["rq1"]["genre_js_bootstrap_ci95"]
            jp = R["rq1"]["by_dimension"]["Platform"]["js"]
            t = pd.DataFrame({
                "Pasangan": ["NA – EU", "NA – JP", "EU – JP"],
                "Jarak genre (JS)": [js["NA-EU"], js["NA-JP"], js["EU-JP"]],
                "CI95% bootstrap": [f"{ci['NA-EU'][0]:.3f} – {ci['NA-EU'][1]:.3f}", f"{ci['NA-JP'][0]:.3f} – {ci['NA-JP'][1]:.3f}", "—"],
                "Jarak platform (JS)": [jp["NA-EU"], jp["NA-JP"], jp["EU-JP"]],
            })
            st.dataframe(t, hide_index=True, use_container_width=True)
            top = R["rq1"]["by_dimension"]["Genre"]["top3"]
            st.markdown("**3 genre teratas per wilayah**")
            for r_ in ["NA", "EU", "JP"]:
                st.markdown(f"- **{r_}**: " + "; ".join(top[r_]))
            overlap = ci["NA-EU"][1] >= ci["NA-JP"][0]
            st.caption("Jarak Jensen–Shannon: 0 = komposisi identik, 1 = berbeda total. "
                       + ("CI NA–EU dan NA–JP tumpang-tindih." if overlap else
                          "CI NA–EU dan NA–JP tidak tumpang-tindih → perbedaan Jepang terhadap Barat nyata, bukan kebetulan sampel."))
        else:
            st.info("Jalankan pipeline untuk melihat jarak Jensen–Shannon dan interval bootstrap.")
        card_close()

# ─────────────────────────────────────────────────────────────────────────────
# RQ2
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="note"><b>RQ2.</b> Sejauh mana genre, platform, publisher, dan tahun rilis dapat memprediksi '
                'penjualan global, dan atribut mana yang paling informatif? Evaluasi: 5-fold cross-validation dibagi per '
                '<i>judul</i>; median tahun, pooling publisher, dan scaler dihitung dari data latih saja.</div>',
                unsafe_allow_html=True)
    if not R:
        st.info("Hasil RQ2 belum ada. Jalankan pipeline (lihat peringatan di atas).")
    else:
        cv = R["rq2"]["cv"]
        names = list(cv.keys())
        a, b = st.columns(2, gap="large")
        with a:
            card_open("📉 R² (lebih tinggi lebih baik)")
            fig = go.Figure(go.Bar(x=names, y=[cv[n]["R2"][0] for n in names],
                                   error_y=dict(type="data", array=[cv[n]["R2"][1] for n in names]),
                                   marker_color=[CLR_GRID if n.startswith("Baseline") else CLR_ACCENT for n in names]))
            fig.update_layout(**PLOTLY_LAYOUT, height=330, yaxis=dict(title="R²", **AXIS), xaxis=dict(tickangle=-25, **AXIS))
            chart(fig, "rq2_r2")
            card_close()
        with b:
            card_open("📏 MAE (lebih rendah lebih baik)")
            fig = go.Figure(go.Bar(x=names, y=[cv[n]["MAE"][0] for n in names],
                                   error_y=dict(type="data", array=[cv[n]["MAE"][1] for n in names]),
                                   marker_color=[CLR_GRID if n.startswith("Baseline") else CLR_WARNING for n in names]))
            fig.update_layout(**PLOTLY_LAYOUT, height=330, yaxis=dict(title="MAE (juta kopi)", **AXIS),
                              xaxis=dict(tickangle=-25, **AXIS))
            chart(fig, "rq2_mae")
            card_close()

        tbl = pd.DataFrame({
            "Model": names,
            "MAE": [f"{cv[n]['MAE'][0]:.3f} ± {cv[n]['MAE'][1]:.3f}" for n in names],
            "RMSE": [f"{cv[n]['RMSE'][0]:.3f} ± {cv[n]['RMSE'][1]:.3f}" for n in names],
            "R²": [f"{cv[n]['R2'][0]:+.3f} ± {cv[n]['R2'][1]:.3f}" for n in names],
        })
        st.dataframe(tbl, hide_index=True, use_container_width=True)
        st.caption("Rata-rata ± simpangan baku antar 5 fold; setiap judul diuji tepat satu kali.")

        cands = [n for n in names if not n.startswith("Baseline")]
        best_r2 = max(cands, key=lambda n: cv[n]["R2"][0])
        best_mae = min(cands, key=lambda n: cv[n]["MAE"][0])
        med = cv["Baseline (median)"]["MAE"][0]
        st.markdown(f'<div class="note"><b>Cara membaca.</b> R² terbaik: <b>{best_r2}</b> '
                    f'({cv[best_r2]["R2"][0]:.3f}) — artinya atribut produk hanya menjelaskan sekitar '
                    f'{cv[best_r2]["R2"][0] * 100:.0f}% variasi penjualan global. '
                    f'MAE terbaik: <b>{best_mae}</b> ({cv[best_mae]["MAE"][0]:.3f}) vs baseline median {med:.3f}. '
                    f'Model berloss kuadrat tidak mengalahkan baseline median pada MAE karena penjualan sangat miring ke kanan; '
                    f'keunggulannya ada pada RMSE/R².</div>', unsafe_allow_html=True)

        card_open("🧭 Atribut Paling Informatif (permutation importance, kelompok fitur)")
        imp = R["rq2"]["importance"]
        feats = ["Publisher", "Platform", "Genre", "Year"]
        fig = go.Figure([go.Bar(name=m, x=feats, y=[imp[m][f]["share_pct"] for f in feats],
                                marker_color=SEG_COLORS[i]) for i, m in enumerate(imp)])
        fig.update_layout(**PLOTLY_LAYOUT, barmode="group", height=330, yaxis=dict(title="% dari total pengaruh", **AXIS),
                          xaxis=dict(**AXIS), legend=dict(orientation="h", y=1.12))
        chart(fig, "rq2_imp")
        first = list(imp)[0]
        rank = sorted(feats, key=lambda f: -imp[first][f]["share_pct"])
        st.caption(f"Urutan ({first}): " + " > ".join(f"{f} {imp[first][f]['share_pct']:.0f}%" for f in rank)
                   + ". Simpangan antar fold cukup besar; baca sebagai urutan kasar, bukan angka eksak.")
        card_close()

        ld = R["rq2"]["leakage_demo"]
        lk, cl = ld["leaky_random_split_global_stats"], ld["clean_grouped_train_only"]
        st.markdown(f'<div class="note"><b>Catatan metodologi.</b> Split acak dengan statistik global memberi R² '
                   f'{lk["R2"]:.3f}, split per judul dengan statistik train saja {cl["R2"]:.3f} (model HGB, prediksi '
                   f'out-of-fold gabungan). Selisihnya kecil, jadi protokol per judul dipakai sebagai pengaman desain — '
                   f'bukan karena terbukti menaikkan atau menurunkan skor.</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# RQ3
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="note"><b>RQ3.</b> Segmen apa yang terbentuk dari pola penjualan regional (NA, EU, JP) dengan '
                'K-Means, dan bagaimana karakteristiknya berdasarkan genre, platform, serta distribusi penjualan? '
                'Fitur: log(1+penjualan) lalu distandardisasi.</div>', unsafe_allow_html=True)
    if not R or R.get("_profile") is None:
        st.info("Hasil RQ3 belum ada. Jalankan pipeline (lihat peringatan di atas).")
    else:
        r3, prof = R["rq3"], R["_profile"].copy()
        segs = list(prof["Segmen"])
        colors = {s: SEG_COLORS[i] for i, s in enumerate(segs)}
        a, b = st.columns(2, gap="large")
        with a:
            card_open("🎯 Distribusi Segmen")
            fig = go.Figure(go.Pie(labels=segs, values=prof["n"], hole=0.55,
                                   marker=dict(colors=[colors[s] for s in segs], line=dict(color=CLR_BG, width=3)),
                                   textinfo="label+percent"))
            fig.add_annotation(text=f"{int(prof['n'].sum()):,}<br>record", x=0.5, y=0.5, showarrow=False,
                               font=dict(size=16, color="#fff"))
            fig.update_layout(**PLOTLY_LAYOUT, height=330, showlegend=False)
            chart(fig, "rq3_donut")
            card_close()
        with b:
            card_open("🧩 Komposisi Regional tiap Segmen (% penjualan segmen)")
            fig = go.Figure([go.Bar(name=r, x=segs, y=prof[f"{r}_share%"], marker_color=REGION_COLORS[r])
                             for r in ["NA", "EU", "JP"]])
            fig.update_layout(**PLOTLY_LAYOUT, barmode="stack", height=330, yaxis=dict(title="%", **AXIS),
                              xaxis=dict(**AXIS), legend=dict(orientation="h", y=1.12))
            chart(fig, "rq3_comp")
            card_close()

        show = prof.rename(columns={"n": "Jumlah", "pct": "% record", "mean_global": "Rata-rata global (M)",
                                    "median_global": "Median global (M)"})
        st.dataframe(show.round(2), hide_index=True, use_container_width=True)

        top = prof.iloc[-1]; low = prof.iloc[0]
        lows = prof.sort_values("JP_share%").iloc[0]
        st.markdown(f'<div class="note"><b>Temuan.</b> {int(top["n"]):,} record ({top["pct"]:.1f}%) berada di segmen '
                    f'<b>{top["Segmen"]}</b> dengan rata-rata {top["mean_global"]:.2f} juta kopi, sedangkan '
                    f'{low["pct"]:.1f}% record di segmen <b>{low["Segmen"]}</b> rata-rata hanya {low["mean_global"]:.2f} juta. '
                    f'Segmen dengan porsi Jepang terendah: <b>{lows["Segmen"]}</b> ({lows["JP_share%"]:.1f}% dari penjualannya).</div>',
                    unsafe_allow_html=True)

        a, b = st.columns(2, gap="large")
        with a:
            card_open("📐 Validasi Jumlah Cluster")
            ks = R["_ksel"]
            if ks is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ks["k"], y=ks["silhouette"], name="Silhouette", mode="lines+markers",
                                         line=dict(color=CLR_SUCCESS)))
                fig.add_trace(go.Scatter(x=ks["k"], y=ks["inertia"], name="Inertia", mode="lines+markers",
                                         line=dict(color=CLR_ACCENT), yaxis="y2"))
                fig.update_layout(**PLOTLY_LAYOUT, height=330, xaxis=dict(title="k", dtick=1, **AXIS),
                                  yaxis=dict(title="Silhouette", **AXIS),
                                  yaxis2=dict(title="Inertia", overlaying="y", side="right", showgrid=False),
                                  legend=dict(orientation="h", y=1.12))
                fig.add_vline(x=r3["chosen_k"], line_dash="dash", line_color=CLR_PRIMARY)
                chart(fig, "rq3_k")
            sil = {int(r_["k"]): r_["silhouette"] for r_ in r3["k_table"]}
            st.caption(f"k={r3['chosen_k']} dipilih (silhouette {sil[r3['chosen_k']]:.3f}; stabilitas ARI "
                       f"{r3['stability_ARI_mean']:.3f}). Silhouette tertinggi: k={r3['best_silhouette_k']} "
                       f"({sil[r3['best_silhouette_k']]:.3f}); elbow: k={r3['elbow_k']}. "
                       + ("" if r3["chosen_k"] == r3["best_silhouette_k"] else
                          "k yang dipilih bukan optimum mutlak — dipilih agar segmen dapat diprofilkan."))
            card_close()
        with b:
            card_open("🔬 Sebaran Segmen (log penjualan NA vs JP)")
            asg = R["_assign"]
            if asg is not None:
                smp = asg.sample(min(4000, len(asg)), random_state=42)
                fig = go.Figure()
                for s in segs:
                    m = smp[smp["Segmen"] == s]
                    fig.add_trace(go.Scatter(x=np.log1p(m["NA_Sales"]), y=np.log1p(m["JP_Sales"]), mode="markers", name=s,
                                             marker=dict(color=colors[s], size=4, opacity=0.6)))
                fig.update_layout(**PLOTLY_LAYOUT, height=330, xaxis=dict(title="log(1 + NA)", **AXIS),
                                  yaxis=dict(title="log(1 + JP)", **AXIS), legend=dict(orientation="h", y=1.12))
                chart(fig, "rq3_scatter")
            card_close()

        card_open("🏷️ Ciri tiap Segmen: genre & platform yang paling berlebih (lift)")
        for dim in ["Genre", "Platform"]:
            comp = r3["composition"][dim]
            st.markdown(f"**{dim}** — χ² p={comp['chi2_p']:.2g}, Cramér's V={comp['cramers_v']:.2f} "
                        f"(V kecil = hubungan lemah; kategori jarang membuat χ² hanya indikatif)")
            for s in segs:
                st.markdown(f"- **{s}**: " + "; ".join(comp["top_lift"][s]))
        card_close()

# ─────────────────────────────────────────────────────────────────────────────
# VALIDASI
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="note">Ringkasan pemeriksaan kualitas data yang menjadi dasar klaim di naskah.</div>',
                unsafe_allow_html=True)
    a, b = st.columns(2, gap="large")
    with a:
        card_open("🗂️ Audit Dataset")
        rows = [
            ("Record (judul × platform)", f"{len(df):,}"),
            ("Judul unik", f"{df['Name'].nunique():,}"),
            ("Baris tabel game / publisher", f"{info['n_game_rows']:,} / {info['n_publisher_rows']:,}"),
            ("Tahun rilis kosong", f"{info['year_null']:,} ({info['year_null'] / len(df):.2%})"),
            ("Publisher 'N/A' atau 'Unknown'", f"{info['pub_missing']:,}"),
            ("Duplikat (judul, platform)", f"{info['dup_name_platform']:,}"),
            ("Penunjuk genre dikoreksi saat dimuat", f"{info['pointer_fixes']}"),
            ("Sumber data aktif", info["source"]),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["Pemeriksaan", "Nilai"]), hide_index=True, use_container_width=True)
        if info.get("db_error") and info["source"] != "MySQL":
            st.caption(f"MySQL tidak terjangkau ({info['db_error']}); memakai dump SQL.")
        card_close()
    with b:
        card_open("📌 Catatan Penggunaan Angka")
        st.markdown(
            "- Pangsa regional dihitung dari **empat wilayah**; sebutkan penyebutnya di naskah.\n"
            "- Prediksi hanya memakai atribut produk (Year, Platform, Genre, Publisher); kolom penjualan, Rank, dan Name "
            "tidak dipakai sebagai fitur.\n"
            "- Segmentasi memakai penjualan regional; prediksi memakai atribut produk — keduanya sengaja dipisah.\n"
            "- Nilai tahun kosong diisi median data latih; Publisher 'N/A'/'Unknown' digabung.\n"
            "- Semua angka model dan segmen berasal dari folder `hasil/`; tidak ada angka bawaan di aplikasi ini.")
        card_close()

st.markdown(f"""
<div style="text-align:center;margin-top:2rem;padding:1rem;border-top:1px solid #2a2a45;color:#555577;font-size:.8rem;">
🎮 Video Game Sales Analytics &nbsp;|&nbsp; {len(df):,} record · {df['Name'].nunique():,} judul unik
</div>""", unsafe_allow_html=True)
