"""Membuat semua gambar artikel dari berkas hasil/ (keluaran rq_pipeline.py / notebook).
Deterministik: tanpa acak kecuali random_state tetap; metadata PNG dikosongkan sehingga dua kali jalan
menghasilkan berkas yang identik byte-per-byte.
Pakai:  python make_figures.py [folder_hasil] [folder_gambar]
Prasyarat: jalankan dulu rq_pipeline.py (atau notebook) agar hasil/ terisi."""
import json, os, sys, hashlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "hasil")
FIG = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .3,
                     "figure.dpi": 100, "savefig.dpi": 220, "svg.hashsalt": "x"})
C = {"NA": "#D55E00", "EU": "#0072B2", "JP": "#E69F00", "Other": "#999999", "base": "#B8BFCC", "model": "#2F6DB5", "hi": "#C0392B"}
R = json.load(open(f"{OUT}/results.json"))

def save(fig, name):
    p = f"{FIG}/{name}.png"
    fig.savefig(p, bbox_inches="tight", metadata={"Software": None}, facecolor="white")
    plt.close(fig)

def idn(x, n=1): return f"{x:.{n}f}".replace(".", ",")

# ---------- Gambar 1: kerangka penelitian ----------
def framework():
    fig, ax = plt.subplots(figsize=(7.2, 7.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off"); ax.grid(False)
    t = R["rq1"]["share_of_4_regions_pct"]; cv = R["rq2"]["cv"]; bm = R["rq2"]["best_model"]; q3 = R["rq3"]
    stages = [
     ("Data &\naudit", "#D8EBD8", "#4C9A4C",
      ["Basis data MySQL relasional (5 tabel) + audit CSV", f"{R['n_records']:,} baris; {R['n_titles']:,} judul unik".replace(",", "."),
       f"Koreksi {R['genre_pointer_fixes']} penunjuk genre yang salah"],
      ("Basis data\nrelasional\nterverifikasi", "Bukan CSV mentah")),
     ("RQ1\nPasar\nregional", "#D8EBD8", "#4C9A4C",
      ["Pangsa NA/EU/JP: penyebut 4 wilayah & 3 wilayah", "Korelasi Spearman antar wilayah",
       "Jarak Jensen–Shannon genre/platform + bootstrap CI"],
      ("Perbandingan\npasar regional", "Kolom 1 matriks")),
     ("RQ2\nPrediksi\npenjualan", "#D8EBD8", "#4C9A4C",
      ["Fitur pra-rilis: Year, Platform, Genre, Publisher", "GroupKFold per judul; statistik dari data latih saja",
       "RF, HGB, MLP vs baseline; MAE, RMSE, R²"],
      ("Prediksi pra-rilis\n+ evaluasi\nout-of-sample", "Kolom 2–3 matriks")),
     ("RQ3\nSegmentasi\nregional", "#D8EBD8", "#4C9A4C",
      ["K-Means pada log(1+penjualan) NA, EU, JP", "Elbow, silhouette, Davies–Bouldin, ARI",
       "Profil segmen: pangsa regional, genre, platform"],
      ("Segmentasi\nregional", "Kolom 4 matriks")),
    ]
    X1, W1, X2, W2, X3, W3 = 2, 17, 25, 46, 76, 19
    H, GAP, top = 15.5, 3.6, 96
    ax.add_patch(FancyBboxPatch((X3 - 1.5, 17.5), W3 + 3, 80.5, boxstyle="round,pad=0.4,rounding_size=1.5", fc="none", ec="#6B7280", lw=1, ls="--"))
    ax.text(X3 + W3 / 2, 96.2, "Kontribusi ilmiah", ha="center", va="center", fontsize=8.5, fontweight="bold")
    top = 93
    for i, (nm, fc, ec, items, (c1, c2)) in enumerate(stages):
        y = top - i * (H + GAP) - H
        ax.add_patch(FancyBboxPatch((X1, y), W1, H, boxstyle="round,pad=0.2,rounding_size=1.2", fc=fc, ec=ec, lw=1.4))
        ax.text(X1 + W1 / 2, y + H / 2, nm, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#1E5A1E")
        for j, it in enumerate(items):
            h = (H - 2 * 1.0) / 3
            yy = y + H - (j + 1) * (h + 0.0) + 0.4
            ax.add_patch(FancyBboxPatch((X2, yy), W2, h - 1.1, boxstyle="round,pad=0.1,rounding_size=0.8", fc="#F2F2F2", ec="#B8B8B8", lw=.8))
            ax.text(X2 + W2 / 2, yy + (h - 1.1) / 2, it, ha="center", va="center", fontsize=6.7, wrap=True)
        ax.annotate("", xy=(X2 - .3, y + H / 2), xytext=(X1 + W1 + .3, y + H / 2), arrowprops=dict(arrowstyle="-|>", color="#777"))
        ax.annotate("", xy=(X3 - .3, y + H / 2), xytext=(X2 + W2 + .3, y + H / 2), arrowprops=dict(arrowstyle="-|>", color="#777"))
        ax.add_patch(FancyBboxPatch((X3, y + 2.5), W3, H - 5, boxstyle="round,pad=0.2,rounding_size=1.2", fc="#FBDAD3", ec="#E0735F", lw=1.2))
        ax.text(X3 + W3 / 2, y + H / 2 + 1.6, c1, ha="center", va="center", fontsize=7.2, fontweight="bold", color="#B03A2E")
        ax.text(X3 + W3 / 2, y + 4.3, c2, ha="center", va="center", fontsize=6, color="#B03A2E")
        if i < 3: ax.annotate("", xy=(X1 + W1 / 2, y - GAP + .3), xytext=(X1 + W1 / 2, y - .3), arrowprops=dict(arrowstyle="-|>", color="#777", lw=1.4))
    ax.add_patch(FancyBboxPatch((X1, 1), 96, 11.5, boxstyle="round,pad=0.2,rounding_size=1.5", fc="#FFF1C2", ec="#C98A00", lw=1.6))
    ax.text(50, 9.2, "Kerangka terpadu tiga pilar", ha="center", fontsize=8.5, fontweight="bold", color="#7A4B00")
    ax.text(50, 4.3, "Perbandingan pasar regional + prediksi atribut pra-rilis dengan evaluasi out-of-sample\n+ segmentasi regional dalam satu rancangan empiris (sinkron dengan Tabel 1)",
            ha="center", fontsize=7)
    ax.annotate("", xy=(X1 + W1 / 2, 12.9), xytext=(X1 + W1 / 2, top - 3 * (H + GAP) - H - .3), arrowprops=dict(arrowstyle="-|>", color="#777", lw=1.4))
    save(fig, "fig1_kerangka")

# ---------- Gambar 2: RQ1 ----------
def rq1():
    q = R["rq1"]; s4, s3 = q["share_of_4_regions_pct"], q["share_of_NA_EU_JP_pct"]
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 6.0))
    a = ax[0, 0]; regs = ["NA", "EU", "JP", "Other"]; x = np.arange(4); w = .38
    b1 = a.bar(x - w / 2, [s4[r] for r in regs], w, color="#2F6DB5", label="Penyebut 4 wilayah")
    b2 = a.bar(x[:3] + w / 2, [s3[r] for r in regs[:3]], w, color="#E69F00", label="Penyebut NA+EU+JP")
    for r_, bs in ((s4, b1), (s3, b2)):
        for bb in bs: a.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + .8, idn(bb.get_height()), ha="center", fontsize=6.5)
    a.set_xticks(x); a.set_xticklabels(regs); a.set_ylabel("% penjualan"); a.set_ylim(0, 65); a.legend(fontsize=6.5, frameon=False)
    a.set_title("(a) Pangsa regional menurut penyebut")
    g = pd.read_csv(f"{OUT}/rq1_genre_share_by_region.csv", index_col=0); g.columns = ["NA", "EU", "JP"]
    g = g.loc[g.mean(axis=1).sort_values(ascending=False).index]
    a = ax[0, 1]; x = np.arange(len(g)); w = .27
    for k, r in enumerate(["NA", "EU", "JP"]): a.bar(x + (k - 1) * w, g[r], w, color=C[r], label=r)
    a.set_xticks(x); a.set_xticklabels(g.index, rotation=60, ha="right", fontsize=6.5); a.set_ylabel("% penjualan wilayah")
    a.legend(fontsize=6.5, frameon=False); a.set_title("(b) Komposisi genre")
    p = pd.read_csv(f"{OUT}/rq1_platform_share_by_region.csv", index_col=0); p.columns = ["NA", "EU", "JP"]
    p = p.loc[p.max(axis=1).sort_values(ascending=False).head(10).index]
    a = ax[1, 0]; x = np.arange(len(p))
    for k, r in enumerate(["NA", "EU", "JP"]): a.bar(x + (k - 1) * w, p[r], w, color=C[r], label=r)
    a.set_xticks(x); a.set_xticklabels(p.index, rotation=45, ha="right", fontsize=6.5); a.set_ylabel("% penjualan wilayah")
    a.legend(fontsize=6.5, frameon=False); a.set_title("(c) Komposisi platform (10 terbesar)")
    sp = pd.DataFrame(q["spearman"]).loc[["NA_Sales", "EU_Sales", "JP_Sales"], ["NA_Sales", "EU_Sales", "JP_Sales"]].values
    a = ax[1, 1]; a.grid(False); im = a.imshow(sp, cmap="RdBu", vmin=-1, vmax=1)
    a.set_xticks(range(3)); a.set_yticks(range(3)); a.set_xticklabels(["NA", "EU", "JP"]); a.set_yticklabels(["NA", "EU", "JP"])
    for i in range(3):
        for j in range(3): a.text(j, i, idn(sp[i, j], 2), ha="center", va="center", color="white" if abs(sp[i, j]) > .55 else "black")
    a.spines[:].set_visible(False); a.set_title("(d) Korelasi Spearman antar wilayah"); fig.colorbar(im, ax=a, shrink=.8)
    fig.tight_layout(); save(fig, "fig2_rq1")

# ---------- Gambar 3: RQ2 ----------
def rq2():
    cv = R["rq2"]["cv"]; names = list(cv); base = [n.startswith("Baseline") for n in names]
    lab = {"Baseline (mean)": "Dasar mean", "Baseline (median)": "Dasar median", "Random Forest": "RF", "HistGradientBoosting": "HGB",
           "HGB (MAE loss)": "HGB loss MAE", "MLP (128-128)": "MLP"}; lab = [lab[n] for n in names]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 3.1))
    for a, m, t in zip(ax[:2], ["R2", "MAE"], ["(a) R² (makin tinggi makin baik)", "(b) MAE (makin rendah makin baik)"]):
        v = [cv[n][m][0] for n in names]; e = [cv[n][m][1] for n in names]
        a.bar(range(len(names)), v, yerr=e, capsize=2, color=[C["base"] if b else C["model"] for b in base], error_kw=dict(lw=.8))
        a.set_xticks(range(len(names))); a.set_xticklabels(lab, fontsize=6, rotation=40, ha="right"); a.set_title(t, fontsize=7.5)
        a.axhline(0, color="k", lw=.6)
    bm = R["rq2"]["best_model"]; imp = R["rq2"]["importance"]; feats = ["Publisher", "Platform", "Genre", "Year"]
    a = ax[2]; w = .38
    for i, (m, col) in enumerate(zip(imp, [C["model"], "#E69F00"])):
        a.bar(np.arange(4) + (i - .5) * w, [imp[m][f]["share_pct"] for f in feats], w, color=col, label=m.replace("HistGradientBoosting", "HGB"))
    a.set_xticks(range(4)); a.set_xticklabels(feats, fontsize=6.5); a.set_ylabel("% dari total pengaruh"); a.legend(fontsize=6, frameon=False)
    a.set_title("(c) Pengaruh atribut", fontsize=7.5)
    fig.tight_layout(); save(fig, "fig3_rq2")

# ---------- Gambar 4: RQ3 ----------
def rq3():
    ks = pd.DataFrame(R["rq3"]["k_table"]); prof = pd.read_csv(f"{OUT}/rq3_profile.csv"); asg = pd.read_csv(f"{OUT}/rq3_assignments.csv")
    k0 = R["rq3"]["chosen_k"]; segs = list(prof["Segmen"]); pal = dict(zip(segs, ["#999999", "#2F6DB5", "#C0392B"]))
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 3.1))
    a = ax[0]; a.plot(ks["k"], ks["silhouette"], "o-", color="#1B7F4B"); a.set_xlabel("k"); a.set_ylabel("Silhouette", color="#1B7F4B")
    b = a.twinx(); b.plot(ks["k"], ks["inertia"], "s--", color="#2F6DB5"); b.set_ylabel("Inertia", color="#2F6DB5"); b.grid(False)
    b.spines["right"].set_visible(True); a.axvline(k0, color=C["hi"], ls=":"); a.set_title(f"(a) Validasi k (dipakai k={k0})", fontsize=7.5)
    a = ax[1]; bot = np.zeros(len(prof))
    for r in ["NA", "EU", "JP"]:
        a.bar(prof["Segmen"], prof[f"{r}_share%"], bottom=bot, color=C[r], label=r); bot += prof[f"{r}_share%"].values
    a.set_ylabel("% penjualan segmen (NA+EU+JP)"); a.legend(fontsize=6, frameon=False, loc="upper right", ncol=3); a.set_ylim(0, 125)
    a.set_title("(b) Komposisi regional segmen", fontsize=7.5); a.tick_params(axis="x", labelsize=6.5)
    a = ax[2]; smp = asg.sample(min(4000, len(asg)), random_state=42)
    for s in segs:
        m = smp[smp["Segmen"] == s]; a.scatter(np.log1p(m["NA_Sales"]), np.log1p(m["JP_Sales"]), s=4, alpha=.5, color=pal[s], label=s, linewidths=0)
    a.set_xlabel("log(1 + NA)"); a.set_ylabel("log(1 + JP)"); a.legend(markerscale=2.5, fontsize=6, frameon=False); a.set_title("(c) Sebaran segmen", fontsize=7.5)
    fig.tight_layout(); save(fig, "fig4_rq3")

framework(); rq1(); rq2(); rq3()
for f in sorted(os.listdir(FIG)):
    if f.endswith(".png"): print(f, hashlib.md5(open(f"{FIG}/{f}", "rb").read()).hexdigest())
