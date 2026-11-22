#!/usr/bin/env python3
"""
rq_pipeline.py -- Pipeline analisis RQ1-RQ3 (Video Game Sales), versi bersih tanpa kebocoran.

RQ1  Pola penjualan genre & platform: Amerika Utara vs Eropa vs Jepang
RQ2  Prediksi penjualan global dari Genre, Platform, Publisher, Tahun rilis (+ atribut paling informatif)
RQ3  Segmentasi K-Means dari pola penjualan regional (NA, EU, JP) + profil tiap segmen

Pakai:
  python rq_pipeline.py --sql video_game_sales.sql --out hasil/
  python rq_pipeline.py --mysql "mysql+pymysql://root:@localhost:3306/video_game_sales" --out hasil/
  opsi: --k 3   (paksa jumlah cluster; default: dipilih otomatis dari silhouette, k>=3)
        --quick (RF 60 pohon, untuk uji cepat)

Prinsip anti-kebocoran (semua ada di fungsi `prep` dan `GroupKFold`):
  * split per JUDUL (Name): versi platform lain dari judul yang sama tidak pernah tersebar ke train & test
  * median Year, pooling publisher, dan StandardScaler dihitung dari TRAIN SAJA di setiap fold
  * fitur hanya atribut produk (Year, Platform, Genre, Publisher); kolom penjualan/Rank/Name tidak dipakai
"""
import argparse, json, os, re, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
SEED = 42
LOG = []


def say(s=""):
    print(s)
    LOG.append(s)


def head(s):
    say("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78)


# ----------------------------------------------------------------------------- loader
_TOK = re.compile(r"\s*(?:(\()|(\))|(,)|(NULL)|('(?:[^'\\]|\\.|'')*')|(-?\d+(?:\.\d+)?))")


def _unq(s):
    return s[1:-1].replace("\\'", "'").replace("''", "'").replace("\\\\", "\\").replace('\\"', '"')


def parse_table(text, table):
    rows, row = [], None
    cols = None
    for sm in re.finditer(r"INSERT INTO `%s` \((.*?)\) VALUES" % table, text):
        cols = [c.strip(" `") for c in sm.group(1).split(",")]
        pos = sm.end()
        while True:
            t = _TOK.match(text, pos)
            if not t:
                pos += 1
                if pos >= len(text) or text[pos - 1] == ";":
                    break
                continue
            pos, g = t.end(), t.lastindex
            if g == 1: row = []
            elif g == 2: rows.append(row); row = None
            elif g == 4: row.append(None)
            elif g == 5: row.append(_unq(t.group(5)))
            elif g == 6:
                v = t.group(6); row.append(float(v) if "." in v else int(v))
            if row is None and text[pos:pos + 1] == ";":
                break
    return pd.DataFrame(rows, columns=cols)


def load_tables(sql=None, mysql=None):
    names = ["game", "game_sales", "genre", "platform", "publisher"]
    if mysql:
        from sqlalchemy import create_engine
        eng = create_engine(mysql)
        return {t: pd.read_sql(f"SELECT * FROM {t}", eng) for t in names}
    text = open(sql, encoding="utf8", errors="replace").read()
    return {t: parse_table(text, t) for t in names}


# Koreksi 2 baris yang menunjuk game_id salah (baris benar sudah ada di tabel `game`).
# Idempotent: hanya berlaku jika penunjuk masih nilai lama (jadi aman dijalankan setelah patch SQL).
SALES_FIX = {12528: (5685, 8820),    # Syndicate (PC, 1992): Shooter -> Strategy
             14860: (9195, 10576)}   # Little Busters! Converted Edition (PSV): Adventure -> Action


def build_flat(T):
    gs = T["game_sales"].copy()
    nfix = 0
    for rank, (old, new) in SALES_FIX.items():
        m = (gs["game_rank"] == rank) & (gs["game_game_id"] == old)
        nfix += int(m.sum()); gs.loc[m, "game_game_id"] = new
    df = (gs.merge(T["game"], left_on="game_game_id", right_on="game_id")
            .merge(T["platform"], left_on="platform_platform_id", right_on="platform_id")
            .merge(T["genre"], left_on="genre_genre_id", right_on="genre_id")
            .merge(T["publisher"], left_on="publisher_publisher_id", right_on="publisher_id"))
    df = df.rename(columns={"game_rank": "Rank", "game_title": "Name", "platform_name": "Platform",
                            "release_year": "Year", "genre_name": "Genre", "publisher_name": "Publisher",
                            "na_sales": "NA_Sales", "eu_sales": "EU_Sales", "jp_sales": "JP_Sales",
                            "other_sales": "Other_Sales", "global_sales": "Global_Sales"})
    df = df[["Rank", "Name", "Platform", "Year", "Genre", "Publisher",
             "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]].sort_values("Rank").reset_index(drop=True)
    for c in ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales", "Year"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, nfix


# ----------------------------------------------------------------------------- RQ1
def rq1(df, out):
    from scipy.spatial.distance import jensenshannon
    head("RQ1 -- Pola penjualan genre & platform: NA vs EU vs JP")
    R = {}
    reg = ["NA_Sales", "EU_Sales", "JP_Sales"]
    tot4 = df[reg + ["Other_Sales"]].sum()
    R["totals_M"] = {k[:-6]: round(float(v), 2) for k, v in tot4.items()}
    R["share_of_4_regions_pct"] = {k[:-6]: round(float(v / tot4.sum() * 100), 2) for k, v in tot4.items()}
    s3 = df[reg].sum()
    R["share_of_NA_EU_JP_pct"] = {k[:-6]: round(float(v / s3.sum() * 100), 2) for k, v in s3.items()}
    say("  Total (juta): " + ", ".join(f"{k}={v:,.2f}" for k, v in R["totals_M"].items()))
    say("  Pangsa dari 4 wilayah (%): " + ", ".join(f"{k}={v}" for k, v in R["share_of_4_regions_pct"].items()))
    say("  Pangsa dari NA+EU+JP (%) : " + ", ".join(f"{k}={v}" for k, v in R["share_of_NA_EU_JP_pct"].items()))

    sp = df[reg].corr(method="spearman").round(3)
    R["spearman"] = sp.to_dict()
    sp.to_csv(f"{out}/rq1_spearman.csv")
    say("\n  Korelasi Spearman antar wilayah (per record):\n" + sp.to_string().replace("\n", "\n    "))

    out_rows = {}
    for dim in ["Genre", "Platform"]:
        g = df.groupby(dim)[reg].sum()
        share = (g / g.sum() * 100).round(2)
        share.to_csv(f"{out}/rq1_{dim.lower()}_share_by_region.csv")
        P = (g / g.sum()).T.values
        js = {"NA-EU": jensenshannon(P[0], P[1], base=2), "NA-JP": jensenshannon(P[0], P[2], base=2),
              "EU-JP": jensenshannon(P[1], P[2], base=2)}
        top = {r[:-6]: [f"{i} ({share.loc[i, r]:.1f}%)" for i in share[r].sort_values(ascending=False).index[:3]]
               for r in reg}
        out_rows[dim] = {"js": {k: round(float(v), 3) for k, v in js.items()}, "top3": top}
        say(f"\n  {dim}: jarak Jensen-Shannon (basis 2; 0=identik, 1=beda total): "
            + ", ".join(f"{k}={v:.3f}" for k, v in js.items()))
        for r, t in top.items():
            say(f"    top-3 {dim} di {r}: " + "; ".join(t))
    R["by_dimension"] = out_rows

    # uji signifikansi sederhana: bootstrap judul untuk JS genre NA-JP (CI 95%)
    rng = np.random.default_rng(SEED)
    titles = df["Name"].unique(); grp = {t: d for t, d in df.groupby("Name")}
    boots = []
    for _ in range(200):
        samp = rng.choice(titles, size=len(titles), replace=True)
        b = pd.concat([grp[t] for t in samp])
        g = b.groupby("Genre")[reg].sum().reindex(df["Genre"].unique()).fillna(0)
        P = (g / g.sum()).T.values
        boots.append((jensenshannon(P[0], P[1], base=2), jensenshannon(P[0], P[2], base=2)))
    boots = np.array(boots)
    ci = {"NA-EU": np.percentile(boots[:, 0], [2.5, 97.5]).round(3).tolist(),
          "NA-JP": np.percentile(boots[:, 1], [2.5, 97.5]).round(3).tolist()}
    R["genre_js_bootstrap_ci95"] = ci
    say(f"\n  Bootstrap judul (200x), JS genre CI95%: NA-EU {ci['NA-EU']} | NA-JP {ci['NA-JP']}  "
        f"-> {'CI tidak tumpang-tindih (asimetri nyata)' if ci['NA-EU'][1] < ci['NA-JP'][0] else 'CI tumpang-tindih'}")
    return R


# ----------------------------------------------------------------------------- RQ2
def prep(df, tr, thr=50):
    """Fitur atribut produk. SEMUA statistik dihitung dari indeks train `tr` saja."""
    d = df[["Year", "Platform", "Genre", "Publisher"]].copy()
    d["Year"] = d["Year"].fillna(d.iloc[tr]["Year"].median())
    d["Publisher"] = d["Publisher"].replace({"N/A": "Unknown"})
    cnt = d.iloc[tr]["Publisher"].value_counts()
    d["Publisher"] = d["Publisher"].map(lambda x: x if cnt.get(x, 0) >= thr else "Other publishers")
    return pd.get_dummies(d, columns=["Platform", "Genre", "Publisher"], dtype=float)


def metr(y, p):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {"MAE": float(mean_absolute_error(y, p)), "RMSE": float(mean_squared_error(y, p) ** 0.5),
            "R2": float(r2_score(y, p))}


def models(quick):
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return {
        "Baseline (mean)": lambda: DummyRegressor(strategy="mean"),
        "Baseline (median)": lambda: DummyRegressor(strategy="median"),
        "Random Forest": lambda: RandomForestRegressor(n_estimators=60 if quick else 300, min_samples_leaf=3,
                                                       n_jobs=-1, random_state=SEED),
        "HistGradientBoosting": lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05,
                                                                      random_state=SEED),
        "HGB (MAE loss)": lambda: HistGradientBoostingRegressor(loss="absolute_error", max_iter=200,
                                                                learning_rate=0.05, random_state=SEED),
        "MLP (128-128)": lambda: make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(128, 128), early_stopping=True, max_iter=300, random_state=SEED)),
    }


def rq2(df, out, quick):
    from sklearn.model_selection import GroupKFold, KFold
    head("RQ2 -- Prediksi penjualan global dari atribut produk (tanpa kebocoran)")
    y = df["Global_Sales"].values
    idx = np.arange(len(df))
    X_cols_note = prep(df, idx).shape[1]
    say(f"  fitur: Year, Platform, Genre, Publisher (pooled)  -> {X_cols_note} kolom setelah encoding")
    say(f"  judul unik = {df['Name'].nunique():,} ; split = GroupKFold(5) per judul, statistik dari train saja")
    folds = list(GroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(idx, groups=df["Name"].values))
    M = models(quick)
    per_fold = {k: [] for k in M}
    oof = {k: np.zeros(len(df)) for k in M}
    for fi, (tr, te) in enumerate(folds):
        X = prep(df, tr).values
        for name, mk in M.items():
            m = mk().fit(X[tr], y[tr]); p = m.predict(X[te])
            per_fold[name].append(metr(y[te], p)); oof[name][te] = p
    R = {"cv": {}}
    rows = []
    say(f"\n  {'Model':24s}{'MAE':>16s}{'RMSE':>16s}{'R2':>16s}")
    for name in M:
        a = {k: np.array([f[k] for f in per_fold[name]]) for k in ["MAE", "RMSE", "R2"]}
        R["cv"][name] = {k: [round(float(v.mean()), 4), round(float(v.std()), 4)] for k, v in a.items()}
        pooled = metr(y, oof[name]); R["cv"][name]["pooled"] = {k: round(v, 4) for k, v in pooled.items()}
        say(f"  {name:24s}" + "".join(f"{a[k].mean():>9.3f}+/-{a[k].std():<5.3f}" for k in ["MAE", "RMSE", "R2"]))
        rows.append({"model": name, **{f"{k}_mean": a[k].mean() for k in a}, **{f"{k}_sd": a[k].std() for k in a}})
    pd.DataFrame(rows).to_csv(f"{out}/rq2_cv_metrics.csv", index=False)
    say("  (mean +/- SD antar 5 fold; setiap judul diuji tepat satu kali)")

    # ---- harga kebocoran: sama-sama HGB, sama-sama diuji sekali per baris (pooled OOF)
    Xg = prep(df, idx).values                                   # statistik dari SEMUA data (gaya notebook)
    leak = np.zeros(len(df))
    for tr, te in KFold(5, shuffle=True, random_state=SEED).split(idx):
        leak[te] = models(quick)["HistGradientBoosting"]().fit(Xg[tr], y[tr]).predict(Xg[te])
    lk = metr(y, leak); cl = R["cv"]["HistGradientBoosting"]["pooled"]
    R["leakage_demo"] = {"leaky_random_split_global_stats": {k: round(v, 4) for k, v in lk.items()},
                         "clean_grouped_train_only": cl}
    say(f"\n  Demo kebocoran (HGB, metrik gabungan 16.598 prediksi OOF):")
    say(f"    split acak + statistik global (gaya notebook) : MAE={lk['MAE']:.3f} RMSE={lk['RMSE']:.3f} R2={lk['R2']:+.3f}")
    say(f"    split per judul + statistik train saja         : MAE={cl['MAE']:.3f} RMSE={cl['RMSE']:.3f} R2={cl['R2']:+.3f}")
    say(f"    -> selisih R2 akibat kebocoran = {lk['R2'] - cl['R2']:+.3f}")

    # ---- permutation importance per KELOMPOK fitur: model terbaik (R2) + HGB sebagai pembanding urutan
    cand = [k for k in M if not k.startswith("Baseline")]
    best = max(cand, key=lambda k: R["cv"][k]["R2"][0])
    R["best_model"] = best
    best_mae = min(cand, key=lambda k: R["cv"][k]["MAE"][0])
    R["best_model_mae"] = best_mae
    med = R["cv"]["Baseline (median)"]["MAE"][0]
    say(f"\n  Model terbaik menurut R2 : {best}")
    say(f"  Model terbaik menurut MAE: {best_mae} (MAE={R['cv'][best_mae]['MAE'][0]:.3f}) vs baseline median {med:.3f} -> "
        + ("model MENGALAHKAN baseline median" if R['cv'][best_mae]['MAE'][0] < med else
           "baseline median tetap lebih baik pada MAE (target sangat miring: median meminimalkan MAE; "
           "model unggul pada RMSE/R2, bukan MAE)"))
    rng = np.random.default_rng(SEED)
    R["importance"] = {}
    irows = []
    for mname in list(dict.fromkeys([best, "HistGradientBoosting"])):
        imp = {g: {"dR2": [], "dMAE": []} for g in ["Year", "Platform", "Genre", "Publisher"]}
        for tr, te in folds:
            Xdf = prep(df, tr); X = Xdf.values
            m = models(quick)[mname]().fit(X[tr], y[tr])
            base = metr(y[te], m.predict(X[te]))
            for g in imp:
                cols = [i for i, c in enumerate(Xdf.columns) if (c == g if g == "Year" else c.startswith(g + "_"))]
                dr, dm = [], []
                for _ in range(8):
                    Xp = X[te].copy(); perm = rng.permutation(len(te))
                    Xp[:, cols] = Xp[perm][:, cols]
                    pm = metr(y[te], m.predict(Xp))
                    dr.append(base["R2"] - pm["R2"]); dm.append(pm["MAE"] - base["MAE"])
                imp[g]["dR2"].append(np.mean(dr)); imp[g]["dMAE"].append(np.mean(dm))
        tot = sum(max(np.mean(v["dR2"]), 0) for v in imp.values()) or 1.0
        R["importance"][mname] = {}
        say(f"\n  Permutation importance [{mname}] (R2 turun & MAE naik saat kelompok fitur diacak; rata-rata 5 fold):")
        say(f"    {'Fitur':10s}{'dR2':>16s}{'dMAE':>16s}{'porsi':>9s}")
        for g, v in sorted(imp.items(), key=lambda kv: -np.mean(kv[1]["dR2"])):
            share = max(np.mean(v["dR2"]), 0) / tot * 100
            R["importance"][mname][g] = {"dR2": [round(float(np.mean(v["dR2"])), 4), round(float(np.std(v["dR2"])), 4)],
                                         "dMAE": [round(float(np.mean(v["dMAE"])), 4), round(float(np.std(v["dMAE"])), 4)],
                                         "share_pct": round(float(share), 1)}
            say(f"    {g:10s}{np.mean(v['dR2']):>9.4f}+/-{np.std(v['dR2']):<5.4f}"
                f"{np.mean(v['dMAE']):>9.4f}+/-{np.std(v['dMAE']):<5.4f}{share:>8.1f}%")
            irows.append({"model": mname, "feature": g, "dR2": np.mean(v["dR2"]), "dR2_sd": np.std(v["dR2"]),
                          "dMAE": np.mean(v["dMAE"]), "dMAE_sd": np.std(v["dMAE"]), "share_pct": share})
    pd.DataFrame(irows).to_csv(f"{out}/rq2_importance.csv", index=False)
    return R


# ----------------------------------------------------------------------------- RQ3
def rq3(df, out, k_force):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
    from scipy.stats import chi2_contingency
    head("RQ3 -- Segmentasi K-Means dari pola penjualan regional (NA, EU, JP)")
    reg = ["NA_Sales", "EU_Sales", "JP_Sales"]
    d = df[df[reg].sum(axis=1) > 0].copy()
    say(f"  baris dipakai: {len(d):,} dari {len(df):,} (dibuang {len(df) - len(d)} baris yang NA+EU+JP = 0)")
    Xs = StandardScaler().fit_transform(np.log1p(d[reg].values))
    say("  fitur: log(1+penjualan) NA/EU/JP, lalu standardisasi (penjualan sangat miring ke kanan)")
    R = {"n_rows": int(len(d)), "k_table": []}
    say(f"\n    {'k':>2s}{'inertia':>12s}{'silhouette':>12s}{'Davies-B':>10s}{'Calinski-H':>12s}")
    for k in range(2, 9):
        km = KMeans(k, n_init=10, random_state=SEED).fit(Xs)
        row = {"k": k, "inertia": float(km.inertia_),
               "silhouette": float(silhouette_score(Xs, km.labels_, sample_size=8000, random_state=SEED)),
               "davies_bouldin": float(davies_bouldin_score(Xs, km.labels_)),
               "calinski_harabasz": float(calinski_harabasz_score(Xs, km.labels_))}
        R["k_table"].append(row)
        say(f"    {k:>2d}{row['inertia']:12.1f}{row['silhouette']:12.3f}{row['davies_bouldin']:10.3f}{row['calinski_harabasz']:12.0f}")
    pd.DataFrame(R["k_table"]).to_csv(f"{out}/rq3_k_selection.csv", index=False)
    ks = np.array([r["k"] for r in R["k_table"]]); inert = np.array([r["inertia"] for r in R["k_table"]])
    p1, p2 = np.array([ks[0], inert[0]], float), np.array([ks[-1], inert[-1]], float)
    dist = [abs((p2 - p1)[0] * (np.array([k, i]) - p1)[1] - (p2 - p1)[1] * (np.array([k, i]) - p1)[0])
            / np.linalg.norm(p2 - p1) for k, i in zip(ks, inert)]
    elbow = int(ks[int(np.argmax(dist))])
    sil = {r["k"]: r["silhouette"] for r in R["k_table"]}
    kbest_all = max(sil, key=sil.get)
    kbest_3 = max([k for k in sil if k >= 3], key=sil.get)
    R.update({"elbow_k": elbow, "best_silhouette_k": kbest_all, "best_silhouette_k_ge3": kbest_3})
    k = k_force or kbest_3
    R["chosen_k"] = int(k)
    say(f"\n  elbow (jarak maks. ke garis ujung) : k={elbow}")
    say(f"  silhouette tertinggi               : k={kbest_all} ({sil[kbest_all]:.3f}); untuk k>=3: k={kbest_3} ({sil[kbest_3]:.3f})")
    say(f"  k yang dipakai                     : {k}  (silhouette={sil[k]:.3f})"
        + ("" if k == kbest_all else "  <- BUKAN optimum mutlak; k=2 hanya memisah 'terjual/hampir tidak'; dipilih agar segmen bisa diprofilkan"))

    km = KMeans(k, n_init=10, random_state=SEED).fit(Xs)
    # stabilitas: ARI antar seed
    aris = [adjusted_rand_score(km.labels_, KMeans(k, n_init=10, random_state=s).fit_predict(Xs)) for s in range(1, 8)]
    R["stability_ARI_mean"] = round(float(np.mean(aris)), 3)
    say(f"  stabilitas (ARI vs 7 seed lain)    : {np.mean(aris):.3f}")

    d["raw"] = km.labels_
    order = d.groupby("raw")["Global_Sales"].mean().sort_values().index
    names = (["Low-Sales", "Mid-Tier", "Mega-Hit"] if k == 3 else [f"Tier {i + 1}" for i in range(k)])
    d["Segmen"] = d["raw"].map({o: names[i] for i, o in enumerate(order)})
    prof = d.groupby("Segmen").agg(n=("Global_Sales", "size"), mean_global=("Global_Sales", "mean"),
                                   median_global=("Global_Sales", "median"), NA=("NA_Sales", "mean"),
                                   EU=("EU_Sales", "mean"), JP=("JP_Sales", "mean")).loc[names]
    prof["pct"] = prof["n"] / prof["n"].sum() * 100
    sh = d.groupby("Segmen")[reg].sum().loc[names]
    sh = sh.div(sh.sum(axis=1), axis=0) * 100
    prof["NA_share%"], prof["EU_share%"], prof["JP_share%"] = sh["NA_Sales"], sh["EU_Sales"], sh["JP_Sales"]
    prof.round(2).to_csv(f"{out}/rq3_profile.csv")
    say("\n  Profil segmen (rata-rata penjualan per judul-platform, juta; share = komposisi regional):")
    say(prof.round(2).to_string().replace("\n", "\n    "))
    R["profile"] = prof.round(3).reset_index().to_dict(orient="records")

    R["composition"] = {}
    for dim in ["Genre", "Platform"]:
        ct = pd.crosstab(d["Segmen"], d[dim]).loc[names]
        chi2, p, dof, _ = chi2_contingency(ct)
        v = float(np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1))))
        overall = d[dim].value_counts(normalize=True)
        tops = {}
        for s in names:
            inside = ct.loc[s]; share = inside / inside.sum()
            lift = (share / overall.reindex(share.index)).where(inside >= 5)
            top = lift.sort_values(ascending=False).head(3).index
            tops[s] = [f"{c} ({share[c] * 100:.1f}%, lift {lift[c]:.1f}x)" for c in top]
        R["composition"][dim] = {"chi2_p": float(p), "cramers_v": round(v, 3), "top_lift": tops}
        say(f"\n  {dim} x Segmen: chi2 p={p:.2g}, Cramer's V={v:.3f}  (V kecil=hubungan lemah; kategori jarang -> chi2 hanya indikatif)")
        for s in names:
            say(f"    {s:10s}: " + "; ".join(tops[s]))
    d[["Rank", "Name", "Platform", "Genre", "NA_Sales", "EU_Sales", "JP_Sales", "Global_Sales", "Segmen"]].to_csv(
        f"{out}/rq3_assignments.csv", index=False)
    top5 = d[d["Segmen"] == names[-1]].nlargest(5, "Global_Sales")[["Name", "Platform", "Global_Sales"]]
    say("\n  Contoh judul segmen tertinggi:\n    " + top5.to_string(index=False).replace("\n", "\n    "))
    return R


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sql"); ap.add_argument("--mysql"); ap.add_argument("--out", default="hasil")
    ap.add_argument("--k", type=int); ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if not (a.sql or a.mysql):
        sys.exit("Berikan --sql <dump> atau --mysql <url>")
    os.makedirs(a.out, exist_ok=True)
    T = load_tables(a.sql, a.mysql)
    df, nfix = build_flat(T)
    head(f"DATA -- {len(df):,} record, {df['Name'].nunique():,} judul unik, {df['Publisher'].nunique()} publisher "
         f"({nfix} penunjuk genre dikoreksi)")
    R = {"n_records": int(len(df)), "n_titles": int(df["Name"].nunique()), "genre_pointer_fixes": nfix}
    R["rq1"] = rq1(df, a.out)
    R["rq2"] = rq2(df, a.out, a.quick)
    R["rq3"] = rq3(df, a.out, a.k)
    json.dump(R, open(f"{a.out}/results.json", "w"), indent=2, default=float)
    open(f"{a.out}/report.txt", "w", encoding="utf8").write("\n".join(LOG))
    say(f"\n[tersimpan] {a.out}/results.json, report.txt, dan tabel CSV")


if __name__ == "__main__":
    main()
