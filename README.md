# Analisis Penjualan Video Game: Pasar Regional, Prediksi Pra-Rilis, dan Segmentasi K-Means

Kode sumber untuk artikel *Perbandingan Pasar Regional, Prediksi Penjualan Global Berbasis Atribut Pra-Rilis, dan Segmentasi K-Means Video Game* (Ichsani, Murfid, dan Ashidiqi).

Seluruh angka dan gambar pada artikel dihasilkan dari kode di repositori ini dengan `seed` 42.

## Isi repositori

| Berkas / folder | Fungsi |
|---|---|
| `analysis_pipeline.py` | Pipeline analisis RQ1 (pasar regional), RQ2 (prediksi), RQ3 (K-Means). Menulis keluaran ke `results/`. |
| `make_figures.py` | Membuat Gambar 1–4 dari `results/`. |
| `video_game_sales_analysis.ipynb` | Notebook akhir berisi seluruh langkah beserta keluarannya. |
| `app.py` | Dasbor Streamlit (membaca data dan `results/`). |
| `video_game_sales_dump.sql` | Dump basis data MySQL (lima tabel). |
| `patch_genre_corrections.sql` | Koreksi dua penunjuk genre yang salah (cadangkan basis data dulu). |
| `results/` | Keluaran pipeline: `results.json`, tabel CSV, `report.txt`. |
| `figures/` | Gambar artikel (PNG). |
| `requirements.txt` | Daftar paket Python. |

## Cara menjalankan

1. Pasang Python 3.12 dan paket: `pip install -r requirements.txt`
2. Jalankan pipeline: `python analysis_pipeline.py --sql video_game_sales_dump.sql --out hasil`
3. Buat gambar: `python make_figures.py`
4. (Opsional) Dasbor: `streamlit run app.py`
5. (Opsional) Notebook: `jupyter notebook video_game_sales_analysis.ipynb`

Variabel lingkungan opsional: `VGS_DB_URL` (koneksi MySQL), `VGS_SQL` (jalur dump SQL), `VGS_RESULTS` (folder hasil), `VGS_CSV` (berkas `vgsales.csv` untuk audit silang di notebook).

## Data

Data penjualan berasal dari dataset publik *Video Game Sales* di Kaggle (estimasi VGChartz). Berkas `vgsales.csv` asli tidak disertakan; unduh dari Kaggle bila ingin menjalankan audit silang pada notebook. Dump SQL di repositori ini adalah hasil pemuatan data tersebut ke lima tabel relasional.

## Lingkungan yang dipakai

Python 3.12, scikit-learn 1.9.0.

## Sitasi

Jika memakai kode ini, mohon kutip artikel terkait (tautan akan ditambahkan setelah terbit).



=================================================================================================================================================================================

# Video Game Sales Analysis: Regional Markets, Pre-Release Predictions, and K-Means Segmentation

Source code for the article *Regional Market Comparison, Pre-Release Attribute-Based Global Sales Predictions, and K-Means Segmentation of Video Games* (Ichsani, Murfid, and Ashidiqi).

All figures and images in the article were generated from code in this repository with a seed of 42.

## Repository Contents

| Files / Folders | Functions |
|---|---|
| `analysis_pipeline.py` | Analysis pipeline for RQ1 (regional markets), RQ2 (prediction), RQ3 (K-Means). Writes output to `results/`. |
| `make_figures.py` | Creates Figures 1–4 from `results/`. |
| `video_game_sales_analysis.ipynb` | Final notebook containing all steps and their output. |
| `app.py` | Streamlit dashboard (reading data and `results/`). |
| `video_game_sales_dump.sql` | MySQL database dump (five tables). |
| `patch_genre_corrections.sql` | Correct two incorrect genre pointers (backup the database first). |
| `results/` | Pipeline output: `results.json`, CSV table, `report.txt`. |
| `figures/` | Article images (PNG). |
| `requirements.txt` | Python package list. |

## How to run

1. Install Python 3.12 and packages: `pip install -r requirements.txt`
2. Run the pipeline: `python analysis_pipeline.py --sql video_game_sales_dump.sql --out results`
3. Create a figure: `python make_figures.py`
4. (Optional) Dashboard: `streamlit run app.py`
5. (Optional) Notebook: `jupyter notebook video_game_sales_analysis.ipynb`

Optional environment variables: `VGS_DB_URL` (MySQL connection), `VGS_SQL` (SQL dump path), `VGS_RESULTS` (results folder), `VGS_CSV` (`vgsales.csv` file for cross-auditing in the notebook).

## Data

Sales data comes from the public *Video Game Sales* dataset on Kaggle (VGChartz estimates). The original `vgsales.csv` file is not included; download it from Kaggle if you want to run a cross-audit in the notebook. The SQL dump in this repository is the result of loading the data into five relational tables.

## Environment used

Python 3.12, scikit-learn 1.9.0.

## Citations

If using this code, please cite the relevant article (a link will be added after publication).