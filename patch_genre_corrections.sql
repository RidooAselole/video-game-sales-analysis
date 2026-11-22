-- fix_genre_pointers.sql
-- Memperbaiki 2 baris game_sales yang menunjuk game_id dengan genre salah.
-- Baris `game` yang benar SUDAH ada tetapi tidak terpakai (game_id 8820 dan 10576).
-- Sumber kebenaran: vgsales.csv Kaggle (Rank 12528 = Strategy, Rank 14860 = Action).
-- Aman dijalankan ulang: klausa AND game_game_id = <nilai lama> membuatnya idempoten.
-- LAKUKAN BACKUP DULU:  mysqldump video_game_sales > backup_sebelum_patch.sql

START TRANSACTION;

-- 1) Syndicate (PC, 1992): Shooter -> Strategy
UPDATE game_sales SET game_game_id = 8820
 WHERE game_rank = 12528 AND game_game_id = 5685;

-- 2) Little Busters! Converted Edition (PSV, 2012): Adventure -> Action
UPDATE game_sales SET game_game_id = 10576
 WHERE game_rank = 14860 AND game_game_id = 9195;

-- Verifikasi (harus: 12528 -> Strategy, 14860 -> Action)
SELECT s.game_rank, g.game_title, gn.genre_name
  FROM game_sales s
  JOIN game g   ON s.game_game_id = g.game_id
  JOIN genre gn ON g.genre_genre_id = gn.genre_id
 WHERE s.game_rank IN (12528, 14860);

COMMIT;

-- Pemeriksaan tambahan (opsional): baris `game` yatim yang tidak dipakai game_sales.
-- Setelah patch di atas, sisa yatim yang diharapkan: 8894 (Steins;Gate: Hiyoku ... Action) dan 11924 (FACHRIE).
-- 8894 TIDAK bisa diaudit lewat CSV Anda karena baris ber-';' rusak di CSV; cek manual ke sumber Kaggle asli.
SELECT g.game_id, g.game_title, g.genre_genre_id
  FROM game g LEFT JOIN game_sales s ON s.game_game_id = g.game_id
 WHERE s.game_game_id IS NULL;

-- Angka acuan untuk naskah (dari DB, 16.598 baris):
SELECT ROUND(SUM(na_sales)/(SUM(na_sales)+SUM(eu_sales)+SUM(jp_sales)+SUM(other_sales))*100,1) AS NA_pct,
       ROUND(SUM(eu_sales)/(SUM(na_sales)+SUM(eu_sales)+SUM(jp_sales)+SUM(other_sales))*100,1) AS EU_pct,
       ROUND(SUM(jp_sales)/(SUM(na_sales)+SUM(eu_sales)+SUM(jp_sales)+SUM(other_sales))*100,1) AS JP_pct
  FROM game_sales;   -- harapan: 49.3 / 27.3 / 14.5
