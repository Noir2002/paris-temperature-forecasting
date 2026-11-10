# Data

All temperatures in this project are the **monthly mean of daily maximum temperature**
(°C) at **Paris-Montsouris**, Météo-France station `75114001`
(lat 48.821667, lon 2.337833, altitude 75 m).

## Files

| File | What it is | In git? |
|---|---|---|
| `original/data2.csv` | The 24 values (2023-01 … 2024-12) supplied with the ISEP IF1205 assignment, copied unchanged from [petitlang/Projet_ACP](https://github.com/petitlang/Projet_ACP) at commit `9da990f` (sha256 `fa5468f2fb34e833a577d7ca54bec856274f664bc9e56ae00a140b572be2be88`). | yes |
| `montsouris_tx_monthly.csv` | Columns `date, TX, QTX, NBTX` for station 75114001, 1950-01 … 2026-10 (922 complete months, no gaps), extracted by `python -m src.data`. | yes |
| `raw/` | The official files as downloaded (see below). Re-created by `python -m src.data`. | no (`.gitignore`) |

## Source

**Météo-France, *Données climatologiques de base – mensuelles*** (data.gouv.fr dataset
`6569b3d7d193b4daf2b43edc`), Licence Ouverte / Open Licence v2.0.

- Dataset page: <https://www.data.gouv.fr/fr/datasets/donnees-climatologiques-de-base-mensuelles/>
- Files used (département 75):
  - `MENSQ_75_previous-1950-2024.csv.gz` – <https://meteofrance.s3.sbg.io.cloud.ovh.net/data/synchro_ftp/BASE/MENS/MENSQ_75_previous-1950-2024.csv.gz>
    (sha256 `d23939a14b38217c14cde2f2e6b61e86d218449cbb356bbcc2175bd9fd0ac809`)
  - `MENSQ_75_latest-2025-2026.csv.gz` – <https://meteofrance.s3.sbg.io.cloud.ovh.net/data/synchro_ftp/BASE/MENS/MENSQ_75_latest-2025-2026.csv.gz>
    (sha256 `850a58e3e39b9d4229f37ceb911a57a828f7269c045d5a2274ab7186171f0cb5`)
- Field description: `MENSQ_descriptif_champs.csv` –
  <https://meteofrance.s3.sbg.io.cloud.ovh.net/data/synchro_ftp/BASE/MENS/MENSQ_descriptif_champs.csv>
- **Downloaded: 2026-11-10 (06:14 UTC).** The `latest` file is refreshed daily and
  older files monthly, so a later download can differ slightly (late corrections,
  new months). `montsouris_tx_monthly.csv` is the frozen snapshot used for every result.

## Fields used (quoted from `MENSQ_descriptif_champs.csv`)

| Field | Official description | Use |
|---|---|---|
| `NUM_POSTE` | numéro Météo-France du poste sur 8 chiffres | select `75114001` |
| `AAAAMM` | mois | date |
| `TX` | moyenne mensuelle des températures maximales (TX) quotidiennes (en °C et 1/10) | **the target series** |
| `QTX` | code qualité de TX | all kept months have `QTX = 1` ("donnée validée") |
| `NBTX` | nombre de valeurs présentes de TX quotidienne | months kept only if `NBTX` = days in month |

So the assignment's "température maximale moyenne" is exactly field `TX`. The current,
still-running month (2026-11, 7 days) is dropped because it is incomplete.

## Verification (stage 1) – output of `python -m src.verify_data`

**1. Does `data2.csv` match Montsouris?** Yes – 24/24 values identical. No other
Paris station matches.

```
75106001 LUXEMBOURG               exact matches  0/24   max |diff| 1.3 °C
75107005 TOUR EIFFEL              exact matches  0/24   max |diff| 5.0 °C
75110001 LARIBOISIERE             exact matches  5/24   max |diff| 0.6 °C
75114001 PARIS-MONTSOURIS         exact matches 24/24   max |diff| 0.0 °C
75114007 PARIS-MONTSOURIS-DOUBLE  exact matches  2/24   max |diff| 0.6 °C
75116008 LONGCHAMP                exact matches  2/24   max |diff| 0.6 °C
```

**2. What were Jan–Apr 2025?** 7.5 / 8.6 / 14.6 / 20.0 °C – the values in the
assignment PDF. The values hard-coded in the original code (8 / 10 / 14 / 18,
introduced by commit `51648b7` "真实温度改正" on 2025-06-03) match no station in Paris.

```
                                        Jan   Feb   Mar   Apr
75106001       LUXEMBOURG               8.0   9.3  16.0  21.1
75107005       TOUR EIFFEL              5.9   6.4  11.7  16.2
75110001       LARIBOISIERE             8.0   8.8  14.3  19.5
75114001       PARIS-MONTSOURIS         7.5   8.6  14.6  20.0
75114007       PARIS-MONTSOURIS-DOUBLE  7.5   8.5  14.5  19.7
75116008       LONGCHAMP                7.9   8.9  14.5  19.5
assignment PDF                          7.5   8.6  14.6  20.0
original code                           8.0  10.0  14.0  18.0
```

All four Montsouris values have `QTX = 1` (validated) and complete `NBTX`.

**3. 1991–2020 normals (Jan–Apr)?** 7.6 / 8.8 / 12.8 / 16.6 °C. Two independent
confirmations:

- mean of the 30 monthly `TX` values 1991–2020 in the open data:
  7.61 / 8.83 / 12.79 / 16.56 (each month has 30 values);
- Météo-France *Fiche climatologique* for 75114001, "Statistiques 1991−2020 et records",
  row "Température maximale (moyenne en °C)": 7.6 / 8.8 / 12.8 / 16.6
  (edited 06/06/2025; <https://donneespubliques.meteofrance.fr/FichesClim/FICHECLIM_75114001.pdf>).
  This PDF is **not** redistributed here: its footer forbids redistribution without
  Météo-France's agreement. Only the four numbers are cited.

Full-year 1991–2020 means of `TX` (computed): 7.61, 8.83, 12.79, 16.56, 20.15, 23.39,
25.72, 25.61, 21.49, 16.49, 11.11, 8.01.

## Attribution

"Source : Météo-France – Données climatologiques de base mensuelles, Licence Ouverte v2.0."
`original/data2.csv` belongs to the ISEP IF1205 (2025) assignment material.
