# Dengue transmission with human residence times in two clusters

[![DOI](https://zenodo.org/badge/1436781892.svg)](https://doi.org/10.5281/zenodo.23724284)

Code and data for the article *Daily Mobility Lowers the Dengue Threshold but Can Increase Infections: A Two-Cluster Residence-Time Model* (Andri Rahadyan, La Ode Sabran, Norma Muhtar, Ilham Dangu Rianjaya, Septian Wulandari).

The scripts implement a deterministic SEIR–SEI dengue model for two clusters connected by a residence-time (Lagrangian) matrix, in which infected residents reduce their travel by a factor θ. They reproduce every number, table, and figure in the article:

- basic reproduction number R0 (closed form, checked against the numerical next generation matrix)
- residence-time threshold p*, symmetric and asymmetric cases
- local sensitivity indices, 730-day simulations
- global sensitivity analysis (Latin hypercube sampling with PRCC, and Sobol indices with SALib)
- global-stability bound R̂0 for the disease-free equilibrium and numerical checks in the gap R0 < 1 ≤ R̂0
- existence and uniqueness of the endemic equilibrium (proof checked numerically) and its local stability (numerical)
- intervention scenarios (mobility restrictions, isolation of infected residents, vector control at equal effort)
- illustrative calibration to the 2024 dengue outbreak in DKI Jakarta and Banten

## Files

| File | Output | Article |
|---|---|---|
| `model_waktu_tinggal.py` | Figures with Indonesian labels and all results in `hasil_waktu_tinggal.json` | – |
| `model_waktu_tinggal_en.py` | R0, threshold, sensitivity, and simulations with English labels, saved in `en/` | Tables 3–4, Figures 1–3 and 5 |
| `revision_analysis.py` | Global-stability bound, endemic equilibrium, interventions; `en/results_revision.json`, `en/gambar5_interventions.png` | Section 3.3 (Theorem 3.3), Remark 3.6, Table 5, Figure 4 |
| `endemic_reduction_check.py` | Check that every positive equilibrium of the full model equals the fixed point of the map H (θ = 0); `en/results_endemic_check.json` | Section 3.4 (Propositions 3.4–3.5) |
| `calibration.py` | Fit to monthly cases, bootstrap, residence-time sensitivity, counterfactuals; `en/results_calibration.json`, `en/gambar6_calibration.png` | Table 6, Figure 6 |
| `data/opendengue_v1_3_dki_banten_monthly.csv` | Monthly reported dengue cases, DKI Jakarta and Banten, 2022–2024 | Input to `calibration.py` |

`revision_analysis.py`, `endemic_reduction_check.py`, and `calibration.py` reuse the model definitions in `model_waktu_tinggal_en.py`, so keep these files in the same folder.

## Data sources

- **Dengue cases:** OpenDengue version 1.3, monthly counts at provincial level (Admin1). Clarke et al. (2024), *A global dataset of publicly available dengue case count data*, Scientific Data 11, 296, https://doi.org/10.1038/s41597-024-03120-7. The OpenDengue data are licensed CC BY-SA.
- **Residence fractions:** Badan Pusat Statistik (2024), *Statistik Komuter Jabodetabek: Hasil Survei Komuter Jabodetabek 2023*, Tables 2, 22, and 25. Commuters between the Tangerang area (Tangerang Regency, Tangerang City, South Tangerang) and Jakarta give p21 = 0.0120 (Banten residents in Jakarta) and p12 = 0.0048 (Jakarta residents in Banten).
- **Population (2024):** Ditjen Dukcapil, Kementerian Dalam Negeri: DKI Jakarta 11.03 million, Banten 12.88 million.

## How to run

```
pip install -r requirements.txt
python model_waktu_tinggal_en.py
python revision_analysis.py
python endemic_reduction_check.py
python calibration.py
```

`model_waktu_tinggal_en.py` runs in about 10 seconds, `revision_analysis.py` in a few minutes, and `calibration.py` (40 bootstrap refits) in about 10 minutes. Random seeds are fixed, so the results are reproducible.

## How to cite

The code is archived on Zenodo: https://doi.org/10.5281/zenodo.23724284. This DOI always resolves to the latest version.
