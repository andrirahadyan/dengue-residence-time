# Dengue transmission with human residence times in two clusters

Code for the article *Daily Mobility Lowers the Dengue Threshold but Can Increase Infections: A Two-Cluster Residence-Time Model* (Andri Rahadyan, Universitas Indraprasta PGRI).

The scripts implement a deterministic SEIR–SEI dengue model for two clusters connected by a residence-time (Lagrangian) matrix, in which infected residents reduce their travel by a factor θ. They reproduce every number, table, and figure in the article:

- basic reproduction number R0 (closed form, checked against the numerical next generation matrix)
- residence-time threshold p*, symmetric and asymmetric cases
- local sensitivity indices, 730-day simulations
- global sensitivity analysis (Latin hypercube sampling with PRCC, and Sobol indices with SALib)

## Files

| File | Output |
|---|---|
| `model_waktu_tinggal.py` | Figures with Indonesian labels (`gambar1.png`–`gambar4_sensitivitas_global.png`) and all results in `hasil_waktu_tinggal.json` |
| `model_waktu_tinggal_en.py` | The same computations with English figure labels, saved in `en/` |

## How to run

```
pip install -r requirements.txt
python model_waktu_tinggal_en.py
```

Each script runs in about 10 seconds. Random seeds are fixed, so the results are reproducible.
