# Datasets

Homing-pigeon GPS datasets used in this project. The `main` branch holds the
data deposit; analysis code and the interactive viewer live on the **`hv`** branch.

---

## 1. Flack et al. (2014) — multi-route homing  *(included here)*

**File:** `flack_2014_homing_routes_gps.zip` (≈22 MB, in this folder)

GPS tracks of homing pigeons released along three routes (`R1`, `R2`, `R3`),
26 birds each, ~6 release flights per bird (468 flights total). 1 Hz fixes.

Each CSV (`R<route>/<COND>_<BIRD>/<BIRD>_R<route>_<NN>.csv`) has columns:

```
Date, Time, Latitude, Longitude, Altitude, Speed, Course, Type, Distance, Essential
```

> Flack A., Guilford T., Biro D. (2014). *Learning multiple routes in homing
> pigeons.* Biology Letters 10(2): 20140119.

Unzip with:

```bash
unzip flack_2014_homing_routes_gps.zip -d flack_2014
```

---

## 2. Valentini et al. (2021) — collective exploration  *(NOT included — 1.1 GB)*

Too large for GitHub (100 MB file limit). Download directly from figshare:

- **DOI:** [10.6084/m9.figshare.14043362](https://doi.org/10.6084/m9.figshare.14043362)
- **Title:** *Data and code from: Naïve individuals promote collective
  exploration in homing pigeons* (Valentini, Pavlic, Walker, Pratt, Biro, Sasaki)

```bash
mkdir -p 2021
# 1.23 GB archive  (md5: 4f4a3cbe268562b14b19b04eaa325470)
curl -L -o 2021/figshare.zip https://ndownloader.figshare.com/files/31043605
unzip -q 2021/figshare.zip -x "__MACOSX/*" -d 2021_tmp
mv 2021_tmp/figshare/* 2021/ && rmdir 2021_tmp/figshare 2021_tmp
```

Raw GPS lives in `2021/data/raw/<Condition>_<ID>_<release>_bare.csv`
(headerless: `latitude, longitude, time-of-day-seconds`). Conditions:
**Solo**, **Pair** (fixed pairs), and transgenerational chains **Gen1–Gen5**.

> ⚠️ The raw CSVs record time-of-day but **no calendar date**, so historical
> weather cannot be matched to 2021 flights (confirmed against the SI).
