"""
01_model_training.py
====================
Nepal Multimodal Drought Prediction Project
Step 1: Train all ML models, run ablation study, early-warning lead-time
        evaluation, LOEO cross-validation, and save results.

Run AFTER 00_data_extraction_audit.py

Output files
------------
models/rf_full_model.pkl               – Trained Random Forest (all features)
models/loeo_predictions.csv           – LOEO CV predictions per event
models/ablation_results.csv           – Ablation study (R²/RMSE/MAE per feature group)
models/lead_time_results.csv          – Early-warning lead-time performance
models/feature_importance.csv         – RF feature importance + correlations

Author : Auto-generated for manuscript preparation
Date   : 2026-09-17
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import netCDF4 as nc
from datetime import datetime

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PROC = '../data/processed/'
DATA_RAW  = '../data/raw/'
OUT_MOD   = '../models/'
os.makedirs(OUT_MOD, exist_ok=True)

YEARS = ['2015','2016','2018','2022','2024']
DISTS = ['Jhapa','Ilam','Bhojpur','Morang','Dhankuta','Sunsari']
BASE  = '../masterdata_of_agriculture/'   # raw NetCDF for lead-time features

# ── Load feature matrix ────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PROC+'feature_matrix.csv')
df['year'] = df['year'].astype(str)
print(f"Feature matrix loaded: {df.shape}")

# All predictive features
FEAT_COLS = [
    'precip_gs','precip_std','temp_gs','evap_gs',
    'spi_mean','spi_min','spi_drought_days',
    'sm_surface','sm_rootzone',
    'ndvi','ndvi_std','evi','ndwi','savi',
    'VV_dB','VH_dB','VV_VH',
    'clay','soc','cec','sand','phh2o','bdod','nitrogen',
    'sos_doy','pos_doy','eos_doy','ndvi_max_phen',
]
FEAT_COLS = [c for c in FEAT_COLS if c in df.columns]
TARGET    = 'yield_anomaly_pct'

X_all = df[FEAT_COLS].fillna(df[FEAT_COLS].mean()).values
y_all = df[TARGET].values
print(f"  Features: {len(FEAT_COLS)} | Target range: {y_all.min():.2f}–{y_all.max():.2f} %")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Leave-One-Event-Out CV
# ─────────────────────────────────────────────────────────────────────────────
def loeo_cv(X, y, years_col, model_cls=RandomForestRegressor,
            model_kwargs=None):
    if model_kwargs is None:
        model_kwargs = {'n_estimators':150,'random_state':42}
    all_true, all_pred, all_yr = [], [], []
    for held in YEARS:
        mask = (years_col == held).values
        X_tr, y_tr = X[~mask], y[~mask]
        X_te, y_te = X[mask],  y[mask]
        sc = StandardScaler().fit(X_tr)
        m  = model_cls(**model_kwargs).fit(sc.transform(X_tr), y_tr)
        yp = m.predict(sc.transform(X_te))
        all_true.extend(y_te); all_pred.extend(yp)
        all_yr.extend([held]*len(y_te))
    at, ap = np.array(all_true), np.array(all_pred)
    return (r2_score(at,ap),
            np.sqrt(mean_squared_error(at,ap)),
            mean_absolute_error(at,ap),
            at, ap, np.array(all_yr))


# ═════════════════════════════════════════════════════════════════════════════
# 1. TRAIN FULL MODEL (all data)
# ═════════════════════════════════════════════════════════════════════════════
print("\nTraining full RF model …")
sc_full = StandardScaler().fit(X_all)
rf_full = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
rf_full.fit(sc_full.transform(X_all), y_all)

with open(OUT_MOD+'rf_full_model.pkl','wb') as f:
    pickle.dump({'model':rf_full,'scaler':sc_full,'features':FEAT_COLS,'target':TARGET}, f)
print(f"  RF full model saved → {OUT_MOD}rf_full_model.pkl")


# ═════════════════════════════════════════════════════════════════════════════
# 2. LOEO CROSS-VALIDATION
# ═════════════════════════════════════════════════════════════════════════════
print("\nRunning LOEO cross-validation …")
loeo_rows = []
all_true_all, all_pred_all = [], []

for held_yr in YEARS:
    mask = (df['year']==held_yr).values
    X_tr, y_tr = X_all[~mask], y_all[~mask]
    X_te, y_te = X_all[mask],  y_all[mask]
    dists_te   = df.loc[mask,'district'].values
    sc = StandardScaler().fit(X_tr)
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(sc.transform(X_tr), y_tr)
    y_pred = rf.predict(sc.transform(X_te))
    all_true_all.extend(y_te); all_pred_all.extend(y_pred)
    r2_ev   = r2_score(y_te,y_pred)
    rmse_ev = np.sqrt(mean_squared_error(y_te,y_pred))
    mae_ev  = mean_absolute_error(y_te,y_pred)
    print(f"  Held-out {held_yr}: R²={r2_ev:.3f}  RMSE={rmse_ev:.3f}  MAE={mae_ev:.3f}")
    for di,d in enumerate(dists_te):
        loeo_rows.append({'held_out_year':held_yr,'district':d,
                          'y_true':y_te[di],'y_pred':y_pred[di],
                          'residual':y_pred[di]-y_te[di]})

df_loeo = pd.DataFrame(loeo_rows)
at = np.array(all_true_all); ap = np.array(all_pred_all)
print(f"\n  Pooled LOEO: R²={r2_score(at,ap):.3f}  RMSE={np.sqrt(mean_squared_error(at,ap)):.3f}")
df_loeo.to_csv(OUT_MOD+'loeo_predictions.csv', index=False)


# ═════════════════════════════════════════════════════════════════════════════
# 3. ABLATION STUDY
# ═════════════════════════════════════════════════════════════════════════════
print("\nRunning ablation study …")
FEAT_GROUPS = {
    'Satellite only':   ['ndvi','ndvi_std','evi','ndwi','savi','VV_dB','VH_dB','VV_VH'],
    'Climate only':     ['precip_gs','precip_std','temp_gs','evap_gs','spi_mean','spi_min','spi_drought_days'],
    'Soil only':        ['clay','soc','cec','sand','phh2o','bdod','nitrogen'],
    'Phenology only':   ['sos_doy','pos_doy','eos_doy','ndvi_max_phen'],
    'Sat + Climate':    ['ndvi','evi','VV_dB','VH_dB','precip_gs','temp_gs','spi_min','sm_surface'],
    'Sat + Soil':       ['ndvi','evi','VV_dB','VH_dB','clay','soc','cec','phh2o'],
    'Climate + Soil':   ['precip_gs','temp_gs','spi_min','sm_surface','sm_rootzone','clay','soc','cec'],
    'Sat + Climate + Soil': ['ndvi','evi','ndwi','VV_dB','VH_dB','precip_gs','temp_gs',
                              'spi_mean','spi_min','sm_surface','clay','soc','cec'],
    'Full multimodal':  FEAT_COLS,
}

abl_rows = []
for gname, feats in FEAT_GROUPS.items():
    avail = [f for f in feats if f in df.columns]
    X_g   = df[avail].fillna(df[avail].mean()).values
    r2,rmse,mae,_,_,_ = loeo_cv(X_g, y_all, df['year'])
    abl_rows.append({'group':gname,'n_features':len(avail),
                     'R2_LOEO':round(r2,4),'RMSE_LOEO':round(rmse,4),'MAE_LOEO':round(mae,4)})
    print(f"  {gname:<26}: R²={r2:.3f}  RMSE={rmse:.3f}  n_feat={len(avail)}")

df_abl = pd.DataFrame(abl_rows)
df_abl.to_csv(OUT_MOD+'ablation_results.csv', index=False)


# ═════════════════════════════════════════════════════════════════════════════
# 4. EARLY-WARNING LEAD-TIME ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
print("\nRunning lead-time analysis …")

def nc2pd(t_var):
    objs = nc.num2date(t_var[:], t_var.units, calendar='standard')
    return pd.to_datetime([datetime(d.year,d.month,d.day) for d in objs])

# Build lead-time precip/temp features
LEAD_WEEKS = [1,2,3,4,6,8]
lead_records = []
for yr in YEARS:
    for d in DISTS:
        fpath = os.path.join(BASE, yr, d, f'{d}_era5_land_{yr}.nc')
        ds = nc.Dataset(fpath)
        t  = nc2pd(ds.variables['time'])
        precip = ds.variables['precip'][:].astype(float)
        temp   = ds.variables['temp_2m'][:].astype(float)
        ds.close()
        doys = pd.DatetimeIndex(t).dayofyear
        peak_idx = int(np.argmin(np.abs(doys-200)))
        row = {'year':yr,'district':d}
        for nw in LEAD_WEEKS:
            win_end   = max(0, peak_idx - nw*7)
            win_start = max(0, win_end - 28)
            row[f'p_lead{nw}w'] = float(np.nanmean(precip[win_start:win_end])) if win_end>win_start else float(np.nanmean(precip))
            row[f't_lead{nw}w'] = float(np.nanmean(temp[win_start:win_end]))   if win_end>win_start else float(np.nanmean(temp))
        lead_records.append(row)

df_lead = pd.DataFrame(lead_records)
df_lead['year'] = df_lead['year'].astype(str)
df_all  = df.merge(df_lead, on=['year','district'])
y_tgt   = df_all['yield_anomaly_pct'].values

BASE_FEATS = [f for f in ['clay','soc','cec','sos_doy','pos_doy','ndvi_max_phen'] if f in df_all.columns]

lt_rows = []
for nw in LEAD_WEEKS:
    feats_lt = BASE_FEATS + [f'p_lead{nw}w',f't_lead{nw}w']
    avail     = [f for f in feats_lt if f in df_all.columns]
    X_lt = df_all[avail].fillna(0).values
    r2,rmse,mae,_,_,_ = loeo_cv(X_lt, y_tgt, df_all['year'])
    lt_rows.append({'lead_weeks':nw,'R2_LOEO':round(r2,4),'RMSE_LOEO':round(rmse,4),'MAE_LOEO':round(mae,4)})
    print(f"  Lead {nw}w: R²={r2:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}")

df_lt = pd.DataFrame(lt_rows)
df_lt.to_csv(OUT_MOD+'lead_time_results.csv', index=False)


# ═════════════════════════════════════════════════════════════════════════════
# 5. FEATURE IMPORTANCE
# ═════════════════════════════════════════════════════════════════════════════
print("\nComputing feature importance …")
sc_imp = StandardScaler().fit(X_all)
rf_imp = RandomForestRegressor(n_estimators=300, random_state=42)
rf_imp.fit(sc_imp.transform(X_all), y_all)

corrs, spearmans = [], []
for c in FEAT_COLS:
    arr = df[c].fillna(df[c].mean()).values
    r,_  = pearsonr(arr, y_all)
    rs,_ = spearmanr(arr, y_all)
    corrs.append(r); spearmans.append(rs)

df_imp = pd.DataFrame({
    'feature':   FEAT_COLS,
    'rf_importance': rf_imp.feature_importances_,
    'pearson_r': corrs,
    'spearman_r': spearmans,
}).sort_values('rf_importance', ascending=False)
df_imp.to_csv(OUT_MOD+'feature_importance.csv', index=False)

print("\nTop 10 features:")
print(df_imp[['feature','rf_importance','pearson_r']].head(10).to_string(index=False))
print("\n✅  01_model_training.py  COMPLETE\n")
