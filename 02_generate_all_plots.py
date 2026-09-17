"""
02_generate_all_plots.py
========================
Nepal Multimodal Drought Prediction Project
Step 2: Generate all 11 publication-quality figures (PNG + PDF + SVG)

Run AFTER 00_data_extraction_audit.py and 01_model_training.py

Figures generated
-----------------
FIG01 – Study Area & Maize Distribution
FIG02 – Drought Verification (5 events, 1 figure each)
FIG03 – Meteorological Evolution (5 events, 1 figure each)
FIG04 – Sentinel-2 Vegetation Response (3 sub-figures)
FIG05 – Sentinel-1 SAR Response (2 sub-figures)
FIG06 – Spatial Drought-Stress Maps (5 events, 1 figure each)
FIG07 – Yield Analysis
FIG08 – Multimodal Ablation Study
FIG09 – Early-Warning Lead-Time Performance
FIG10 – Feature Importance & Correlation
FIG11 – Cross-Event Generalisation (LOEO)

Author : Auto-generated for manuscript preparation
Date   : 2026-09-17
"""

import os, warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import netCDF4 as nc
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = '../masterdata_of_agriculture/'
DATA_PROC = '../data/processed/'
DATA_RAW  = '../data/raw/'
MODELS    = '../models/'
OUT_FIGS  = '../figures/'

YEARS  = ['2015','2016','2018','2022','2024']
DISTS  = ['Jhapa','Ilam','Bhojpur','Morang','Dhankuta','Sunsari']
COLORS = ['#2166ac','#d6604d','#1a9641','#7b2d8b','#e6821e']

EVENT_META = {
    '2015':{'color':'#2166ac','onset':'2015-06-15','peak':'2015-08-01','end':'2015-09-30',
             'ms':'2015-04-15','me':'2015-10-31','desc':'El Niño-induced pre-monsoon dry spell'},
    '2016':{'color':'#d6604d','onset':'2016-06-01','peak':'2016-07-15','end':'2016-09-30',
             'ms':'2016-04-15','me':'2016-09-30','desc':'Delayed monsoon onset; erratic distribution'},
    '2018':{'color':'#1a9641','onset':'2018-06-01','peak':'2018-07-20','end':'2018-09-30',
             'ms':'2018-04-15','me':'2018-09-30','desc':'Mid-season dry spell during grain-fill stage'},
    '2022':{'color':'#7b2d8b','onset':'2022-07-01','peak':'2022-08-01','end':'2022-09-30',
             'ms':'2022-06-01','me':'2022-09-30','desc':'Severe July–August precipitation deficit'},
    '2024':{'color':'#e6821e','onset':'2024-06-01','peak':'2024-07-15','end':'2024-09-30',
             'ms':'2024-02-01','me':'2024-09-30','desc':'Spring relay season heat and moisture stress'},
}

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':0.8,
                     'xtick.labelsize':8.5,'ytick.labelsize':8.5})

# ── Create figure subdirectories ──────────────────────────────────────────────
for sub in ['FIG01','FIG02','FIG03','FIG04','FIG05','FIG06','FIG07','FIG08','FIG09','FIG10','FIG11']:
    os.makedirs(os.path.join(OUT_FIGS, sub), exist_ok=True)

def save_fig(fig, outdir, fname, dpi=300):
    for ext in ['png','pdf','svg']:
        fig.savefig(os.path.join(outdir, f'{fname}.{ext}'),
                    dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✅  {fname}")

# ── Load data ─────────────────────────────────────────────────────────────────
df      = pd.read_csv(DATA_PROC+'feature_matrix.csv')
df['year'] = df['year'].astype(str)
df_era5 = pd.read_csv(DATA_RAW+'era5_daily_all.csv')
df_spi  = pd.read_csv(DATA_RAW+'spi_daily_all.csv')
df_smap = pd.read_csv(DATA_RAW+'smap_daily_all.csv')
df_s2   = pd.read_csv(DATA_RAW+'sentinel2_snapshots.csv')
df_s1   = pd.read_csv(DATA_RAW+'sentinel1_snapshots.csv')
df_loeo = pd.read_csv(MODELS+'loeo_predictions.csv')
df_abl  = pd.read_csv(MODELS+'ablation_results.csv')
df_lt   = pd.read_csv(MODELS+'lead_time_results.csv')
df_imp  = pd.read_csv(MODELS+'feature_importance.csv')
gdf     = gpd.read_file(BASE+'2022/district_boundaries.shp')
gdf_study = gdf[gdf['DISTRICT'].isin([d.upper() for d in DISTS])].copy()

def nc2pd(t_var):
    objs = nc.num2date(t_var[:], t_var.units, calendar='standard')
    return pd.to_datetime([datetime(d.year,d.month,d.day) for d in objs])

def get_era5_district_mean(yr):
    """Return district-averaged ERA5 daily time series for a given year."""
    sub = df_era5[df_era5['year'].astype(str)==yr].copy()
    sub['date'] = pd.to_datetime(sub['date'])
    return sub.groupby('date')[['precip_mm_d','temp_2m_C','evap_mm_d']].mean()

def get_spi_district_mean(yr):
    sub = df_spi[df_spi['year'].astype(str)==yr].copy()
    sub['date'] = pd.to_datetime(sub['date'])
    return sub.groupby('date')['spi'].mean()

def get_smap_district_mean(yr):
    sub = df_smap[df_smap['year'].astype(str)==yr].copy()
    sub['date'] = pd.to_datetime(sub['date'])
    return sub.groupby('date')[['sm_surface_m3m3','sm_rootzone_m3m3']].mean()

def roll(series, w):
    return series.rolling(w, center=True, min_periods=3).mean()

# ══════════════════════════════════════════════════════════════════════════════
# FIG01 – Study Area
# ══════════════════════════════════════════════════════════════════════════════
print("\n── FIG01 Study Area ──")
MAIZE = {'JHAPA':{'area':12400,'prod':34720},'ILAM':{'area':15200,'prod':38000},
         'BHOJPUR':{'area':16100,'prod':35420},'MORANG':{'area':11800,'prod':33040},
         'DHANKUTA':{'area':18300,'prod':47580},'SUNSARI':{'area':9500,'prod':26600}}
md = pd.DataFrame.from_dict(MAIZE,orient='index').reset_index().rename(columns={'index':'DISTRICT'})
gdf_s = gdf_study.merge(md, on='DISTRICT')
b = gdf_s.total_bounds; pad=0.20

fig, axes = plt.subplots(2,2,figsize=(14,10))
fig.patch.set_facecolor('white')
fig.subplots_adjust(hspace=0.35,wspace=0.30,left=0.06,right=0.97,top=0.92,bottom=0.07)

ax=axes[0,0]
gdf.plot(ax=ax,color='#f0ede8',edgecolor='#888',linewidth=0.35)
gdf_s.plot(ax=ax,color='#c0392b',edgecolor='#7b1f1f',linewidth=1.0)
ax.set_xlim(79.9,88.5); ax.set_ylim(26.0,30.6)
ax.set_title('(a) Nepal – Study Districts (red)',fontsize=11,fontweight='bold',pad=5)
ax.set_xlabel('Longitude (°E)',fontsize=9); ax.set_ylabel('Latitude (°N)',fontsize=9)
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
ax.legend(handles=[mpatches.Patch(color='#c0392b',label='Study districts (n=6)'),
                   mpatches.Patch(color='#f0ede8',edgecolor='#888',label='Other districts')],
          fontsize=8,loc='lower right',framealpha=0.9)
ax.annotate('N',xy=(80.3,30.2),fontsize=11,ha='center',fontweight='bold')
ax.annotate('',xy=(80.3,30.45),xytext=(80.3,30.2),arrowprops=dict(arrowstyle='->',color='k',lw=1.5))
ax.plot([85.5,86.6],[26.35,26.35],'k-',lw=2.5); ax.text(86.05,26.18,'~122 km',ha='center',fontsize=7.5)

colors6=['#2166ac','#74add1','#fee090','#f46d43','#d73027','#a50026']
ax=axes[0,1]
gdf_s.plot(ax=ax,color=colors6,edgecolor='#1a1a1a',linewidth=1.0)
for _,row in gdf_s.iterrows():
    cx,cy=row.geometry.centroid.x,row.geometry.centroid.y
    ax.text(cx,cy,row['DISTRICT'].capitalize(),ha='center',va='center',fontsize=8.5,fontweight='bold',
            bbox=dict(facecolor='white',alpha=0.72,edgecolor='none',pad=1.5))
ax.set_xlim(b[0]-pad,b[2]+pad); ax.set_ylim(b[1]-pad,b[3]+pad)
ax.set_title('(b) Study Districts – Province 1 (Koshi)',fontsize=11,fontweight='bold',pad=5)
ax.set_xlabel('Longitude (°E)',fontsize=9); ax.set_ylabel('Latitude (°N)',fontsize=9)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5)); ax.yaxis.set_major_locator(ticker.MultipleLocator(0.3))
xl2,yl2=ax.get_xlim(),ax.get_ylim()
ax.annotate('N',xy=(xl2[0]+0.14,yl2[1]-0.18),fontsize=10,ha='center',fontweight='bold')
ax.annotate('',xy=(xl2[0]+0.14,yl2[1]-0.06),xytext=(xl2[0]+0.14,yl2[1]-0.18),
            arrowprops=dict(arrowstyle='->',color='k',lw=1.5))
ax.plot([xl2[0]+0.1,xl2[0]+0.65],[yl2[0]+0.08,yl2[0]+0.08],'k-',lw=2.5)
ax.text(xl2[0]+0.375,yl2[0]+0.02,'~60 km',ha='center',fontsize=7.5)

ax=axes[1,0]
gdf_s.plot(ax=ax,column='area',cmap='YlGn',vmin=8000,vmax=20000,edgecolor='#1a1a1a',linewidth=1.0)
for _,row in gdf_s.iterrows():
    cx,cy=row.geometry.centroid.x,row.geometry.centroid.y
    ax.text(cx,cy,f"{int(row['area']):,}\nha",ha='center',va='center',fontsize=8,fontweight='bold',
            bbox=dict(facecolor='white',alpha=0.75,edgecolor='none',pad=1.5))
ax.set_xlim(b[0]-pad,b[2]+pad); ax.set_ylim(b[1]-pad,b[3]+pad)
ax.set_title('(c) Maize Cultivated Area (ha)',fontsize=11,fontweight='bold',pad=5)
ax.set_xlabel('Longitude (°E)',fontsize=9); ax.set_ylabel('Latitude (°N)',fontsize=9)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5)); ax.yaxis.set_major_locator(ticker.MultipleLocator(0.3))
div=make_axes_locatable(ax); cax=div.append_axes('right',size='4%',pad=0.06)
sm_cb=plt.cm.ScalarMappable(cmap='YlGn',norm=plt.Normalize(8000,20000)); sm_cb.set_array([])
cb=plt.colorbar(sm_cb,cax=cax); cb.set_label('Area (ha)',fontsize=8); cb.ax.tick_params(labelsize=7)

ax=axes[1,1]
dnames=[d.capitalize() for d in DISTS]
areas=[MAIZE[d.upper()]['area'] for d in DISTS]; prods=[MAIZE[d.upper()]['prod'] for d in DISTS]
x=np.arange(6); w=0.35
ax.bar(x-w/2,areas,w,label='Area (ha)',color='#4575b4',edgecolor='#1a1a1a',lw=0.7)
ax2r=ax.twinx()
ax2r.bar(x+w/2,prods,w,label='Production (mt)',color='#d6604d',edgecolor='#1a1a1a',lw=0.7)
ax.set_xticks(x); ax.set_xticklabels(dnames,fontsize=9)
ax.set_ylabel('Area (ha)',fontsize=9,color='#4575b4'); ax2r.set_ylabel('Production (mt)',fontsize=9,color='#d6604d')
ax.set_title('(d) Maize Area and Production',fontsize=11,fontweight='bold',pad=5)
ax.spines[['top']].set_visible(False); ax2r.spines[['top']].set_visible(False)
ax.legend(loc='upper left',fontsize=8,framealpha=0.9); ax2r.legend(loc='upper right',fontsize=8,framealpha=0.9)

fig.suptitle('Fig. 1. Study Area: Six Maize-Growing Districts of Eastern Nepal (Province 1 – Koshi)\n'
             'Five drought events: 2015, 2016, 2018, 2022, 2024 | CRS: WGS84 EPSG:4326',
             fontsize=11,fontweight='bold',y=0.98)
save_fig(fig, os.path.join(OUT_FIGS,'FIG01'), 'FIG01_Study_Area')


# ══════════════════════════════════════════════════════════════════════════════
# FIG02 – Drought Verification (one figure per event)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── FIG02 Drought Verification ──")
for yr, ev in EVENT_META.items():
    col=ev['color']
    onset=pd.to_datetime(ev['onset']); peak=pd.to_datetime(ev['peak'])
    end_d=pd.to_datetime(ev['end']);   ms=pd.to_datetime(ev['ms']); me=pd.to_datetime(ev['me'])

    era5  = get_era5_district_mean(yr)
    spi_s = get_spi_district_mean(yr)
    smap  = get_smap_district_mean(yr)

    gs_e = (era5.index.dayofyear>=91)&(era5.index.dayofyear<=273)
    p_ref= era5.loc[gs_e,'precip_mm_d'].mean() if gs_e.any() else era5['precip_mm_d'].mean()
    t_ref= era5.loc[gs_e,'temp_2m_C'].mean()   if gs_e.any() else era5['temp_2m_C'].mean()
    gs_s = (smap.index.dayofyear>=91)&(smap.index.dayofyear<=273)
    sm_ref=smap.loc[gs_s,'sm_surface_m3m3'].mean() if gs_s.any() else smap['sm_surface_m3m3'].mean()

    p_anom  = era5['precip_mm_d'] - p_ref
    t_anom  = era5['temp_2m_C']   - t_ref
    sm_anom = smap['sm_surface_m3m3'] - sm_ref

    fig,axes=plt.subplots(4,1,figsize=(13,14))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(hspace=0.50,left=0.10,right=0.97,top=0.91,bottom=0.06)

    def shade(ax):
        ax.axvspan(ms,me,alpha=0.07,color='#33a02c',zorder=0)
        ax.axvspan(onset,end_d,alpha=0.14,color=col,zorder=1)
        ax.axvline(onset,color=col,lw=1.3,ls='--',alpha=0.85,zorder=3)
        ax.axvline(peak, color=col,lw=2.2,ls='-', alpha=0.95,zorder=4)
        ax.axvline(end_d,color=col,lw=1.3,ls='--',alpha=0.85,zorder=3)
        ax.axhline(0,color='#444',lw=0.9,ls=':')
        ax.spines[['top','right']].set_visible(False)

    ax=axes[0]
    ax.bar(era5.index,p_anom,color=np.where(p_anom>=0,'#4393c3','#d6604d'),width=1,alpha=0.55,zorder=2)
    ax.plot(roll(p_anom,7).index,roll(p_anom,7).values,color='#08306b',lw=1.8,zorder=5,label='7-day mean')
    shade(ax); ax.set_ylabel('Precip. anomaly (mm day⁻¹)',fontsize=9.5)
    ax.set_title('(a) Precipitation Anomaly',fontsize=10.5,fontweight='bold',pad=4)
    ax.legend(fontsize=8,loc='upper right',framealpha=0.9); yabs=max(abs(p_anom.min()),abs(p_anom.max()))
    ax.set_ylim(-yabs*1.3,yabs*1.45)

    ax=axes[1]
    ax.fill_between(era5.index,t_anom,0,where=t_anom>=0,color='#d6604d',alpha=0.45,label='Warm anomaly')
    ax.fill_between(era5.index,t_anom,0,where=t_anom<0, color='#4393c3',alpha=0.45,label='Cool anomaly')
    ax.plot(roll(t_anom,7).index,roll(t_anom,7).values,color='#8b0000',lw=1.8,zorder=5,label='7-day mean')
    shade(ax); ax.set_ylabel('Temp. anomaly (°C)',fontsize=9.5)
    ax.set_title('(b) Temperature Anomaly',fontsize=10.5,fontweight='bold',pad=4)
    ax.legend(fontsize=8,ncol=3,loc='upper right',framealpha=0.9)

    ax=axes[2]
    ax.fill_between(smap.index,sm_anom,0,where=sm_anom>=0,color='#74add1',alpha=0.55,label='Wet')
    ax.fill_between(smap.index,sm_anom,0,where=sm_anom<0, color='#d73027',alpha=0.50,label='Dry')
    ax.plot(roll(sm_anom,7).index,roll(sm_anom,7).values,color='#023858',lw=1.8,zorder=5,label='7-day mean')
    shade(ax); ax.set_ylabel('SM anomaly (m³ m⁻³)',fontsize=9.5)
    ax.set_title('(c) Soil-Moisture Anomaly (SMAP)',fontsize=10.5,fontweight='bold',pad=4)
    ax.legend(fontsize=8,ncol=3,loc='upper right',framealpha=0.9)

    ax=axes[3]
    ax.fill_between(spi_s.index,spi_s,0,where=spi_s>=0,color='#4393c3',alpha=0.50,label='SPI>0')
    ax.fill_between(spi_s.index,spi_s,0,where=spi_s<0, color='#d6604d',alpha=0.65,label='SPI<0')
    ax.plot(roll(spi_s,14).index,roll(spi_s,14).values,color='#1a1a1a',lw=1.8,zorder=5,label='14-day mean')
    for th,lc,lb in [(-1.0,'#f4a582','Moderate'),(-1.5,'#d6604d','Severe'),(-2.0,'#8b0000','Extreme')]:
        ax.axhline(th,color=lc,lw=1.4,ls='--',label=f'{lb} ({th})')
    shade(ax); ax.set_ylabel('SPI (–)',fontsize=9.5)
    ax.set_title('(d) SPI and Drought Severity Thresholds',fontsize=10.5,fontweight='bold',pad=4)
    ax.legend(fontsize=7.8,ncol=3,loc='upper left',framealpha=0.9)
    ax.set_ylim(min(-0.3,float(spi_s.min())*1.3),max(0.5,float(spi_s.max())*1.3))
    spi_min=float(spi_s.min())
    cls=('Extreme' if spi_min<=-2 else 'Severe' if spi_min<=-1.5 else
         'Moderate' if spi_min<=-1 else 'Near-normal / mild')
    ax.text(0.99,0.04,f'SPI_min={spi_min:.2f} ({cls} drought)',transform=ax.transAxes,
            ha='right',fontsize=8.5,color='#8b0000',fontweight='bold',
            bbox=dict(facecolor='#fff0f0',edgecolor='#d6604d',pad=3,alpha=0.95))

    stats=(f'Precip. anom. min (7d)={float(roll(p_anom,7).min()):.1f} mm d⁻¹  |  '
           f'Temp. anom. max (7d)=+{float(roll(t_anom,7).max()):.1f} °C  |  '
           f'SM anom. min={float(sm_anom.min()):.3f} m³ m⁻³  |  SPI_min={spi_min:.2f}')
    fig.text(0.5,0.925,stats,ha='center',fontsize=8.5,
             bbox=dict(facecolor='#f0f4fa',edgecolor='#aaaacc',alpha=0.96,pad=4))

    leg_h=[mpatches.Patch(color=col,alpha=0.25,label='Drought period'),
           mpatches.Patch(color='#33a02c',alpha=0.2,label='Maize growing season'),
           plt.Line2D([0],[0],color=col,lw=2.0,ls='-',label='Drought peak'),
           plt.Line2D([0],[0],color=col,lw=1.3,ls='--',label='Onset/end')]
    fig.legend(handles=leg_h,loc='lower center',ncol=4,fontsize=9,
               bbox_to_anchor=(0.5,0.005),framealpha=0.95,edgecolor='#aaa')
    fig.suptitle(f'Fig. 2. Drought Verification — {yr} Event\n{ev["desc"]}\n'
                 f'District average | Province 1, Eastern Nepal',fontsize=12,fontweight='bold',y=0.997)
    save_fig(fig, os.path.join(OUT_FIGS,'FIG02'), f'FIG02_{yr}_Drought_Verification')


# ══════════════════════════════════════════════════════════════════════════════
# FIG03 – Meteorological Evolution (one figure per event, 5 panels)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── FIG03 Meteorological Evolution ──")
for yr, ev in EVENT_META.items():
    col=ev['color']
    onset=pd.to_datetime(ev['onset']); peak=pd.to_datetime(ev['peak'])
    end_d=pd.to_datetime(ev['end']);   ms=pd.to_datetime(ev['ms']); me=pd.to_datetime(ev['me'])

    era5 = get_era5_district_mean(yr)
    spi_s= get_spi_district_mean(yr)
    smap = get_smap_district_mean(yr)

    gs_e=(era5.index.dayofyear>=91)&(era5.index.dayofyear<=273)
    p_ref=era5.loc[gs_e,'precip_mm_d'].mean() if gs_e.any() else era5['precip_mm_d'].mean()
    t_ref=era5.loc[gs_e,'temp_2m_C'].mean()   if gs_e.any() else era5['temp_2m_C'].mean()
    e_ref=era5.loc[gs_e,'evap_mm_d'].mean()   if gs_e.any() else era5['evap_mm_d'].mean()
    gs_s=(smap.index.dayofyear>=91)&(smap.index.dayofyear<=273)
    sm_ref=smap.loc[gs_s,'sm_surface_m3m3'].mean() if gs_s.any() else smap['sm_surface_m3m3'].mean()

    fig,axes=plt.subplots(5,1,figsize=(13,17))
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(hspace=0.48,left=0.10,right=0.97,top=0.93,bottom=0.05)

    def shade(ax):
        ax.axvspan(ms,me,alpha=0.07,color='#33a02c',zorder=0)
        ax.axvspan(onset,end_d,alpha=0.13,color=col,zorder=1)
        ax.axvline(onset,color=col,lw=1.3,ls='--',alpha=0.8,zorder=3)
        ax.axvline(peak, color=col,lw=2.0,ls='-', alpha=0.95,zorder=4)
        ax.axvline(end_d,color=col,lw=1.3,ls='--',alpha=0.8,zorder=3)
        ax.spines[['top','right']].set_visible(False)

    # (a) Precipitation
    ax=axes[0]; p=era5['precip_mm_d']
    ax.bar(era5.index,p,color='#4393c3',width=1,alpha=0.55,zorder=2)
    ax.plot(roll(p,7).index,roll(p,7).values,color='#08306b',lw=1.8,zorder=5,label='7-day mean')
    ax.axhline(p_ref,color='#666',lw=0.9,ls=':',label=f'GS mean={p_ref:.1f} mm d⁻¹')
    shade(ax); ax.set_ylabel('Precipitation\n(mm day⁻¹)',fontsize=9.5)
    ax.set_title('(a) Daily Precipitation',fontsize=10.5,fontweight='bold',pad=4)
    ax.legend(fontsize=8,ncol=2,loc='upper right',framealpha=0.9)

    # (b) Temperature
    ax=axes[1]; t=era5['temp_2m_C']
    ax.plot(era5.index,t,color='#f4a582',lw=0.7,alpha=0.55)
    ax.plot(roll(t,7).index,roll(t,7).values,color='#d6604d',lw=1.8,zorder=5,label='7-day mean')
    ax.axhline(t_ref,color='#666',lw=0.9,ls=':',label=f'GS mean={t_ref:.1f} °C')
    shade(ax); ax.set_ylabel('Temperature\n(°C)',fontsize=9.5)
    ax.set_title('(b) 2-m Air Temperature',fontsize=10.5,fontweight='bold',pad=4)
    ax.legend(fontsize=8,ncol=2,loc='upper right',framealpha=0.9)

    # (c) Evapotranspiration
    ax=axes[2]; e=era5['evap_mm_d']
    ax.fill_between(era5.index,e,alpha=0.40,color='#74add1')
    ax.plot(roll(e,7).index,roll(e,7).values,color='#023858',lw=1.8,zorder=5,label='7-day mean')
    ax.axhline(e_ref,color='#666',lw=0.9,ls=':',label=f'GS mean={e_ref:.1f} mm d⁻¹')
    shade(ax); ax.set_ylabel('Evapotranspiration\n(mm day⁻¹)',fontsize=9.5)
    ax.set_title('(c) Actual Evapotranspiration (ERA5-Land)',fontsize=10.5,fontweight='bold',pad=4)
    ax.legend(fontsize=8,ncol=2,loc='upper right',framealpha=0.9)

    # (d) Soil moisture
    ax=axes[3]; sm=smap['sm_surface_m3m3']
    ax.fill_between(smap.index,sm,sm_ref,where=sm>=sm_ref,color='#74add1',alpha=0.55,label='Above GS mean')
    ax.fill_between(smap.index,sm,sm_ref,where=sm< sm_ref,color='#d73027',alpha=0.50,label='Below GS mean')
    ax.plot(roll(sm,7).index,roll(sm,7).values,color='#023858',lw=1.8,zorder=5,label='7-day mean')
    ax.axhline(sm_ref,color='#666',lw=0.9,ls=':',label=f'GS mean={sm_ref:.3f} m³ m⁻³')
    shade(ax); ax.set_ylabel('Soil moisture\n(m³ m⁻³)',fontsize=9.5)
    ax.set_title('(d) Surface Soil Moisture (SMAP L4)',fontsize=10.5,fontweight='bold',pad=4)
    ax.legend(fontsize=8,ncol=4,loc='upper right',framealpha=0.9)

    # (e) SPI
    ax=axes[4]
    ax.fill_between(spi_s.index,spi_s,0,where=spi_s>=0,color='#4393c3',alpha=0.50)
    ax.fill_between(spi_s.index,spi_s,0,where=spi_s<0, color='#d6604d',alpha=0.65)
    ax.plot(roll(spi_s,14).index,roll(spi_s,14).values,color='#1a1a1a',lw=1.8,zorder=5,label='14-day mean')
    for th,lc,lb in [(-1.0,'#f4a582','Moderate'),(-1.5,'#d6604d','Severe'),(-2.0,'#8b0000','Extreme')]:
        ax.axhline(th,color=lc,lw=1.4,ls='--',label=f'{lb} (SPI={th})')
    shade(ax); ax.set_ylabel('SPI (–)',fontsize=9.5)
    ax.set_title('(e) SPI and Drought Severity Classification',fontsize=10.5,fontweight='bold',pad=4)
    ax.legend(fontsize=7.8,ncol=4,loc='upper left',framealpha=0.9)
    ax.set_ylim(min(-0.3,float(spi_s.min())*1.3),max(0.5,float(spi_s.max())*1.3))

    leg_h=[mpatches.Patch(color=col,alpha=0.25,label='Drought period'),
           mpatches.Patch(color='#33a02c',alpha=0.2,label='Maize season'),
           plt.Line2D([0],[0],color=col,lw=2.0,ls='-',label='Drought peak'),
           plt.Line2D([0],[0],color=col,lw=1.3,ls='--',label='Onset/end')]
    fig.legend(handles=leg_h,loc='lower center',ncol=4,fontsize=9,
               bbox_to_anchor=(0.5,0.003),framealpha=0.95,edgecolor='#aaa')
    fig.suptitle(f'Fig. 3. Meteorological Evolution — {yr} Drought Event\n'
                 f'District average | Province 1, Eastern Nepal',fontsize=12,fontweight='bold',y=0.997)
    save_fig(fig, os.path.join(OUT_FIGS,'FIG03'), f'FIG03_Meteorological_{yr}')


# ══════════════════════════════════════════════════════════════════════════════
# FIG04 – Sentinel-2 (3 panels)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── FIG04 Sentinel-2 ──")
# FIG04A: 5×6 NDVI spatial maps
fig,axes=plt.subplots(5,6,figsize=(16,13))
fig.patch.set_facecolor('white'); fig.subplots_adjust(hspace=0.25,wspace=0.07,left=0.06,right=0.96,top=0.92,bottom=0.04)
for i,yr in enumerate(YEARS):
    for j,d in enumerate(DISTS):
        ax=axes[i,j]
        arr=np.load(os.path.join(BASE,yr,d,f'{d}_sentinel2.npy'),allow_pickle=True).astype(float)/10000
        R,NIR=arr[2],arr[3]; NDVI=np.clip((NIR-R)/(NIR+R+1e-8),-1,1)
        ax.imshow(NDVI,cmap='RdYlGn',vmin=-0.3,vmax=0.4,interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        if i==0: ax.set_title(d.capitalize(),fontsize=9,fontweight='bold',pad=3)
        if j==0: ax.set_ylabel(yr,fontsize=9,fontweight='bold',rotation=90,labelpad=4)
        ax.text(2,96,f'μ={NDVI.mean():.3f}',color='white',fontsize=6,fontweight='bold',
                va='bottom',bbox=dict(facecolor='#1a1a1a',alpha=0.65,edgecolor='none',pad=1.5))
cbar_ax=fig.add_axes([0.97,0.07,0.012,0.82])
sm=plt.cm.ScalarMappable(cmap='RdYlGn',norm=plt.Normalize(-0.3,0.4)); sm.set_array([])
cb=fig.colorbar(sm,cax=cbar_ax); cb.set_label('NDVI (–)',fontsize=10,fontweight='bold'); cb.ax.tick_params(labelsize=8)
fig.suptitle('Fig. 4a. Sentinel-2 NDVI Spatial Distribution | 5 Events × 6 Districts | 100×100 px',fontsize=10.5,fontweight='bold',y=0.98)
save_fig(fig, os.path.join(OUT_FIGS,'FIG04'), 'FIG04A_S2_NDVI_Maps')

# FIG04B: box plots
idx_names=['NDVI','EVI','NDWI','SAVI']
fig,axes=plt.subplots(2,2,figsize=(13,9))
fig.patch.set_facecolor('white'); fig.subplots_adjust(hspace=0.38,wspace=0.32,left=0.08,right=0.97,top=0.91,bottom=0.09)
for pi,idx_name in enumerate(idx_names):
    ax=axes[pi//2,pi%2]; data_box=[]; means=[]
    for yr in YEARS:
        per_d=[]
        for d in DISTS:
            arr=np.load(os.path.join(BASE,yr,d,f'{d}_sentinel2.npy'),allow_pickle=True).astype(float)/10000
            B,G,R,NIR=arr[0],arr[1],arr[2],arr[3]; eps=1e-8
            m={'NDVI':np.clip((NIR-R)/(NIR+R+eps),-1,1),'EVI':np.clip(2.5*(NIR-R)/(NIR+6*R-7.5*B+1+eps),-1,2),
               'NDWI':np.clip((G-NIR)/(G+NIR+eps),-1,1),'SAVI':np.clip(1.5*(NIR-R)/(NIR+R+0.5+eps),-1,1)}
            per_d.extend(m[idx_name].flatten()[::50].tolist())
        data_box.append(per_d); means.append(np.nanmean(per_d))
    bp=ax.boxplot(data_box,positions=range(5),widths=0.5,patch_artist=True,
                  medianprops={'color':'#1a1a1a','lw':2},whiskerprops={'lw':1.2},capprops={'lw':1.2},
                  flierprops={'marker':'.','ms':3,'alpha':0.3})
    for patch,color in zip(bp['boxes'],COLORS): patch.set_facecolor(color); patch.set_alpha(0.75)
    ax.scatter(range(5),means,color='#1a1a1a',s=60,zorder=5,marker='D',label='Mean')
    ax.plot(range(5),means,color='#1a1a1a',lw=1.5,ls='--',alpha=0.7)
    ax.axhline(np.mean(means),color='gray',lw=0.9,ls=':',label=f'5-event mean={np.mean(means):.3f}')
    ax.set_xticks(range(5)); ax.set_xticklabels(YEARS,fontsize=9)
    ax.set_xlabel('Drought Event Year',fontsize=9); ax.set_ylabel(f'{idx_name} (–)',fontsize=9)
    ax.set_title(f'({["a","b","c","d"][pi]}) {idx_name}',fontsize=11,fontweight='bold',pad=4)
    ax.legend(fontsize=8,framealpha=0.9); ax.spines[['top','right']].set_visible(False)
leg=[plt.Rectangle((0,0),1,1,color=c,alpha=0.75,label=yr) for c,yr in zip(COLORS,YEARS)]
fig.legend(handles=leg,loc='lower center',ncol=5,fontsize=9,bbox_to_anchor=(0.5,0.01),framealpha=0.9)
fig.suptitle('Fig. 4b. Sentinel-2 Vegetation Indices per Drought Event (pixel distribution)',fontsize=10.5,fontweight='bold',y=0.98)
save_fig(fig, os.path.join(OUT_FIGS,'FIG04'), 'FIG04B_S2_Index_Comparison')

# FIG04C: district NDVI bar chart
fig,ax=plt.subplots(figsize=(11,5.5))
fig.patch.set_facecolor('white'); fig.subplots_adjust(left=0.08,right=0.97,top=0.88,bottom=0.12)
x=np.arange(len(DISTS)); w=0.15
for i,yr in enumerate(YEARS):
    vals=[]
    for d in DISTS:
        sub=df_s2[(df_s2['year'].astype(str)==yr)&(df_s2['district']==d)]
        vals.append(float(sub['NDVI_mean'].iloc[0]) if len(sub)>0 else np.nan)
    ax.bar(x+i*w,vals,w,label=yr,color=COLORS[i],edgecolor='#1a1a1a',lw=0.6,alpha=0.85)
ax.set_xticks(x+2*w); ax.set_xticklabels([d.capitalize() for d in DISTS],fontsize=10)
ax.set_ylabel('Mean NDVI (–)',fontsize=10); ax.set_xlabel('District',fontsize=10)
ax.set_title('(c) Mean NDVI per District and Drought Event Year',fontsize=11,fontweight='bold',pad=5)
ax.legend(title='Event year',fontsize=9,ncol=5,framealpha=0.9)
ax.spines[['top','right']].set_visible(False)
fig.suptitle('Fig. 4c. District-Level NDVI Across Five Drought Events',fontsize=11,fontweight='bold',y=0.98)
save_fig(fig, os.path.join(OUT_FIGS,'FIG04'), 'FIG04C_S2_District_NDVI')


# ══════════════════════════════════════════════════════════════════════════════
# FIG05 – Sentinel-1 (2 panels)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── FIG05 Sentinel-1 ──")
# FIG05A: VV maps
fig,axes=plt.subplots(5,6,figsize=(16,13))
fig.patch.set_facecolor('white'); fig.subplots_adjust(hspace=0.25,wspace=0.07,left=0.06,right=0.96,top=0.92,bottom=0.04)
for i,yr in enumerate(YEARS):
    for j,d in enumerate(DISTS):
        ax=axes[i,j]
        arr=np.load(os.path.join(BASE,yr,d,f'{d}_sentinel1.npy'),allow_pickle=True).astype(float)/10000
        VV=10*np.log10(np.clip(arr[0],1e-10,None))
        ax.imshow(VV,cmap='Greys_r',vmin=-20,vmax=-5,interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        if i==0: ax.set_title(d.capitalize(),fontsize=9,fontweight='bold',pad=3)
        if j==0: ax.set_ylabel(yr,fontsize=9,fontweight='bold',rotation=90,labelpad=4)
        ax.text(2,96,f'VV={VV.mean():.1f}',color='white',fontsize=5.5,fontweight='bold',
                va='bottom',bbox=dict(facecolor='#1a1a1a',alpha=0.65,edgecolor='none',pad=1.5))
cbar_ax=fig.add_axes([0.97,0.07,0.012,0.82])
sm=plt.cm.ScalarMappable(cmap='Greys_r',norm=plt.Normalize(-20,-5)); sm.set_array([])
cb=fig.colorbar(sm,cax=cbar_ax); cb.set_label('VV backscatter (dB)',fontsize=10,fontweight='bold'); cb.ax.tick_params(labelsize=8)
fig.suptitle('Fig. 5a. Sentinel-1 VV Backscatter (dB) | 5 Events × 6 Districts | 100×100 px',fontsize=10.5,fontweight='bold',y=0.98)
save_fig(fig, os.path.join(OUT_FIGS,'FIG05'), 'FIG05A_S1_VV_Maps')

# FIG05B: box plots
fig,axes=plt.subplots(1,3,figsize=(15,5.5))
fig.patch.set_facecolor('white'); fig.subplots_adjust(wspace=0.32,left=0.07,right=0.97,top=0.87,bottom=0.13)
for ci,(col_name,label) in enumerate([('VV_dB_mean','VV (dB)'),('VH_dB_mean','VH (dB)'),('VV_VH_ratio_dB_mean','VV–VH (dB)')]):
    ax=axes[ci]; data_box=[]; means=[]
    for yr in YEARS:
        sub=df_s1[df_s1['year'].astype(str)==yr]; per_d=sub[col_name].tolist(); data_box.append(per_d); means.append(np.mean(per_d))
    bp=ax.boxplot(data_box,positions=range(5),widths=0.5,patch_artist=True,
                  medianprops={'color':'#1a1a1a','lw':2},whiskerprops={'lw':1.2},capprops={'lw':1.2})
    for patch,color in zip(bp['boxes'],COLORS): patch.set_facecolor(color); patch.set_alpha(0.75)
    ax.scatter(range(5),means,color='#1a1a1a',s=60,zorder=5,marker='D')
    ax.plot(range(5),means,color='#1a1a1a',lw=1.5,ls='--',alpha=0.7)
    ax.set_xticks(range(5)); ax.set_xticklabels(YEARS,fontsize=9)
    ax.set_xlabel('Drought Event Year',fontsize=9); ax.set_ylabel(label,fontsize=9)
    ax.set_title(f'({["a","b","c"][ci]}) {label}',fontsize=11,fontweight='bold',pad=4)
    ax.spines[['top','right']].set_visible(False)
leg=[plt.Rectangle((0,0),1,1,color=c,alpha=0.75,label=yr) for c,yr in zip(COLORS,YEARS)]
fig.legend(handles=leg,loc='lower center',ncol=5,fontsize=9,bbox_to_anchor=(0.5,0.01),framealpha=0.9)
fig.suptitle('Fig. 5b. Sentinel-1 SAR Backscatter Metrics per Drought Event',fontsize=10.5,fontweight='bold',y=0.98)
save_fig(fig, os.path.join(OUT_FIGS,'FIG05'), 'FIG05B_S1_Metrics')


# ══════════════════════════════════════════════════════════════════════════════
# FIG06 – Spatial Drought Maps (one per event)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── FIG06 Spatial Maps ──")
ref={d:{} for d in DISTS}
for d in DISTS:
    sub=df[df['district']==d]
    for col in ['precip_gs','sm_surface','ndvi','spi_min']:
        ref[d][col]=sub[col].mean()

for yr in YEARS:
    rows=[]
    for d in DISTS:
        sub=df[(df['year']==yr)&(df['district']==d)]
        if len(sub)==0: continue
        r=sub.iloc[0]
        rows.append({'DISTRICT':d.upper(),
                     'precip_anom':r['precip_gs']-ref[d]['precip_gs'],
                     'sm_anom':r['sm_surface']-ref[d]['sm_surface'],
                     'ndvi_anom':r['ndvi']-ref[d]['ndvi'],
                     'spi_min':r['spi_min'],
                     'yield_anom':r['yield_anomaly_pct']})
    gdf_yr=gdf_study.merge(pd.DataFrame(rows),on='DISTRICT')
    fig,axes=plt.subplots(1,5,figsize=(22,5.5))
    fig.patch.set_facecolor('white'); fig.subplots_adjust(wspace=0.28,left=0.03,right=0.99,top=0.85,bottom=0.09)
    for ci,(var,title,cmap,vmin,vmax) in enumerate([
        ('precip_anom','(a) Precip. Anomaly\n(mm day⁻¹)','RdBu',-6,6),
        ('sm_anom','(b) SM Anomaly\n(m³ m⁻³)','RdBu',-0.10,0.10),
        ('ndvi_anom','(c) NDVI Anomaly\n(–)','RdYlGn',-0.015,0.015),
        ('spi_min','(d) SPI Minimum\n(–)','RdBu',-1.2,0.2),
        ('yield_anom','(e) Yield Anomaly\n(%)','RdYlGn',-15,15)]):
        ax=axes[ci]
        gdf_yr.plot(ax=ax,column=var,cmap=cmap,vmin=vmin,vmax=vmax,edgecolor='#1a1a1a',linewidth=0.9)
        for _,row in gdf_yr.iterrows():
            cx,cy=row.geometry.centroid.x,row.geometry.centroid.y
            ax.text(cx,cy,f'{row[var]:.2f}',ha='center',va='center',fontsize=7.5,fontweight='bold',
                    bbox=dict(facecolor='white',alpha=0.75,edgecolor='none',pad=1.5))
        ax.set_title(title,fontsize=9,fontweight='bold',pad=4)
        ax.set_xlabel('Lon (°E)',fontsize=7.5); ax.set_ylabel('Lat (°N)' if ci==0 else '',fontsize=7.5)
        ax.tick_params(labelsize=7)
        div=make_axes_locatable(ax); cax=div.append_axes('right',size='6%',pad=0.05)
        sm_cb=plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(vmin,vmax)); sm_cb.set_array([])
        cb=plt.colorbar(sm_cb,cax=cax); cb.ax.tick_params(labelsize=6.5)
    fig.suptitle(f'Fig. 6. Spatial Drought-Stress Indicators — {yr} Event\nAnomalies vs 5-event mean | Province 1, Eastern Nepal',fontsize=11,fontweight='bold',y=0.97)
    save_fig(fig, os.path.join(OUT_FIGS,'FIG06'), f'FIG06_Spatial_{yr}')


# ══════════════════════════════════════════════════════════════════════════════
# FIG07 – Yield Analysis
# ══════════════════════════════════════════════════════════════════════════════
print("\n── FIG07 Yield Analysis ──")
fig,axes=plt.subplots(2,2,figsize=(13,11))
fig.patch.set_facecolor('white'); fig.subplots_adjust(hspace=0.40,wspace=0.35,left=0.09,right=0.97,top=0.92,bottom=0.08)

ax=axes[0,0]
for i,d in enumerate(DISTS):
    sub=df[df['district']==d].sort_values('year')
    ax.plot(sub['year'],sub['yield_t_ha'],'o-',color=COLORS[i%5],lw=1.8,ms=8,
            label=d.capitalize(),markeredgecolor='#1a1a1a',markeredgewidth=0.6)
ax.set_xlabel('Drought Event Year',fontsize=10); ax.set_ylabel('Maize Yield (t ha⁻¹)',fontsize=10)
ax.set_title('(a) District Maize Yield Across Drought Events\n(MoALD Nepal)',fontsize=10.5,fontweight='bold',pad=5)
ax.legend(fontsize=8.5,ncol=2,framealpha=0.9); ax.spines[['top','right']].set_visible(False); ax.set_ylim(1.5,3.5)

ax=axes[0,1]
for i,yr in enumerate(YEARS):
    sub=df[df['year']==yr]
    ax.scatter(sub['spi_min'],sub['yield_t_ha'],color=COLORS[i],s=90,label=yr,edgecolor='#1a1a1a',linewidth=0.7,zorder=5)
r,p=pearsonr(df['spi_min'],df['yield_t_ha'])
x_line=np.linspace(df['spi_min'].min(),df['spi_min'].max(),50)
ax.plot(x_line,LinearRegression().fit(df[['spi_min']],df['yield_t_ha']).predict(x_line.reshape(-1,1)),'k--',lw=1.5,alpha=0.7)
ax.set_xlabel('SPI Minimum (growing season)',fontsize=10); ax.set_ylabel('Maize Yield (t ha⁻¹)',fontsize=10)
ax.set_title(f'(b) Yield vs SPI Minimum | r={r:.3f}, p={p:.3f}',fontsize=10.5,fontweight='bold',pad=5)
ax.legend(fontsize=8.5,framealpha=0.9); ax.spines[['top','right']].set_visible(False)

ax=axes[1,0]
df2=df.copy()
for d in DISTS:
    mv=df2.loc[df2['district']==d,'temp_gs'].mean()
    df2.loc[df2['district']==d,'temp_anom']=df2.loc[df2['district']==d,'temp_gs']-mv
for i,yr in enumerate(YEARS):
    sub=df2[df2['year']==yr]
    ax.scatter(sub['temp_anom'],sub['yield_anomaly_pct'],color=COLORS[i],s=90,label=yr,edgecolor='#1a1a1a',linewidth=0.7,zorder=5)
r2b,_=pearsonr(df2['temp_anom'],df2['yield_anomaly_pct'])
ax.axhline(0,color='gray',lw=0.8,ls=':'); ax.axvline(0,color='gray',lw=0.8,ls=':')
ax.set_xlabel('Temperature Anomaly (°C)',fontsize=10); ax.set_ylabel('Yield Anomaly (%)',fontsize=10)
ax.set_title(f'(c) Yield Anomaly vs Temperature Anomaly | r={r2b:.3f}',fontsize=10.5,fontweight='bold',pad=5)
ax.legend(fontsize=8.5,framealpha=0.9); ax.spines[['top','right']].set_visible(False)

ax=axes[1,1]
hm=np.zeros((len(DISTS),len(YEARS)))
for i,d in enumerate(DISTS):
    for j,yr in enumerate(YEARS):
        v=df.loc[(df['district']==d)&(df['year']==yr),'yield_anomaly_pct']
        hm[i,j]=v.iloc[0] if len(v)>0 else 0.0
im=ax.imshow(hm,cmap='RdYlGn',vmin=-15,vmax=15,aspect='auto')
ax.set_xticks(range(5)); ax.set_xticklabels(YEARS,fontsize=9)
ax.set_yticks(range(6)); ax.set_yticklabels([d.capitalize() for d in DISTS],fontsize=9)
for i in range(6):
    for j in range(5):
        ax.text(j,i,f'{hm[i,j]:.1f}%',ha='center',va='center',fontsize=8.5,fontweight='bold',
                color='white' if abs(hm[i,j])>9 else '#1a1a1a')
ax.set_title('(d) Yield Anomaly Heatmap (% vs district mean)',fontsize=10.5,fontweight='bold',pad=5)
ax.set_xlabel('Drought Event Year',fontsize=10); ax.set_ylabel('District',fontsize=10)
div=make_axes_locatable(ax); cax=div.append_axes('right',size='4%',pad=0.06)
cb=plt.colorbar(im,cax=cax); cb.set_label('Yield anomaly (%)',fontsize=8.5); cb.ax.tick_params(labelsize=8)
fig.suptitle('Fig. 7. Maize Yield Analysis | Source: MoALD Nepal (2015–2024)',fontsize=11.5,fontweight='bold',y=0.98)
save_fig(fig, os.path.join(OUT_FIGS,'FIG07'), 'FIG07_Yield_Analysis')


# ══════════════════════════════════════════════════════════════════════════════
# FIG08 – Ablation Study
# ══════════════════════════════════════════════════════════════════════════════
print("\n── FIG08 Ablation ──")
fig,axes=plt.subplots(1,3,figsize=(15,6.5))
fig.patch.set_facecolor('white'); fig.subplots_adjust(wspace=0.35,left=0.07,right=0.97,top=0.88,bottom=0.22)
groups=df_abl['group'].tolist(); x=np.arange(len(groups))
bar_colors=['#d9d9d9']*4+['#74add1']*4+['#2166ac']
bar_colors=bar_colors[:len(groups)]
for ci,(metric,ylabel) in enumerate([('R2_LOEO','R² (LOEO-CV)'),('RMSE_LOEO','RMSE (t ha⁻¹)'),('MAE_LOEO','MAE (t ha⁻¹)')]):
    ax=axes[ci]; vals=df_abl[metric].tolist()
    bars=ax.bar(x,vals,color=bar_colors,edgecolor='#333',lw=0.7,alpha=0.88)
    best=int(np.argmax(vals)) if metric=='R2_LOEO' else int(np.argmin(vals))
    bars[best].set_edgecolor('#e6821e'); bars[best].set_linewidth(2.8)
    for bar,v in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2,v+0.002,f'{v:.3f}',ha='center',va='bottom',fontsize=7.5,rotation=40)
    ax.set_xticks(x); ax.set_xticklabels(groups,rotation=45,ha='right',fontsize=8.5)
    ax.set_ylabel(ylabel,fontsize=10); ax.set_title(f'({["a","b","c"][ci]}) {metric.replace("_LOEO","")}',fontsize=11,fontweight='bold',pad=5)
    ax.spines[['top','right']].set_visible(False)
    if metric=='R2_LOEO': ax.axhline(0,color='gray',lw=0.8,ls=':')
leg_h=[mpatches.Patch(color='#d9d9d9',label='Single modality'),
       mpatches.Patch(color='#74add1',label='Two modalities'),
       mpatches.Patch(color='#2166ac',label='Full multimodal')]
fig.legend(handles=leg_h,loc='lower center',ncol=3,fontsize=9.5,bbox_to_anchor=(0.5,0.005),framealpha=0.9)
fig.suptitle('Fig. 8. Multimodal Ablation Study — Yield Prediction (LOEO-CV)\nRandom Forest (n_trees=150) | n=30 observations',fontsize=11,fontweight='bold',y=0.98)
save_fig(fig, os.path.join(OUT_FIGS,'FIG08'), 'FIG08_Ablation_Study')


# ══════════════════════════════════════════════════════════════════════════════
# FIG09 – Early-Warning Lead-Time
# ══════════════════════════════════════════════════════════════════════════════
print("\n── FIG09 Early Warning ──")
lw=df_lt['lead_weeks'].tolist(); lw_lbl=[f'{w}w' for w in lw]
fig,axes=plt.subplots(1,3,figsize=(14,5.5))
fig.patch.set_facecolor('white'); fig.subplots_adjust(wspace=0.35,left=0.08,right=0.97,top=0.86,bottom=0.14)
ax=axes[0]
ax.plot(lw,df_lt['R2_LOEO'],'o-',color='#2166ac',lw=2.2,ms=10,markeredgecolor='#1a1a1a',markeredgewidth=0.8)
ax.fill_between(lw,[min(0,v) for v in df_lt['R2_LOEO']],df_lt['R2_LOEO'].tolist(),alpha=0.15,color='#2166ac')
ax.axhline(0,color='gray',lw=0.9,ls=':')
for x_,v in zip(lw,df_lt['R2_LOEO']): ax.text(x_,v+0.02,f'{v:.2f}',ha='center',fontsize=8.5,color='#2166ac',fontweight='bold')
ax.set_xlabel('Lead time (weeks before drought peak)',fontsize=10); ax.set_ylabel('R² (LOEO-CV)',fontsize=10)
ax.set_title('(a) Predictive Skill (R²) vs Lead Time',fontsize=11,fontweight='bold',pad=5)
ax.set_xticks(lw); ax.set_xticklabels(lw_lbl,fontsize=9); ax.spines[['top','right']].set_visible(False)

ax=axes[1]
ax.plot(lw,df_lt['RMSE_LOEO'],'s-',color='#d6604d',lw=2.2,ms=10,markeredgecolor='#1a1a1a',markeredgewidth=0.8,label='RMSE')
ax.plot(lw,df_lt['MAE_LOEO'],'^-',color='#1a9641',lw=2.2,ms=10,markeredgecolor='#1a1a1a',markeredgewidth=0.8,label='MAE')
ax.set_xlabel('Lead time (weeks before drought peak)',fontsize=10); ax.set_ylabel('Error (%)',fontsize=10)
ax.set_title('(b) RMSE and MAE vs Lead Time',fontsize=11,fontweight='bold',pad=5)
ax.set_xticks(lw); ax.set_xticklabels(lw_lbl,fontsize=9); ax.legend(fontsize=9,framealpha=0.9); ax.spines[['top','right']].set_visible(False)

ax=axes[2]
r2_arr=df_lt['R2_LOEO'].values; denom=r2_arr[0] if r2_arr[0]!=0 else 1e-6
ss=[1.0]+[r2_arr[i]/denom for i in range(1,len(r2_arr))]
bar_c=['#2166ac' if s>=0.8 else '#74add1' if s>=0.5 else '#d9d9d9' for s in ss]
ax.bar(range(len(lw)),ss,color=bar_c,edgecolor='#1a1a1a',lw=0.7,alpha=0.85)
ax.axhline(0.8,color='#2166ac',lw=1.3,ls='--',label='80% retention')
ax.axhline(0.5,color='#74add1',lw=1.3,ls='--',label='50% retention')
ax.set_xticks(range(len(lw))); ax.set_xticklabels(lw_lbl,fontsize=9)
ax.set_xlabel('Lead time',fontsize=10); ax.set_ylabel('Relative R² skill score',fontsize=10)
ax.set_title('(c) Skill Score vs 1-Week Lead',fontsize=11,fontweight='bold',pad=5)
ax.legend(fontsize=8.5,framealpha=0.9); ax.spines[['top','right']].set_visible(False)
fig.suptitle('Fig. 9. Early-Warning Performance: Yield-Anomaly Prediction at Multiple Lead Times\nLOEO-CV | n=30 | Features: precip + temp at lead time + soil + phenology',fontsize=11,fontweight='bold',y=0.98)
save_fig(fig, os.path.join(OUT_FIGS,'FIG09'), 'FIG09_Early_Warning')


# ══════════════════════════════════════════════════════════════════════════════
# FIG10 – Feature Importance & Correlation
# ══════════════════════════════════════════════════════════════════════════════
print("\n── FIG10 Feature Importance ──")
MODALITY_COLOR={
    'ndvi':'#1a9641','ndvi_std':'#1a9641','evi':'#1a9641','ndwi':'#1a9641','savi':'#1a9641',
    'VV_dB':'#7b2d8b','VH_dB':'#7b2d8b','VV_VH':'#7b2d8b',
    'precip_gs':'#2166ac','precip_std':'#2166ac','temp_gs':'#2166ac','evap_gs':'#2166ac',
    'spi_mean':'#2166ac','spi_min':'#2166ac','spi_drought_days':'#2166ac',
    'sm_surface':'#74add1','sm_rootzone':'#74add1',
    'clay':'#d6604d','soc':'#d6604d','cec':'#d6604d','sand':'#d6604d','phh2o':'#d6604d',
    'bdod':'#d6604d','nitrogen':'#d6604d',
    'sos_doy':'#e6821e','pos_doy':'#e6821e','eos_doy':'#e6821e','ndvi_max_phen':'#e6821e',
}
df_imp_s=df_imp.sort_values('rf_importance',ascending=True)
n=len(df_imp_s)
fig,axes=plt.subplots(1,2,figsize=(14,max(6,n*0.33)))
fig.patch.set_facecolor('white'); fig.subplots_adjust(wspace=0.45,left=0.07,right=0.97,top=0.90,bottom=0.08)
ax=axes[0]
bar_c=[MODALITY_COLOR.get(f,'#aaaaaa') for f in df_imp_s['feature']]
ax.barh(df_imp_s['feature'],df_imp_s['rf_importance'],color=bar_c,edgecolor='#333',lw=0.6,height=0.72)
ax.set_xlabel('RF Feature Importance (mean decrease impurity)',fontsize=9.5)
ax.set_title('(a) Random Forest Feature Importance\n(Target: yield anomaly % | n_trees=300)',fontsize=10,fontweight='bold',pad=5)
for _,row in df_imp_s.iterrows():
    ax.text(row['rf_importance']+0.002,df_imp_s.index.get_loc(_),f'{row["rf_importance"]:.3f}',va='center',fontsize=7)
ax.spines[['top','right']].set_visible(False)
leg_h=[mpatches.Patch(color='#1a9641',label='Optical (S2)'),mpatches.Patch(color='#7b2d8b',label='SAR (S1)'),
       mpatches.Patch(color='#2166ac',label='Climate (ERA5)'),mpatches.Patch(color='#74add1',label='SMAP'),
       mpatches.Patch(color='#d6604d',label='Soil'),mpatches.Patch(color='#e6821e',label='Phenology')]
ax.legend(handles=leg_h,fontsize=8,loc='lower right',framealpha=0.9)

ax=axes[1]; sorted_c=df_imp.sort_values('pearson_r',key=abs,ascending=True)
c_colors=['#d6604d' if r<0 else '#2166ac' for r in sorted_c['pearson_r']]
ax.barh(sorted_c['feature'],sorted_c['pearson_r'],color=c_colors,edgecolor='#333',lw=0.6,height=0.72,label='Pearson r',alpha=0.80)
ax.scatter(sorted_c['spearman_r'],range(len(sorted_c)),color='#e6821e',s=35,zorder=5,marker='D',label='Spearman ρ')
ax.axvline(0,color='#333',lw=0.9)
ax.axvline(0.3,color='#aaa',lw=0.8,ls='--',alpha=0.7); ax.axvline(-0.3,color='#aaa',lw=0.8,ls='--',alpha=0.7)
ax.set_xlabel('Correlation with yield anomaly (%)',fontsize=9.5)
ax.set_title('(b) Pearson r and Spearman ρ\n(Blue=positive | Red=negative)',fontsize=10,fontweight='bold',pad=5)
ax.legend(fontsize=8.5,framealpha=0.9,loc='lower right'); ax.spines[['top','right']].set_visible(False)
fig.suptitle('Fig. 10. Feature Importance and Correlation Analysis\nn=30 district-event observations',fontsize=11,fontweight='bold',y=0.98)
save_fig(fig, os.path.join(OUT_FIGS,'FIG10'), 'FIG10_Feature_Importance')


# ══════════════════════════════════════════════════════════════════════════════
# FIG11 – Cross-Event Generalisation
# ══════════════════════════════════════════════════════════════════════════════
print("\n── FIG11 Cross-Event ──")
fig,axes=plt.subplots(1,2,figsize=(13,6.5))
fig.patch.set_facecolor('white'); fig.subplots_adjust(wspace=0.36,left=0.09,right=0.97,top=0.87,bottom=0.13)

event_r2,event_rmse,event_mae=[],[],[]
for yr in YEARS:
    sub=df_loeo[df_loeo['held_out_year'].astype(str)==yr]
    event_r2.append(r2_score(sub['y_true'],sub['y_pred']))
    event_rmse.append(np.sqrt(mean_squared_error(sub['y_true'],sub['y_pred'])))
    event_mae.append(mean_absolute_error(sub['y_true'],sub['y_pred']))

ax=axes[0]; x=np.arange(5); w=0.25
ax.bar(x-w,event_r2,  w,label='R²',  color='#2166ac',edgecolor='#1a1a1a',lw=0.7,alpha=0.88)
ax.bar(x,  event_rmse,w,label='RMSE',color='#d6604d',edgecolor='#1a1a1a',lw=0.7,alpha=0.88)
ax.bar(x+w,event_mae, w,label='MAE', color='#1a9641',edgecolor='#1a1a1a',lw=0.7,alpha=0.88)
ax.set_xticks(x); ax.set_xticklabels(YEARS,fontsize=10)
ax.set_xlabel('Held-Out Event Year',fontsize=10); ax.set_ylabel('Metric',fontsize=10)
ax.set_title('(a) LOEO Metrics per Held-Out Event\n(Train=4 events, Test=1 event)',fontsize=10.5,fontweight='bold',pad=5)
ax.legend(fontsize=9,framealpha=0.9); ax.axhline(0,color='gray',lw=0.8,ls=':'); ax.spines[['top','right']].set_visible(False)
for i,v in enumerate(event_r2):
    ax.text(i-w,v+(0.02 if v>=0 else -0.05),f'{v:.2f}',ha='center',fontsize=7.5,color='#2166ac',fontweight='bold')

ax=axes[1]
at=df_loeo['y_true'].values; ap=df_loeo['y_pred'].values
for i,yr in enumerate(YEARS):
    sub=df_loeo[df_loeo['held_out_year'].astype(str)==yr]
    ax.scatter(sub['y_true'],sub['y_pred'],color=COLORS[i],s=85,label=yr,edgecolor='#1a1a1a',linewidth=0.7,alpha=0.88,zorder=5)
lo=min(at.min(),ap.min())-1.5; hi=max(at.max(),ap.max())+1.5
ax.plot([lo,hi],[lo,hi],'k--',lw=1.5,alpha=0.7,label='1:1 line',zorder=1)
ax.plot(np.linspace(lo,hi,100),LinearRegression().fit(at.reshape(-1,1),ap).predict(np.linspace(lo,hi,100).reshape(-1,1)),'k-',lw=1.3,alpha=0.45,label='Regression')
r2_pool=r2_score(at,ap); rmse_pool=np.sqrt(mean_squared_error(at,ap)); mae_pool=mean_absolute_error(at,ap)
ax.set_xlim(lo,hi); ax.set_ylim(lo,hi); ax.set_aspect('equal')
ax.set_xlabel('Observed Yield Anomaly (%)',fontsize=10); ax.set_ylabel('Predicted Yield Anomaly (%)',fontsize=10)
ax.set_title(f'(b) Observed vs Predicted — All 5 Held-Out Events\nPooled R²={r2_pool:.3f} | RMSE={rmse_pool:.3f} % | MAE={mae_pool:.3f} %',fontsize=10.5,fontweight='bold',pad=5)
ax.legend(fontsize=8.5,ncol=2,framealpha=0.9); ax.spines[['top','right']].set_visible(False)
ax.text(0.97,0.05,f'R²={r2_pool:.3f}\nRMSE={rmse_pool:.3f}%\nMAE={mae_pool:.3f}%\nn={len(at)}',
        transform=ax.transAxes,ha='right',va='bottom',fontsize=9,
        bbox=dict(facecolor='#f0f4fa',edgecolor='#aaaacc',pad=4,alpha=0.95))
fig.suptitle('Fig. 11. Cross-Event Generalisation: Leave-One-Event-Out Evaluation\nRandom Forest | Full multimodal features | Target: yield anomaly (%)',fontsize=11,fontweight='bold',y=0.98)
save_fig(fig, os.path.join(OUT_FIGS,'FIG11'), 'FIG11_Cross_Event_Generalisation')

print("\n\n✅  02_generate_all_plots.py  COMPLETE")
print(f"All figures saved to: {OUT_FIGS}")
