from __future__ import annotations

import argparse, json, math, time
from pathlib import Path
from urllib.parse import urlencode
import numpy as np, pandas as pd, requests, xarray as xr
from research.topic_lock.gfc17_seed_null import HORIZON, exact_cut_indices, model_factory, safe_feature_columns, _load_fev_long, _supervised

BASE='https://tds.gdex.ucar.edu/thredds/ncss/grid/files/g/d084001'; SOURCE_TZ='America/New_York'
BBOX=dict(north=45.6,south=40.5,west=-74.5,east=-68.5)
STATIONS={'BOS':(42.3606,-71.0098),'BDR':(41.1635,-73.1262),'BTV':(44.4683,-73.1499),'CON':(43.2049,-71.5026),'PWM':(43.6423,-70.3045),'PVD':(41.7223,-71.4325),'BDL':(41.9389,-72.6832),'ORH':(42.2706,-71.8731)}
MAPPING_VARIANTS={'historical_weighted':{'ME':{'PWM':1.0},'NH':{'CON':1.0},'VT':{'BTV':1.0},'CT':{'BDL':.13,'BDR':.87},'RI':{'PVD':1.0},'SEMASS':{'PVD':1.0},'WCMASS':{'BDL':.5,'ORH':.5},'NEMASSBOST':{'BOS':1.0}},'single_station':{'ME':{'PWM':1.0},'NH':{'CON':1.0},'VT':{'BTV':1.0},'CT':{'BDL':1.0},'RI':{'PVD':1.0},'SEMASS':{'PVD':1.0},'WCMASS':{'ORH':1.0},'NEMASSBOST':{'BOS':1.0}}}

def normalize_zone(x):
 s=str(x).upper().replace('-','').replace('_','').replace(' ','')
 for key,z in [('NEMASSBOST','NEMASSBOST'),('NEMASS','NEMASSBOST'),('NEMA','NEMASSBOST'),('SEMASS','SEMASS'),('SEMA','SEMASS'),('WCMASS','WCMASS'),('WCMA','WCMASS'),('ME','ME'),('NH','NH'),('VT','VT'),('CT','CT'),('RI','RI')]:
  if key in s:return z
 raise ValueError(x)
def local_horizon_to_utc(forecast_start):
 local=pd.date_range(pd.Timestamp(forecast_start),periods=HORIZON,freq='h')
 try: aware=local.tz_localize(SOURCE_TZ,ambiguous='infer',nonexistent='shift_forward')
 except ValueError: aware=local.tz_localize(SOURCE_TZ,ambiguous=False,nonexistent='shift_forward')
 return aware.tz_convert('UTC').tz_localize(None)
def cycle_and_leads(forecast_start):
 t=local_horizon_to_utc(forecast_start); cycle=pd.Timestamp(forecast_start).normalize()-pd.Timedelta(hours=6)
 lo=int(math.floor(float((t[0]-cycle)/pd.Timedelta(hours=1))/3)*3); hi=int(math.ceil(float((t[-1]-cycle)/pd.Timedelta(hours=1))/3)*3)
 return cycle,list(range(lo,hi+1,3))
def build_ncss_url(cycle,lead_h):
 cycle=pd.Timestamp(cycle).tz_localize(None); d=cycle.strftime('%Y%m%d'); init=cycle.strftime('%Y%m%d%H'); fn=f'gfs.0p25.{init}.f{int(lead_h):03d}.grib2'
 return f'{BASE}/{cycle:%Y}/{d}/{fn}?'+urlencode({'var':'Temperature_height_above_ground','vertCoord':2,**BBOX,'horizStride':1,'accept':'netcdf'})
def _temperature_var(ds):
 if 'Temperature_height_above_ground' in ds.data_vars:return 'Temperature_height_above_ground'
 cand=[v for v in ds.data_vars if 'Temperature' in v and 'height_above_ground' in v]
 if not cand: raise RuntimeError(f'missing temp {list(ds.data_vars)}')
 return cand[0]
def _to_fahrenheit(values,units):
 u=(units or '').lower().strip(); x=np.asarray(values,dtype=float)
 if u=='k' or 'kelvin' in u or (not u and np.nanmedian(x)>150):return (x-273.15)*9/5+32
 if u=='c' or 'celsius' in u or 'degc' in u:return x*9/5+32
 if u=='f' or 'fahrenheit' in u:return x
 if np.nanmedian(x)>150:return (x-273.15)*9/5+32
 raise RuntimeError(units)
def _request_with_retries(session,url,attempts=8,timeout=90,sleep_fn=time.sleep):
 for attempt in range(attempts):
  try:
   r=session.get(url,timeout=timeout)
   if 400<=r.status_code<500: raise RuntimeError(f'NCSS {r.status_code}: {r.text[:500]} URL={url}')
   r.raise_for_status(); return r
  except RuntimeError: raise
  except (requests.exceptions.ReadTimeout,requests.exceptions.ConnectionError):
   if attempt==attempts-1: raise
   sleep_fn(min(30,2**attempt))
def fetch_grid(cycle,lead_h,session,cache_dir):
 cache_dir.mkdir(parents=True,exist_ok=True); p=cache_dir/f'{cycle:%Y%m%d%H}_f{lead_h:03d}.nc'
 if not p.exists() or p.stat().st_size<100:
  r=_request_with_retries(session,build_ncss_url(cycle,lead_h));
  if len(r.content)<100: raise RuntimeError('tiny response')
  p.write_bytes(r.content)
 return xr.open_dataset(p,engine='netcdf4')
def station_values(ds):
 da=ds[_temperature_var(ds)].squeeze(drop=True); lat_name=next(n for n in da.coords if 'lat' in n.lower()); lon_name=next(n for n in da.coords if 'lon' in n.lower()); lons=np.asarray(da[lon_name].values,dtype=float); use360=np.nanmin(lons)>=0; units=str(da.attrs.get('units','')); out={}
 for name,(lat,lon) in STATIONS.items(): out[name]=float(_to_fahrenheit(np.asarray(da.sel({lat_name:lat,lon_name:(lon%360 if use360 else lon)},method='nearest').values),units).reshape(-1)[0])
 return out
def acquire_window_weather(forecast_start,cache_dir):
 cycle,leads=cycle_and_leads(forecast_start); session=requests.Session(); rows=[]
 for lead in leads:
  ds=fetch_grid(cycle,lead,session,cache_dir); vals=station_values(ds); ds.close(); rows.append({'timestamp_utc':cycle+pd.Timedelta(hours=lead),**vals}); print('GFS',cycle,'lead',lead,'ok',flush=True)
 s=pd.DataFrame(rows).set_index('timestamp_utc').sort_index(); target=local_horizon_to_utc(forecast_start); s=s.reindex(s.index.union(target)).interpolate(method='time').reindex(target); out=s.reset_index(); out.insert(0,'timestamp_local',pd.date_range(pd.Timestamp(forecast_start),periods=HORIZON,freq='h')); return out
def zone_weather(station_df,variant):
 out=pd.DataFrame({'timestamp_local':pd.to_datetime(station_df['timestamp_local'])})
 for zone,weights in MAPPING_VARIANTS[variant].items():
  z=np.zeros(len(station_df));
  for station,w in weights.items(): z+=float(w)*pd.to_numeric(station_df[station]).to_numpy(float)
  out[zone]=z
 return out
def evaluate_windows(windows,outdir,seed=20260829):
 df=_load_fev_long(); rows=[]; cache_dir=outdir/'gfs_cache'; first=next(iter(df.groupby('id',sort=True)))[1].sort_values('timestamp').reset_index(drop=True); cuts0=exact_cut_indices(len(first)); station_by_window={}
 for w in windows: station_by_window[w]=acquire_window_weather(pd.Timestamp(first.iloc[cuts0[w-1]]['timestamp']),cache_dir)
 for sid,g in df.groupby('id',sort=True):
  zone=normalize_zone(str(sid)); g=g.sort_values('timestamp').reset_index(drop=True); cuts=exact_cut_indices(len(g)); sup=_supervised(g); cols=safe_feature_columns(True)
  for w in windows:
   cut=cuts[w-1]; train=sup.iloc[:cut]; good=train[cols+['target']].dropna().index; Xtr=train.loc[good,cols]; ytr=train.loc[good,'target'].astype(float); oracle=sup.iloc[cut:cut+HORIZON].copy(); truth=oracle['target'].astype(float).to_numpy(); station=station_by_window[w]; ops={}
   for variant in MAPPING_VARIANTS:
    zw=zone_weather(station,variant)[zone].to_numpy(float); test=oracle.copy(); test.loc[:,'weather']=zw; test.loc[:,'weather_sq']=zw**2; test.loc[:,'weather_hour']=zw*test['hour_sin'].to_numpy(float); ops[variant]=test[cols]
   for model_name in ('HGB','RF','ET'):
    model=model_factory(model_name,seed); model.fit(Xtr,ytr); oracle_mae=float(np.mean(np.abs(truth-model.predict(oracle[cols]))))
    for variant in MAPPING_VARIANTS:
     rows.append({'zone':zone,'window':w,'mapping':variant,'condition':'ORACLE_WEATHER','model':model_name,'mae':oracle_mae}); rows.append({'zone':zone,'window':w,'mapping':variant,'condition':'OPERATIONAL_GFS','model':model_name,'mae':float(np.mean(np.abs(truth-model.predict(ops[variant]))))})
   print('DONE',zone,w,flush=True)
 result=pd.DataFrame(rows); result.to_csv(outdir/'operational_mae_by_window.csv',index=False); return result
def summarize(result,outdir):
 p=result.pivot_table(index=['zone','window','mapping','model'],columns='condition',values='mae').reset_index(); p['D_operational_minus_oracle']=p['OPERATIONAL_GFS']-p['ORACLE_WEATHER']; p.to_csv(outdir/'operational_contrasts.csv',index=False)
 rank_rows=[]
 for (mapping,zone,window),q in result.groupby(['mapping','zone','window']):
  m=q.pivot_table(index='condition',columns='model',values='mae'); mods=['HGB','RF','ET']; ro=tuple(sorted(mods,key=lambda x:(m.loc['ORACLE_WEATHER',x],x))); oo=tuple(sorted(mods,key=lambda x:(m.loc['OPERATIONAL_GFS',x],x))); vals=sorted(float(m.loc['ORACLE_WEATHER',x]) for x in mods); rank_rows.append({'mapping':mapping,'zone':zone,'window':window,'rank_changed':ro!=oo,'winner_changed':ro[0]!=oo[0],'oracle_margin':vals[1]-vals[0]})
 pd.DataFrame(rank_rows).to_csv(outdir/'operational_ranking.csv',index=False)
def parse_windows(s):
 out=[]
 for part in s.split(','):
  if '-' in part:
   a,b=map(int,part.split('-')); out.extend(range(a,b+1))
  else: out.append(int(part))
 return out
def main():
 p=argparse.ArgumentParser(); p.add_argument('--windows',required=True); p.add_argument('--outdir',required=True); p.add_argument('--seed',type=int,default=20260829); a=p.parse_args(); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True); summarize(evaluate_windows(parse_windows(a.windows),out,a.seed),out)
if __name__=='__main__': main()
