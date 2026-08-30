import glob, json, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')
rng=np.random.default_rng(20260830)

files=glob.glob('dominick_raw/**/*.dta',recursive=True)+glob.glob('*.dta')
if not files: raise RuntimeError('No .dta found after unzip')
fn=max(files,key=lambda p: Path(p).stat().st_size)
print('DTA',fn,Path(fn).stat().st_size)
df=pd.read_stata(fn,convert_categoricals=False)
df.columns=[str(c).lower() for c in df.columns]
print('RAW',df.shape,list(df.columns))
if 'store' not in df or 'date' not in df: raise RuntimeError('store/date missing')
df['store']=pd.to_numeric(df['store'],errors='coerce').replace({69:137})
s=df['date'].astype(str).str.replace(r'\.0$','',regex=True).str.zfill(6)
df['date2']=pd.to_datetime('19'+s,format='%Y%m%d',errors='coerce')
df=df.dropna(subset=['store','date2']).copy(); df['store']=df['store'].astype(int)
known=['grocery','gm','dairy','frozen','meat','fish','produce','saladbar','floral','deli','cheese','bakery','pharmacy','jewelry','cosmetics','haba','camera','photofin','video','beer','wine','spirits']
sales=[c for c in known if c in df.columns]
if len(sales)<12: raise RuntimeError(f'Too few sales cols: {sales}')
for c in sales:
    df[c]=pd.to_numeric(df[c],errors='coerce').mask(lambda x:x<0)
df['cust']=pd.to_numeric(df['custcoun'],errors='coerce').mask(lambda x:x<0) if 'custcoun' in df else np.nan
df['total_sales']=df[sales].sum(axis=1,min_count=1)
df['week']=df.date2.dt.to_period('W-SUN').dt.start_time
wk=df.groupby(['store','week'],as_index=False).agg(days=('date2','nunique'),total_sales=('total_sales','sum'),cust=('cust','sum'))
dep=df.groupby(['store','week'],as_index=False)[sales].sum(); wk=wk.merge(dep,on=['store','week'],how='left')
good=wk[(wk.days>=4)&(wk.total_sales>0)].copy(); gmin,gmax=good.week.min(),good.week.max()
print('GOOD',len(good),'stores',good.store.nunique(),'range',gmin,gmax)
endp=good.groupby('store').agg(first_week=('week','min'),last_week=('week','max'),n_good_weeks=('week','nunique'),mean_week_sales=('total_sales','mean'),mean_week_cust=('cust','mean')).reset_index()
meta=pd.read_csv('stores.csv',header=None,names=['store','city','price_tier','zone','zip','address'])
meta['store']=pd.to_numeric(meta.store,errors='coerce').astype('Int64'); meta['metadata_closed']=meta.address.astype(str).str.strip().str.lower().eq('closed')
endp=endp.merge(meta[['store','city','metadata_closed']],on='store',how='left'); endp['metadata_closed']=endp.metadata_closed.fillna(False)
endp['early_26w']=endp.last_week<=gmax-pd.Timedelta(weeks=26); endp['early_52w']=endp.last_week<=gmax-pd.Timedelta(weeks=52); endp['closure_consistent_26w']=endp.early_26w&endp.metadata_closed
endp.to_csv('dominick_store_endpoints.csv',index=False)
print('ENDPOINT_COUNTS',json.dumps({'early26':int(endp.early_26w.sum()),'early52':int(endp.early_52w.sum()),'metadata_closed':int(endp.metadata_closed.sum()),'early26_and_closed':int(endp.closure_consistent_26w.sum())}))
print(endp[endp.early_26w].sort_values('last_week').head(40).to_string(index=False))
last=endp.set_index('store').last_week.to_dict(); closed=endp.set_index('store').metadata_closed.to_dict()

def snap(cutoff,h=26,k=8,nperm=250):
    p0=cutoff-pd.Timedelta(weeks=26); r0=cutoff-pd.Timedelta(weeks=4); target=cutoff+pd.Timedelta(weeks=h)
    pre=good[(good.week>=p0)&(good.week<cutoff)]; recent=good[(good.week>=r0)&(good.week<cutoff)]
    a=pre.groupby('store').week.nunique(); b=recent.groupby('store').week.nunique(); risk=sorted(set(a[a>=18].index)&set(b[b>=2].index))
    surv=[s for s in risk if last.get(s,pd.Timestamp.min)>=target]; exits=[s for s in risk if last.get(s,pd.Timestamp.max)<target]
    if len(risk)<25 or len(surv)<20 or len(exits)<3: return None
    p=pre[pre.store.isin(risk)]
    base=p.groupby('store').agg(y=('total_sales','mean'),cust_mean=('cust','mean'),cust_sd=('cust','std'),sales_sd=('total_sales','std'))
    base['cust_cv']=base.cust_sd/base.cust_mean.replace(0,np.nan); base['sales_cv']=base.sales_sd/base.y.replace(0,np.nan)
    dm=p.groupby('store')[sales].mean(); sh=dm.div(dm.sum(axis=1).replace(0,np.nan),axis=0).add_prefix('share_')
    feat=base[['cust_mean','cust_cv','sales_cv']].join(sh).replace([np.inf,-np.inf],np.nan); feat=feat.fillna(feat.median()).fillna(0).loc[risk]
    X=pd.DataFrame(StandardScaler().fit_transform(feat),index=feat.index); y=base.loc[risk,'y'].to_dict()
    def bench(cands,focals):
        out={}; cands=list(cands)
        for s in focals:
            ch=[j for j in cands if j!=s]; kk=min(k,len(ch))
            if kk<3: out[s]=np.nan; continue
            d=np.sqrt(((X.loc[ch].values-X.loc[s].values)**2).sum(axis=1)); peers=[ch[z] for z in np.argsort(d)[:kk]]; out[s]=float(np.median([y[j] for j in peers]))
        return out
    br=bench(risk,surv); bs=bench(surv,surv); rows=[]
    for s in surv:
        yi,a1,a2=y.get(s),br.get(s),bs.get(s)
        if all(np.isfinite(v) and v>0 for v in [yi,a1,a2]): rows.append((s,yi,a1,a2,math.log(yi/a1),math.log(yi/a2)))
    rr=pd.DataFrame(rows,columns=['store','y','b_risk','b_surv','g_risk','g_surv'])
    if len(rr)<15:return None
    rr['shift']=np.log(rr.b_surv/rr.b_risk); rr['lr']=rr.g_risk<math.log(.9); rr['ls']=rr.g_surv<math.log(.9)
    q=max(1,int(np.ceil(.2*len(rr)))); A=set(rr.nsmallest(q,'g_risk').store); B=set(rr.nsmallest(q,'g_surv').store)
    act={'med_signed_shift':float(rr['shift'].median()),'med_abs_shift':float(rr['shift'].abs().median()),'p90_abs_shift':float(rr['shift'].abs().quantile(.9)),'label_flip':float((rr.lr!=rr.ls).mean()),'spearman':float(spearmanr(rr.g_risk,rr.g_surv).statistic),'bottom20_jaccard':float(len(A&B)/len(A|B))}
    null=[]
    for _ in range(nperm):
        removed=set(rng.choice(risk,size=len(exits),replace=False).tolist()); bx=bench([s for s in risk if s not in removed],surv); vals=[]
        for _,row in rr.iterrows():
            v=bx.get(int(row.store),np.nan)
            if np.isfinite(v) and v>0: vals.append((int(row.store),math.log(v/row.b_risk),math.log(row.y/v)))
        z=pd.DataFrame(vals,columns=['store','shift','g']); z=rr[['store','g_risk']].merge(z,on='store')
        if len(z)>=10:null.append((float(z['shift'].abs().median()),float(((z.g_risk<math.log(.9))!=(z.g<math.log(.9))).mean()),float(spearmanr(z.g_risk,z.g).statistic)))
    nul=pd.DataFrame(null,columns=['shift','flip','rho'])
    out={'cutoff':str(cutoff.date()),'target':str(target.date()),'n_risk':len(risk),'n_survivors':len(surv),'n_exits':len(exits),'n_focal':len(rr),**act,'null_p_shift':float((nul['shift']>=act['med_abs_shift']).mean()),'null_p_flip':float((nul['flip']>=act['label_flip']).mean()),'null_p_rho':float((nul['rho']<=act['spearman']).mean()),'exit_closed_share':float(np.mean([closed.get(s,False) for s in exits]))}
    return out,rr

res=[]; det=[]
for c in pd.date_range(gmin+pd.Timedelta(weeks=52),gmax-pd.Timedelta(weeks=30),freq='13W'):
    z=snap(pd.Timestamp(c))
    if z:
        s,r=z;res.append(s);r['cutoff']=s['cutoff'];det.append(r)
R=pd.DataFrame(res);R.to_csv('dominick_survival_smoke_results.csv',index=False)
if det:pd.concat(det,ignore_index=True).to_csv('dominick_survival_smoke_detail.csv',index=False)
print('VALID_WINDOWS',len(R))
if len(R):
    print(R.to_string(index=False)); print('AGG',json.dumps({'median_exits':float(R.n_exits.median()),'median_abs_shift':float(R.med_abs_shift.median()),'median_flip':float(R.label_flip.median()),'median_spearman':float(R.spearman.median()),'median_jaccard':float(R.bottom20_jaccard.median()),'shift_p05_share':float((R.null_p_shift<=.05).mean()),'flip_p05_share':float((R.null_p_flip<=.05).mean())}))
else:print('DATA_GATE_FAIL')
Path('dominick_survival_smoke.md').write_text(f'# Dominick survivor benchmark smoke\n\nStores: {good.store.nunique()}\nRange: {gmin.date()} to {gmax.date()}\nEarly26: {int(endp.early_26w.sum())}\nEarly26+Closed: {int(endp.closure_consistent_26w.sum())}\nValid windows: {len(R)}\n')
