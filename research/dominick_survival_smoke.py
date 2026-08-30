import json, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadr
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
rng = np.random.default_rng(20260830)
STORE = 'store'

r = pyreadr.read_r('dominick.rda')
if not r:
    raise RuntimeError('RDA contained no objects')
key = list(r.keys())[0]
df = r[key].copy()
print('RDA_KEY', key, 'SHAPE', df.shape)
print('COLS', list(df.columns))
low = {c.lower(): c for c in df.columns}
store_col = low.get('store') or low.get('store_id')
date_col = low.get('date')
cust_col = low.get('custcoun') or low.get('customer_count')
if store_col is None or date_col is None:
    raise RuntimeError('store/date columns not found')
df[store_col] = pd.to_numeric(df[store_col], errors='coerce').astype('Int64')
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
df = df.dropna(subset=[store_col,date_col]).copy()
df[store_col] = df[store_col].astype(int)

numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
exclude = ('store','cust','count','coup','week','year','month','day')
sales_cols = [c for c in numeric if c != store_col and not any(t in c.lower() for t in exclude)]
if len(sales_cols) < 5:
    raise RuntimeError(f'Failed sales-column inference: {numeric}')
print('SALES_COLS', sales_cols)
df['total_sales'] = df[sales_cols].fillna(0).sum(axis=1)
df['cust'] = pd.to_numeric(df[cust_col], errors='coerce') if cust_col else np.nan
df['week'] = df[date_col].dt.to_period('W-SUN').dt.start_time

wk = df.groupby([store_col,'week'],as_index=False).agg(days=(date_col,'nunique'), total_sales=('total_sales','sum'), cust=('cust','sum'))
dept = df.groupby([store_col,'week'],as_index=False)[sales_cols].sum()
wk = wk.merge(dept,on=[store_col,'week'],how='left')
good = wk[wk.days >= 4].copy()
gmin,gmax = good.week.min(),good.week.max()
print('GLOBAL_WEEK_RANGE',gmin.date(),gmax.date(),'N_STORES',good[store_col].nunique())

endp = good.groupby(store_col).agg(first_week=('week','min'),last_week=('week','max'),n_good_weeks=('week','nunique'),mean_week_sales=('total_sales','mean'),mean_week_cust=('cust','mean')).reset_index().rename(columns={store_col:'store'})
stores = pd.read_csv('stores.csv',header=None,names=['store','city','price_tier','zone','zip','address'])
stores['store'] = pd.to_numeric(stores.store,errors='coerce').astype('Int64')
stores['metadata_closed'] = stores.address.astype(str).str.strip().str.lower().eq('closed')
endp = endp.merge(stores[['store','city','metadata_closed']],on='store',how='left')
endp['early_26w'] = endp.last_week <= gmax-pd.Timedelta(weeks=26)
endp['early_52w'] = endp.last_week <= gmax-pd.Timedelta(weeks=52)
endp['closure_consistent_26w'] = endp.early_26w & endp.metadata_closed.fillna(False)
endp.to_csv('dominick_store_endpoints.csv',index=False)
print('ENDPOINT_COUNTS',json.dumps({'early_26w':int(endp.early_26w.sum()),'early_52w':int(endp.early_52w.sum()),'metadata_closed':int(endp.metadata_closed.fillna(False).sum()),'closure_consistent_26w':int(endp.closure_consistent_26w.sum())}))
print(endp[endp.early_26w].sort_values('last_week').head(30).to_string(index=False))
last_map = endp.set_index('store').last_week.to_dict()
closed_map = endp.set_index('store').metadata_closed.fillna(False).to_dict()

def make_snapshot(cutoff,h=26,k=8,nperm=250):
    pre0=cutoff-pd.Timedelta(weeks=26); recent0=cutoff-pd.Timedelta(weeks=4); target=cutoff+pd.Timedelta(weeks=h)
    pre=good[(good.week>=pre0)&(good.week<cutoff)].copy()
    recent=good[(good.week>=recent0)&(good.week<cutoff)]
    cnt=pre.groupby(store_col).week.nunique(); rcnt=recent.groupby(store_col).week.nunique()
    risk=sorted(set(cnt[cnt>=18].index)&set(rcnt[rcnt>=2].index))
    if len(risk)<25: return None
    surv=[s for s in risk if last_map.get(s,pd.Timestamp.min)>=target]
    exits=[s for s in risk if last_map.get(s,pd.Timestamp.max)<target]
    if len(exits)<3 or len(surv)<20: return None
    p=pre[pre[store_col].isin(risk)]
    base=p.groupby(store_col).agg(y=('total_sales','mean'),cust_mean=('cust','mean'),cust_sd=('cust','std'),sales_sd=('total_sales','std'))
    base['cust_cv']=base.cust_sd/base.cust_mean.replace(0,np.nan); base['sales_cv']=base.sales_sd/base.y.replace(0,np.nan)
    dm=p.groupby(store_col)[sales_cols].mean(); shares=dm.div(dm.sum(axis=1).replace(0,np.nan),axis=0).add_prefix('share_')
    feat=base[['cust_mean','cust_cv','sales_cv']].join(shares).replace([np.inf,-np.inf],np.nan)
    feat=feat.fillna(feat.median(numeric_only=True)).fillna(0).loc[risk]
    X=pd.DataFrame(StandardScaler().fit_transform(feat),index=feat.index)
    y=base.loc[risk,'y'].to_dict()
    def bench(candidates,focals):
        candidates=list(candidates); out={}
        for s in focals:
            ch=[j for j in candidates if j!=s]; kk=min(k,len(ch))
            if kk<3: out[s]=np.nan; continue
            d=np.sqrt(((X.loc[ch].values-X.loc[s].values)**2).sum(axis=1)); peers=[ch[z] for z in np.argsort(d)[:kk]]
            out[s]=float(np.median([y[j] for j in peers]))
        return out
    focal=surv
    br=bench(risk,focal); bs=bench(surv,focal)
    rows=[]
    for s in focal:
        a,b,yi=br.get(s),bs.get(s),y.get(s)
        if all(np.isfinite(v) and v>0 for v in [a,b,yi]): rows.append((s,yi,a,b,math.log(yi/a),math.log(yi/b)))
    rr=pd.DataFrame(rows,columns=['store','y','b_risk','b_surv','g_risk','g_surv'])
    if len(rr)<15: return None
    rr['bench_shift']=np.log(rr.b_surv/rr.b_risk)
    rr['label_risk']=rr.g_risk<math.log(.90); rr['label_surv']=rr.g_surv<math.log(.90)
    n20=max(1,int(math.ceil(.2*len(rr)))); A=set(rr.nsmallest(n20,'g_risk').store); B=set(rr.nsmallest(n20,'g_surv').store)
    act={'med_signed_shift':float(rr.bench_shift.median()),'med_abs_shift':float(rr.bench_shift.abs().median()),'p90_abs_shift':float(rr.bench_shift.abs().quantile(.9)),'label_flip':float((rr.label_risk!=rr.label_surv).mean()),'spearman':float(spearmanr(rr.g_risk,rr.g_surv).statistic),'bottom20_jaccard':float(len(A&B)/len(A|B))}
    null=[]
    for _ in range(nperm):
        removed=set(rng.choice(risk,size=len(exits),replace=False).tolist()); cand=[s for s in risk if s not in removed]; bx=bench(cand,focal)
        vals=[]
        for _,row in rr.iterrows():
            v=bx.get(int(row.store),np.nan)
            if np.isfinite(v) and v>0: vals.append((int(row.store),math.log(v/row.b_risk),math.log(row.y/v)))
        nd=pd.DataFrame(vals,columns=['store','shift','g']); jj=rr[['store','g_risk']].merge(nd,on='store')
        if len(jj)>=10: null.append((float(jj['shift'].abs().median()),float(((jj.g_risk<math.log(.9))!=(jj.g<math.log(.9))).mean()),float(spearmanr(jj.g_risk,jj.g).statistic)))
    nul=pd.DataFrame(null,columns=['shift','flip','rho'])
    out={'cutoff':str(cutoff.date()),'target':str(target.date()),'n_risk':len(risk),'n_survivors':len(surv),'n_exits':len(exits),'n_focal':len(rr),**act,
         'null_p_med_abs_shift':float((nul['shift']>=act['med_abs_shift']).mean()),'null_p_label_flip':float((nul['flip']>=act['label_flip']).mean()),'null_p_spearman':float((nul['rho']<=act['spearman']).mean()),
         'exit_metadata_closed_share':float(np.mean([closed_map.get(s,False) for s in exits]))}
    return out,rr

results=[];details=[]
for c in pd.date_range(gmin+pd.Timedelta(weeks=52),gmax-pd.Timedelta(weeks=30),freq='13W'):
    z=make_snapshot(pd.Timestamp(c))
    if z:
        s,r=z; results.append(s); r['cutoff']=s['cutoff']; details.append(r)
res=pd.DataFrame(results); res.to_csv('dominick_survival_smoke_results.csv',index=False)
if details: pd.concat(details,ignore_index=True).to_csv('dominick_survival_smoke_detail.csv',index=False)
print('VALID_WINDOWS',len(res))
if len(res):
    print(res.to_string(index=False))
    agg={'median_n_exits':float(res.n_exits.median()),'median_med_abs_shift':float(res.med_abs_shift.median()),'median_label_flip':float(res.label_flip.median()),'median_spearman':float(res.spearman.median()),'median_bottom20_jaccard':float(res.bottom20_jaccard.median()),'share_shift_p05':float((res.null_p_med_abs_shift<=.05).mean()),'share_flip_p05':float((res.null_p_label_flip<=.05).mean())}
    print('AGG',json.dumps(agg))
else:
    print('DATA_GATE_FAIL')

lines=['# Dominick survival-conditioned benchmark smoke test','',f'- R object: {key}; rows: {len(df):,}',f'- Good-week stores: {good[store_col].nunique()}',f'- Week range: {gmin.date()} to {gmax.date()}',f'- Early endpoints >=26w: {int(endp.early_26w.sum())}',f'- Early endpoints also metadata Closed: {int(endp.closure_consistent_26w.sum())}',f'- Valid 26w windows: {len(res)}']
if len(res):
    lines += ['',f'- Median |log benchmark shift|: {res.med_abs_shift.median():.4f}',f'- Median label flip: {res.label_flip.median():.3f}',f'- Median rank Spearman: {res.spearman.median():.3f}',f'- Median bottom20 Jaccard: {res.bottom20_jaccard.median():.3f}',f'- Shift > random-deletion null at p<=.05: {(res.null_p_med_abs_shift<=.05).sum()}/{len(res)}']
Path('dominick_survival_smoke.md').write_text('\n'.join(lines))
