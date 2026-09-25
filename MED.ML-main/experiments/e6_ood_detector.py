# E7: can simple scores separate in-distribution (ID) complaints from shifted (OOD) ones?
import numpy as np, pandas as pd, json, re
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from specialist_mapping import DISEASE_TO_SPECIALIST as M
from common import *
LV=['routine','urgent','emergency']; urg=lambda d: LV.index(M[d]['urgency'])
df=clean(load()); g=groups(df,0.8); ood=pd.read_csv('ood_test_set.csv')
tok=lambda s: re.findall(r'[a-z]+', s.lower())
rows=[]
for f,(tr,te) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=0).split(df.text,df.label,g)):
    Xtr,ytr=df.text.iloc[tr],df.label.iloc[tr]
    m=MODELS['Cal. LinearSVC (word+char)']().fit(Xtr,ytr)
    wv=word_tfidf().fit(Xtr); cv=char_tfidf().fit(Xtr); Wtr=wv.transform(Xtr); Ctr=cv.transform(Xtr)
    vocab=set(t for s in Xtr for t in tok(s))
    for src,X,y in [('id',df.text.iloc[te],df.label.iloc[te].values),('ood',ood.text,ood.label.values)]:
        P=m.predict_proba(X); pred=m.classes_[P.argmax(1)]
        simw=cosine_similarity(wv.transform(X),Wtr).max(1); simc=cosine_similarity(cv.transform(X),Ctr).max(1)
        cov=[np.mean([t in vocab for t in tok(s)]) for s in X]
        for i in range(len(X)):
            rows.append(dict(fold=f,src=src,conf=P[i].max(),simw=simw[i],simc=simc[i],vcov=cov[i],
                             correct=pred[i]==y[i],under=urg(pred[i])<urg(y[i])))
R=pd.DataFrame(rows); R.to_csv('e7_scores.csv',index=False)
isood=(R.src=='ood').astype(int); out={'auroc':{}}
for s in ['conf','simw','simc','vcov']: out['auroc'][s]=float(roc_auc_score(isood,-R[s]))
print('AUROC (ID vs OOD):',{k:round(v,3) for k,v in out['auroc'].items()})
print(R.groupby('src')[['conf','simw','simc','vcov']].describe().T.loc[(slice(None),['25%','50%','75%']),:].round(3))
# gates: threshold on sim chosen so that 95% / 90% of ID complaints pass
res=[]
def ev(name,pas):
    I=R[R.src=='id']; O=R[R.src=='ood']; pi=pas[R.src=='id']; po=pas[R.src=='ood']
    res.append(dict(gate=name,id_coverage=pi.mean(),id_acc_answered=I.correct[pi].mean(),
        ood_pass=po.mean(),ood_acc_answered=O.correct[po].mean() if po.any() else np.nan,
        ood_under_triage_overall=(O.under&po).mean()))
ev('none',np.ones(len(R),bool)); ev('conf>=0.5',R.conf>=.5)
for s in ['simw','simc']:
    for q in [0.05,0.10]:
        t=R[R.src=='id'][s].quantile(q); out[f'{s}_thr_q{q}']=float(t)
        ev(f'{s}>={t:.2f} (ID q{int(q*100)})',R[s]>=t); ev(f'conf>=0.5 & {s}>={t:.2f}',(R.conf>=.5)&(R[s]>=t))
T=pd.DataFrame(res); print(T.round(3).to_string()); out['gates']=T.to_dict('records'); json.dump(out,open('e7.json','w'),indent=1,default=float)
