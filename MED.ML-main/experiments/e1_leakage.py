# E1: effect of duplicates / near-duplicates on reported accuracy
import numpy as np, json
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score
from common import *
df=load(); res={}
m=MODELS['Cal. LinearSVC (word) [baseline]']
# original protocol
Xtr,Xte,ytr,yte=train_test_split(df.text,df.label,test_size=.2,stratify=df.label,random_state=42)
p=m().fit(Xtr,ytr).predict(Xte); res['orig_single_split']=accuracy_score(yte,p)
te_set=set(Xte.str.lower().str.strip()); tr_set=set(Xtr.str.lower().str.strip())
res['orig_test_exact_in_train']=int(sum(t in tr_set for t in Xte.str.lower().str.strip()))
def cv(d, grp=None, reps=5):
    accs,f1s=[],[]
    for r in range(reps):
        sp = StratifiedGroupKFold(5,shuffle=True,random_state=r).split(d.text,d.label,grp) if grp is not None else StratifiedKFold(5,shuffle=True,random_state=r).split(d.text,d.label)
        for tr,te in sp:
            p=m().fit(d.text.iloc[tr],d.label.iloc[tr]).predict(d.text.iloc[te])
            accs.append(accuracy_score(d.label.iloc[te],p)); f1s.append(f1_score(d.label.iloc[te],p,average='macro'))
    return float(np.mean(accs)),float(np.std(accs)),float(np.mean(f1s)),float(np.std(f1s))
res['raw_cv']=cv(df)
dc=clean(df); res['n_after_exact_dedup']=len(dc); res['dedup_cv']=cv(dc)
for th in [0.9,0.8,0.7]:
    g=groups(dc,th); res[f'group_cv_th{th}']=cv(dc,g); res[f'n_groups_th{th}']=int(len(set(g)))
print(json.dumps(res,indent=1)); json.dump(res,open('e1.json','w'),indent=1)
