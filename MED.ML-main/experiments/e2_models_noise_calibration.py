import numpy as np, json, time
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score
from common import *
df=clean(load()); g=groups(df,0.8)
NOISE=[0,0.05,0.10,0.20]
res={k:{'acc':[], 'f1':[], **{f'typo{n}':[] for n in NOISE[1:]}, 'ece':[], 'time':[]} for k in MODELS}
sel={'conf':[], 'correct':[]}; sel_char={'conf':[], 'correct':[]}
for r in range(3):
    for fold,(tr,te) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=r).split(df.text,df.label,g)):
        Xtr,ytr,Xte,yte=df.text.iloc[tr],df.label.iloc[tr],df.text.iloc[te],df.label.iloc[te].values
        rng=np.random.default_rng(100*r+fold)
        noisy={n:[typo(t,n,rng) for t in Xte] for n in NOISE[1:]}
        for name,mk in MODELS.items():
            t0=time.time(); m=mk().fit(Xtr,ytr); res[name]['time'].append(time.time()-t0)
            P=m.predict_proba(Xte); pred=m.classes_[P.argmax(1)]; conf=P.max(1); cor=(pred==yte)
            res[name]['acc'].append(cor.mean()); res[name]['f1'].append(f1_score(yte,pred,average='macro'))
            res[name]['ece'].append(ece(conf,cor.astype(float)))
            for n in NOISE[1:]: res[name][f'typo{n}'].append(accuracy_score(yte,m.predict(noisy[n])))
            if name.endswith('[baseline]'):
                sel['conf']+=conf.tolist(); sel['correct']+=cor.tolist()
                P2=m.predict_proba(noisy[0.10]); sel['conf_n10']=sel.get('conf_n10',[])+P2.max(1).tolist(); sel['correct_n10']=sel.get('correct_n10',[])+(m.classes_[P2.argmax(1)]==yte).tolist()
            if name=='Cal. LinearSVC (word+char)':
                P2=m.predict_proba(noisy[0.10]); sel_char['conf_n10']=sel_char.get('conf_n10',[])+P2.max(1).tolist(); sel_char['correct_n10']=sel_char.get('correct_n10',[])+(m.classes_[P2.argmax(1)]==yte).tolist()
                sel_char['conf']+=P.max(1).tolist(); sel_char['correct']+=cor.tolist()
    print('rep',r,'done',flush=True)
summ={k:{m:[float(np.mean(v)),float(np.std(v))] for m,v in d.items()} for k,d in res.items()}
json.dump({'summary':summ,'sel':sel,'sel_char':sel_char},open('e234.json','w'))
for k,d in summ.items(): print(f"{k:34s} acc {d['acc'][0]*100:.2f}±{d['acc'][1]*100:.2f} F1 {d['f1'][0]*100:.2f} typo5 {d['typo0.05'][0]*100:.1f} typo10 {d['typo0.1'][0]*100:.1f} typo20 {d['typo0.2'][0]*100:.1f} ECE {d['ece'][0]:.3f} t {d['time'][0]:.2f}s")
