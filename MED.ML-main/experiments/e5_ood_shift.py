# E5: out-of-distribution test. Rows of the 773-disease symptom matrix are rendered into
# patient-style sentences for diseases that also exist in Symptom2Disease.
import numpy as np, pandas as pd, json
from sklearn.metrics import accuracy_score, f1_score
from common import *
MAP={'acne':'Acne','allergy':'allergy','osteoarthritis':'Arthritis','asthma':'Bronchial Asthma','spondylosis':'Cervical spondylosis',
 'chickenpox':'Chicken pox','common cold':'Common Cold','hemorrhoids':'Dimorphic Hemorrhoids','fungal infection of the skin':'Fungal infection',
 'gastroesophageal reflux disease (gerd)':'gastroesophageal reflux disease','gastroduodenal ulcer':'peptic ulcer disease','impetigo':'Impetigo',
 'malaria':'Malaria','migraine':'Migraine','pneumonia':'Pneumonia','psoriasis':'Psoriasis','urinary tract infection':'urinary tract infection',
 'varicose veins':'Varicose Veins','drug reaction':'drug reaction'}
big=pd.read_csv(os.path.join(DATA_DIR,'Final_Augmented_dataset_Diseases_and_Symptoms.csv'))
big=big[big.diseases.isin(MAP)]
rng=np.random.default_rng(0)
sub=pd.concat([x.sample(min(len(x),100),random_state=0) for _,x in big.groupby('diseases')])
cols=np.array(big.columns[1:])
T=["I have {}.","For the past few days I have been having {}.","I am suffering from {}.","My symptoms are {}.","Recently I noticed {}."]
def render(row):
    s=list(cols[row[cols].values.astype(bool)]); rng.shuffle(s)
    if not s: return None
    body=s[0] if len(s)==1 else ', '.join(s[:-1])+' and '+s[-1]
    return T[rng.integers(len(T))].format(body)
sub=sub.assign(text=[render(r) for _,r in sub.iterrows()], label=sub.diseases.map(MAP)).dropna(subset=['text'])
sub=sub.drop_duplicates('text')
print('OOD samples',len(sub),'classes',sub.label.nunique(),'mean symptoms/row',big.iloc[:,1:].sum(1).mean().round(2))
print(sub.text.head(3).tolist())
tr=clean(load()); out={}
for name in ['LogReg (word)','Cal. LinearSVC (word) [baseline]','Cal. LinearSVC (char)','Cal. LinearSVC (word+char)']:
    m=MODELS[name]().fit(tr.text,tr.label); P=m.predict_proba(sub.text); pred=m.classes_[P.argmax(1)]; c=P.max(1); k=pred==sub.label.values
    out[name]={'acc':float(k.mean()),'f1':float(f1_score(sub.label,pred,average='macro')),'mean_conf':float(c.mean()),
               'cov@0.5':float((c>=.5).mean()),'acc@0.5':float(k[c>=.5].mean()) if (c>=.5).any() else None}
    if name.endswith('[baseline]'):
        per=pd.DataFrame({'y':sub.label.values,'k':k}).groupby('y').k.mean().sort_values(); out['per_class_baseline']=per.round(2).to_dict()
print(json.dumps(out,indent=1)); json.dump(out,open('e5.json','w'),indent=1)
sub[['label','text']].to_csv('ood_test_set.csv',index=False)
