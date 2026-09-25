"""
E8: влияют ли красные флаги на заниженную срочность (under-triage) и сколько
лишних срабатываний они дают.

Проверка на двух наборах, которые НЕ использовались для составления правил:
  - Symptom2Disease (свои жалобы, групповая CV) — ложные тревоги
  - ood_test_set.csv (1813 жалоб из другого датасета, сначала запустить e5_ood_shift.py)
"""
import json, os, sys
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from specialist_mapping import DISEASE_TO_SPECIALIST as M
from red_flags import check_red_flags, escalate, LEVELS
from common import MODELS, clean, load, groups

U = lambda d: M[d]['urgency']
NAME = 'Cal. LinearSVC (word+char)'

def evaluate(texts, true, pred):
    flags = [check_red_flags(t) for t in texts]
    base = np.array([U(p) for p in pred]); true_u = np.array([U(t) for t in true])
    esc = np.array([escalate(b, f[1]) for b, f in zip(base, flags)])
    lv = lambda a: np.array([LEVELS.index(x) for x in a])
    return {
        'n': len(texts),
        'flag_rate': float(np.mean([bool(f[0]) for f in flags])),
        'under_triage_model': float((lv(base) < lv(true_u)).mean()),
        'under_triage_with_flags': float((lv(esc) < lv(true_u)).mean()),
        'over_triage_model': float((lv(base) > lv(true_u)).mean()),
        'over_triage_with_flags': float((lv(esc) > lv(true_u)).mean()),
        'emergency_recall_model': float((lv(base)[true_u == 'emergency'] == 2).mean()) if (true_u == 'emergency').any() else None,
        'emergency_recall_with_flags': float((lv(esc)[true_u == 'emergency'] == 2).mean()) if (true_u == 'emergency').any() else None,
        'flag_counts': pd.Series([n for f in flags for n in f[0]]).value_counts().to_dict(),
    }

df = clean(load()); g = groups(df, 0.8)
texts, true, pred = [], [], []
for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(df.text, df.label, g):
    m = MODELS[NAME]().fit(df.text.iloc[a], df.label.iloc[a])
    texts += df.text.iloc[b].tolist(); true += df.label.iloc[b].tolist(); pred += m.predict(df.text.iloc[b]).tolist()
res = {'in_distribution': evaluate(texts, true, pred)}

ood = pd.read_csv('ood_test_set.csv')
m = MODELS[NAME]().fit(df.text, df.label)
res['ood'] = evaluate(ood.text.tolist(), ood.label.tolist(), m.predict(ood.text).tolist())

json.dump(res, open('e8_red_flags.json', 'w'), indent=1)
for k, v in res.items():
    print(f'\n{k}:'); [print(f'  {a}: {b}') for a, b in v.items()]
