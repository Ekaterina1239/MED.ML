"""
E7: проверка всего конвейера сервиса на разговорных жалобах на трёх языках.

96 параллельных жалоб (24 диагноза x 4) из lay_complaints_en_ru_uz.py:
  en  -> сразу в модель
  ru  -> перевод через src/translation.py (Google Translate) -> модель
  uz  -> то же самое

Нужен интернет (для перевода). Запуск из папки experiments:
    python e7_multilingual.py

Результат:
    e7_multilingual.json          — метрики по языкам и моделям
    e7_translations.csv           — что вернул переводчик и что предсказала модель
"""
import json, os, sys, time
import numpy as np, pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from specialist_mapping import DISEASE_TO_SPECIALIST as M
from translation import translate_to_english
from lay_complaints_en_ru_uz import ROWS
from common import MODELS, clean, load, groups, char_tfidf

LEVELS = ['routine', 'urgent', 'emergency']
CONF_T = 0.5

lay = pd.DataFrame(ROWS, columns=['label', 'en', 'ru', 'uz'])
cache_path = 'e7_translation_cache.json'
cache = json.load(open(cache_path, encoding='utf-8')) if os.path.exists(cache_path) else {}
for lang in ['ru', 'uz']:
    out = []
    for i, text in enumerate(lay[lang]):
        key = f'{lang}::{text}'
        if key not in cache:
            for attempt in range(3):
                try:
                    cache[key] = translate_to_english(text, lang); break
                except Exception as e:
                    print('retry', attempt, e); time.sleep(2)
            else:
                raise SystemExit('Перевод не удался — проверьте интернет')
            json.dump(cache, open(cache_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
            time.sleep(0.3)
        out.append(cache[key])
        print(f'\r{lang}: {i + 1}/{len(lay)}', end='', flush=True)
    lay[f'{lang}_to_en'] = out
    print()

train = clean(load())
# порог сходства, как в обучении: 5-й перцентиль сходства "своих" жалоб в групповой CV
from sklearn.model_selection import StratifiedGroupKFold
g = groups(train, 0.8); sims = []
for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(train.text, train.label, g):
    v = char_tfidf().fit(train.text.iloc[a])
    sims += cosine_similarity(v.transform(train.text.iloc[b]), v.transform(train.text.iloc[a])).max(1).tolist()
SIM_T = float(np.quantile(sims, 0.05))
ood_vec = char_tfidf().fit(train.text); Tm = ood_vec.transform(train.text)

results = {'similarity_threshold': SIM_T}
for name in ['Cal. LinearSVC (word) [baseline]', 'Cal. LinearSVC (word+char)']:
    model = MODELS[name]().fit(train.text, train.label)
    results[name] = {}
    for lang, col in [('en', 'en'), ('ru', 'ru_to_en'), ('uz', 'uz_to_en')]:
        P = model.predict_proba(lay[col]); pred = model.classes_[P.argmax(1)]; conf = P.max(1)
        sim = cosine_similarity(ood_vec.transform(lay[col]), Tm).max(1)
        y = lay.label.values; ok = pred == y
        spec = np.array([M[p]['specialist']['en'] == M[t]['specialist']['en'] for p, t in zip(pred, y)])
        under = np.array([LEVELS.index(M[p]['urgency']) < LEVELS.index(M[t]['urgency']) for p, t in zip(pred, y)])
        gate = (conf >= CONF_T) & (sim >= SIM_T)
        results[name][lang] = {
            'accuracy': float(ok.mean()), 'specialist_accuracy': float(spec.mean()),
            'under_triage': float(under.mean()),
            'coverage_conf': float((conf >= CONF_T).mean()),
            'acc_answered_conf': float(ok[conf >= CONF_T].mean()) if (conf >= CONF_T).any() else None,
            'coverage_conf_and_sim': float(gate.mean()),
            'acc_answered_conf_and_sim': float(ok[gate].mean()) if gate.any() else None,
            'under_triage_passed_gate': float((under & gate).mean()),
            'mean_similarity': float(sim.mean()),
        }
        if name.endswith('(word+char)'):
            lay[f'{lang}_pred'] = pred; lay[f'{lang}_conf'] = conf.round(3); lay[f'{lang}_sim'] = sim.round(3)

json.dump(results, open('e7_multilingual.json', 'w'), indent=1)
lay.to_csv('e7_translations.csv', index=False, encoding='utf-8-sig')
for name in results:
    if name == 'similarity_threshold': continue
    print('\n' + name)
    print(pd.DataFrame(results[name]).round(3).to_string())
print('\nГотово: e7_multilingual.json, e7_translations.csv')
