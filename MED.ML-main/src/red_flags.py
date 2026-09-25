"""
Правила «красных флагов»: опасные симптомы, при которых пациента нужно направить
к врачу срочно, независимо от того, что предсказала модель.

Правила проверяются на английском тексте (после перевода), поэтому один список
работает для ru/uz/en. Модель может ошибиться в диагнозе, но если в жалобе есть,
например, боль в груди и одышка, срочность всё равно будет поднята.

ВНИМАНИЕ: черновой список на основе общих принципов неотложной помощи,
НЕ проверен врачом. Перед реальным использованием его должен просмотреть клиницист.
"""

import re

LEVELS = ["routine", "urgent", "emergency"]

# (название, уровень, регулярное выражение по английскому тексту)
RED_FLAGS = [
    ("chest_pain", "emergency", r"chest (pain|pressure|tightness)|pain in (my|the) chest|(sharp|crushing|severe) chest"),
    ("breathing_difficulty", "emergency",
     r"short(ness)? (of|in) breath|difficulty breathing|hard to breathe|(can ?not|can't|unable to) breathe"
     r"|breathless|breathing (is )?(fast|rapid|difficult)|gasping|suffocat"),
    ("coughing_blood", "emergency", r"cough(ing)? (up )?blood|blood in (my )?(sputum|phlegm)|bloody (sputum|phlegm)|rusty sputum"),
    ("gi_bleeding", "emergency", r"vomit(ing)? blood|blood in (my )?vomit|black (stool|stools|feces)|tarry stool"),
    ("altered_consciousness", "emergency", r"confus|faint|passed out|loss of consciousness|unconscious|seizure|convulsion"),
    ("stroke_signs", "emergency", r"slurred speech|face droop|one side of (my|the) body|numb(ness)? on one side|paralys"),
    ("anaphylaxis", "emergency", r"(lips?|tongue|throat|face) (is |are |got |became )?swell|swelling of (the |my )?(lips|tongue|throat|face)|throat (is )?closing"),
    ("meningism", "emergency", r"stiff neck.*fever|fever.*stiff neck"),
    ("severe_headache", "urgent", r"worst headache|sudden (severe )?headache|thunderclap"),
    ("high_or_long_fever", "urgent", r"high fever|fever (for|since) (a|one|two|several|\d+) (week|weeks)|\b(39|40)([.,]\d)? ?(°|degrees|c\b)"),
    ("bleeding_gums_bruising", "urgent", r"gums? (bleed|bleeding)|bleeding gums|bruis"),
    ("jaundice_signs", "urgent", r"yellow(ish)? (skin|eyes)|(skin|eyes) (turned |look |are )?yellow"),
    ("suicidal", "emergency", r"suicid|kill myself|end my life"),
]
_COMPILED = [(n, lvl, re.compile(p, re.IGNORECASE)) for n, lvl, p in RED_FLAGS]


def check_red_flags(english_text: str) -> tuple[list[str], str | None]:
    """Возвращает (список сработавших флагов, максимальный уровень срочности или None)."""
    hits = [(n, lvl) for n, lvl, rx in _COMPILED if rx.search(english_text)]
    if not hits:
        return [], None
    return [n for n, _ in hits], max((lvl for _, lvl in hits), key=LEVELS.index)


def escalate(model_urgency: str, flag_urgency: str | None) -> str:
    """Итоговая срочность: максимум из срочности модели и красных флагов."""
    if flag_urgency is None:
        return model_urgency
    return max(model_urgency, flag_urgency, key=LEVELS.index)
