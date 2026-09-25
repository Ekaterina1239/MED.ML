"""
Сопоставление диагноза (метки из Symptom2Disease) со специалистом и уровнем срочности.

ВНИМАНИЕ: это черновой словарь, составленный для демо на основе общей медицинской
логики, НЕ провалидирован практикующим врачом. Перед реальным использованием
каждая строка должна быть проверена.

Название специалиста хранится готовыми переводами на 3 языках (ru/uz/en), а не
переводится налету через LLM — для медицинских терминов важна консистентность
(один и тот же специалист должен называться одинаково при каждом запросе),
а список специальностей конечный и небольшой, так что легче держать таблицу.

urgency: routine | urgent | emergency — код, не переводится, перевод сообщения
для пациента на этот статус делается на стороне C# бэка (см. README раздел
"Мультиязычность").
"""

DISEASE_TO_SPECIALIST = {
    "Acne": {
        "specialist": {"ru": "дерматолог", "uz": "dermatolog", "en": "dermatologist"},
        "urgency": "routine",
    },
    "Arthritis": {
        "specialist": {"ru": "ревматолог", "uz": "revmatolog", "en": "rheumatologist"},
        "urgency": "routine",
    },
    "Bronchial Asthma": {
        "specialist": {"ru": "пульмонолог", "uz": "pulmonolog", "en": "pulmonologist"},
        "urgency": "urgent",
    },
    "Cervical spondylosis": {
        "specialist": {"ru": "невролог", "uz": "nevrolog", "en": "neurologist"},
        "urgency": "routine",
    },
    "Chicken pox": {
        "specialist": {
            "ru": "терапевт / инфекционист",
            "uz": "terapevt / infeksionist",
            "en": "general practitioner / infectious disease specialist",
        },
        "urgency": "urgent",
    },
    "Common Cold": {
        "specialist": {"ru": "терапевт", "uz": "terapevt", "en": "general practitioner"},
        "urgency": "routine",
    },
    "Dengue": {
        "specialist": {"ru": "инфекционист", "uz": "infeksionist", "en": "infectious disease specialist"},
        "urgency": "emergency",
    },
    "Dimorphic Hemorrhoids": {
        "specialist": {"ru": "проктолог", "uz": "proktolog", "en": "proctologist"},
        "urgency": "routine",
    },
    "Fungal infection": {
        "specialist": {"ru": "дерматолог", "uz": "dermatolog", "en": "dermatologist"},
        "urgency": "routine",
    },
    "Hypertension": {
        "specialist": {"ru": "кардиолог", "uz": "kardiolog", "en": "cardiologist"},
        "urgency": "urgent",
    },
    "Impetigo": {
        "specialist": {"ru": "дерматолог", "uz": "dermatolog", "en": "dermatologist"},
        "urgency": "urgent",
    },
    "Jaundice": {
        "specialist": {
            "ru": "гастроэнтеролог / гепатолог",
            "uz": "gastroenterolog / gepatolog",
            "en": "gastroenterologist / hepatologist",
        },
        "urgency": "urgent",
    },
    "Malaria": {
        "specialist": {"ru": "инфекционист", "uz": "infeksionist", "en": "infectious disease specialist"},
        "urgency": "emergency",
    },
    "Migraine": {
        "specialist": {"ru": "невролог", "uz": "nevrolog", "en": "neurologist"},
        "urgency": "routine",
    },
    "Pneumonia": {
        "specialist": {"ru": "пульмонолог", "uz": "pulmonolog", "en": "pulmonologist"},
        "urgency": "emergency",
    },
    "Psoriasis": {
        "specialist": {"ru": "дерматолог", "uz": "dermatolog", "en": "dermatologist"},
        "urgency": "routine",
    },
    "Typhoid": {
        "specialist": {"ru": "инфекционист", "uz": "infeksionist", "en": "infectious disease specialist"},
        "urgency": "urgent",
    },
    "Varicose Veins": {
        "specialist": {
            "ru": "флеболог / сосудистый хирург",
            "uz": "flebolog / qon tomir jarrohi",
            "en": "phlebologist / vascular surgeon",
        },
        "urgency": "routine",
    },
    "allergy": {
        "specialist": {"ru": "аллерголог", "uz": "allergolog", "en": "allergist"},
        "urgency": "routine",
    },
    "diabetes": {
        "specialist": {"ru": "эндокринолог", "uz": "endokrinolog", "en": "endocrinologist"},
        "urgency": "urgent",
    },
    "drug reaction": {
        "specialist": {"ru": "аллерголог", "uz": "allergolog", "en": "allergist"},
        "urgency": "urgent",
    },
    "gastroesophageal reflux disease": {
        "specialist": {"ru": "гастроэнтеролог", "uz": "gastroenterolog", "en": "gastroenterologist"},
        "urgency": "routine",
    },
    "peptic ulcer disease": {
        "specialist": {"ru": "гастроэнтеролог", "uz": "gastroenterolog", "en": "gastroenterologist"},
        "urgency": "urgent",
    },
    "urinary tract infection": {
        "specialist": {"ru": "уролог", "uz": "urolog", "en": "urologist"},
        "urgency": "urgent",
    },
}

DEFAULT_SPECIALIST = {
    "specialist": {"ru": "терапевт", "uz": "terapevt", "en": "general practitioner"},
    "urgency": "routine",
}


def get_specialist_info(disease_label: str, language: str = "ru") -> dict:
    """
    Возвращает {specialist, urgency} для диагноза на нужном языке,
    или безопасный дефолт, если диагноз или язык неизвестны.
    """
    entry = DISEASE_TO_SPECIALIST.get(disease_label, DEFAULT_SPECIALIST)
    specialist_name = entry["specialist"].get(language, entry["specialist"]["en"])
    return {"specialist": specialist_name, "urgency": entry["urgency"]}