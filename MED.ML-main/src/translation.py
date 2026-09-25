"""
Перевод жалобы пациента на английский перед подачей в модель.

Модель обучена только на английских текстах (Symptom2Disease). Вместо того
чтобы переобучать модель на трёх языках (для этого нужен размеченный датасет
на каждом языке, которого у нас нет), переводим входной текст на английский
перед классификацией — это работает надёжно даже на маленьком количестве
поддерживаемых языков и не требует ничего размечать заново.

Используется deep-translator (обёртка над бесплатным Google Translate) —
не требует API-ключа и регистрации, только интернет-соединение. Не такой
точный на медицинских терминах, как перевод через LLM, но для демо и без
затрат на API этого достаточно.
"""

from deep_translator import GoogleTranslator

SUPPORTED_LANGUAGES = {"ru", "uz", "en"}


def translate_to_english(text: str, source_language: str) -> str:
    """
    Переводит текст на английский. Если source_language == 'en', возвращает
    текст как есть, без обращения к переводчику (быстрее).
    """
    if source_language not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Язык '{source_language}' не поддерживается. "
            f"Поддерживаются: {sorted(SUPPORTED_LANGUAGES)}"
        )

    if source_language == "en":
        return text

    try:
        translated = GoogleTranslator(source=source_language, target="en").translate(text)
    except Exception as e:
        raise RuntimeError(f"Ошибка перевода (проверьте интернет-соединение): {e}")

    if not translated:
        raise RuntimeError("Переводчик вернул пустой результат")

    return translated