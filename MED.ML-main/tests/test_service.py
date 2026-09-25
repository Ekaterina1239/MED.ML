"""
Тесты сервиса. Запуск из корня проекта (нужна обученная модель в model/):
    pytest -q

Перевод в тестах подменяется заглушкой, поэтому интернет не нужен.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import main  # noqa: E402
from red_flags import check_red_flags, escalate  # noqa: E402

MODEL_READY = os.path.exists(os.path.join(main.MODEL_DIR, "triage_model.joblib"))
needs_model = pytest.mark.skipif(not MODEL_READY, reason="сначала обучите модель: training/train_model.py")

FAKE_TRANSLATIONS = {
    "У меня на локтях и коленях красные пятна с серебристыми чешуйками, чешутся и кожа трескается.":
        "I have red spots with silvery scales on my elbows and knees, itchy and the skin cracks.",
}


@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    def fake_translate(text, lang):
        return text if lang == "en" else FAKE_TRANSLATIONS.get(text, text)

    monkeypatch.setattr(main, "translate_to_english", fake_translate)
    with TestClient(main.app) as c:
        yield c


# --- красные флаги (модель не нужна) -------------------------------------------

@pytest.mark.parametrize("text, flag, level", [
    ("I have chest pain and I can't breathe", "chest_pain", "emergency"),
    ("my lips swelled after eating nuts", "anaphylaxis", "emergency"),
    ("I am coughing up blood", "coughing_blood", "emergency"),
    ("fever 39.5 degrees for two days", "high_or_long_fever", "urgent"),
])
def test_red_flags_detected(text, flag, level):
    flags, urgency = check_red_flags(text)
    assert flag in flags and urgency == level


@pytest.mark.parametrize("text", [
    "I am 39 years old and have acne on my face",
    "itchy rash on my arms",
])
def test_no_false_red_flags(text):
    assert check_red_flags(text) == ([], None)


def test_escalate_never_lowers_urgency():
    assert escalate("emergency", "urgent") == "emergency"
    assert escalate("routine", "emergency") == "emergency"
    assert escalate("urgent", None) == "urgent"


# --- API ------------------------------------------------------------------------

@needs_model
def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


@needs_model
def test_in_distribution_complaint_is_answered(client):
    text = ("I have been experiencing a skin rash on my arms, legs, and torso for the past few weeks. "
            "It is red, itchy, and covered in dry, scaly patches.")
    r = client.post("/predict", json={"symptoms_text": text, "language": "en"}).json()
    assert r["predicted_disease"] == "Psoriasis"
    assert not r["is_out_of_distribution"] and not r["needs_doctor_review"]
    assert len(r["top_candidates"]) == 3
    assert r["top_candidates"][0]["disease"] == "Psoriasis"


@needs_model
def test_red_flag_escalates_and_requires_doctor(client):
    r = client.post("/predict", json={
        "symptoms_text": "High fever, cough with yellow phlegm and chest pain when I breathe",
        "language": "en"}).json()
    assert r["urgency"] == "emergency"
    assert "chest_pain" in r["red_flags"]
    assert r["needs_doctor_review"]


@needs_model
def test_russian_goes_through_translation_and_specialist_is_localized(client):
    text = next(iter(FAKE_TRANSLATIONS))
    r = client.post("/predict", json={"symptoms_text": text, "language": "ru"}).json()
    assert r["translated_text"] == FAKE_TRANSLATIONS[text]
    assert r["specialist"] == "дерматолог"


@needs_model
@pytest.mark.parametrize("payload, status", [
    ({"symptoms_text": "   ", "language": "en"}, 400),
    ({"symptoms_text": "rash", "language": "fr"}, 422),
])
def test_bad_requests(client, payload, status):
    assert client.post("/predict", json=payload).status_code == status
