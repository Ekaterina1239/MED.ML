"""
ML-сервис триажа симптомов. Отдельный Python-сервис, вызывается из C# бэкенда по HTTP.

Запуск:
    uvicorn main:app --host 0.0.0.0 --port 8001

Проверка:
    curl -X POST http://localhost:8001/predict \\
        -H "Content-Type: application/json" \\
        -d '{"symptoms_text": "I have a rash and itchy skin on my arms", "language": "en"}'

Защита от ошибок (подробности в README):
    is_low_confidence       — уверенность модели ниже порога
    is_out_of_distribution  — жалоба не похожа на обучающие данные
    red_flags               — опасные симптомы, срочность поднимается независимо от модели
    needs_doctor_review     — итог: не показывать пациенту диагноз как рекомендацию
"""

import os
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from sklearn.metrics.pairwise import cosine_similarity

from red_flags import check_red_flags, escalate
from specialist_mapping import get_specialist_info
from translation import SUPPORTED_LANGUAGES, translate_to_english

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "model"))

# Ниже этого порога модель считается недостаточно уверенной.
LOW_CONFIDENCE_THRESHOLD = 0.5
TOP_K = 3

artifacts: dict = {}


def load_artifacts(model_dir: str = MODEL_DIR) -> None:
    paths = {name: os.path.join(model_dir, f"{name}.joblib")
             for name in ("triage_model", "vectorizer", "ood_reference")}
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise RuntimeError(
            f"Не найдены файлы модели: {missing}. Сначала запустите training/train_model.py"
        )
    for name, path in paths.items():
        artifacts[name] = joblib.load(path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield
    artifacts.clear()


app = FastAPI(title="Triage ML Service", version="2.1", lifespan=lifespan)


class SymptomsRequest(BaseModel):
    symptoms_text: str
    language: str = "ru"  # ru | uz | en

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(f"language должен быть одним из {sorted(SUPPORTED_LANGUAGES)}")
        return v


class Candidate(BaseModel):
    disease: str
    specialist: str
    probability: float


class TriagePrediction(BaseModel):
    predicted_disease: str
    specialist: str
    urgency: str                   # итоговая срочность (с учётом красных флагов)
    model_urgency: str             # срочность только по диагнозу модели
    confidence_score: float
    is_low_confidence: bool
    similarity_score: float        # сходство с ближайшей обучающей жалобой (0..1)
    is_out_of_distribution: bool   # жалоба не похожа на то, на чём училась модель
    red_flags: list[str]           # сработавшие правила опасных симптомов
    needs_doctor_review: bool      # итоговый флаг: не показывать пациенту как рекомендацию
    top_candidates: list[Candidate]  # несколько наиболее вероятных диагнозов — для врача
    translated_text: str           # что реально ушло в модель — для отладки и аудита
    model_type: str = "symptom_triage_v2"


def similarity_to_training(text: str) -> float:
    ref = artifacts["ood_reference"]
    return float(cosine_similarity(ref["vectorizer"].transform([text]), ref["train_matrix"]).max())


@app.post("/predict", response_model=TriagePrediction)
def predict(request: SymptomsRequest):
    if not request.symptoms_text or not request.symptoms_text.strip():
        raise HTTPException(status_code=400, detail="symptoms_text не может быть пустым")

    try:
        english_text = translate_to_english(request.symptoms_text, request.language)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Ошибка перевода: {e}")

    model = artifacts["triage_model"]
    probabilities = model.predict_proba(artifacts["vectorizer"].transform([english_text]))[0]
    ranked = probabilities.argsort()[::-1]
    predicted_disease = str(model.classes_[ranked[0]])
    confidence = float(probabilities[ranked[0]])

    info = get_specialist_info(predicted_disease, request.language)
    similarity = similarity_to_training(english_text)
    flags, flag_urgency = check_red_flags(english_text)

    is_low_confidence = confidence < LOW_CONFIDENCE_THRESHOLD
    is_ood = similarity < artifacts["ood_reference"]["similarity_threshold"]

    return TriagePrediction(
        predicted_disease=predicted_disease,
        specialist=info["specialist"],
        urgency=escalate(info["urgency"], flag_urgency),
        model_urgency=info["urgency"],
        confidence_score=round(confidence, 4),
        is_low_confidence=is_low_confidence,
        similarity_score=round(similarity, 4),
        is_out_of_distribution=is_ood,
        red_flags=flags,
        needs_doctor_review=is_low_confidence or is_ood or bool(flags),
        top_candidates=[
            Candidate(
                disease=str(model.classes_[i]),
                specialist=get_specialist_info(str(model.classes_[i]), request.language)["specialist"],
                probability=round(float(probabilities[i]), 4),
            )
            for i in ranked[:TOP_K]
        ],
        translated_text=english_text,
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "triage_model" in artifacts}
