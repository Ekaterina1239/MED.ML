"""
ML-сервис триажа симптомов. Отдельный Python-сервис, вызывается из C# бэкенда по HTTP.

Запуск:
    uvicorn main:app --host 0.0.0.0 --port 8001

Проверка:
    curl -X POST http://localhost:8001/predict \
        -H "Content-Type: application/json" \
        -d '{"symptoms_text": "I have a rash and itchy skin on my arms"}'
"""

import os

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from specialist_mapping import get_specialist_info
from translation import SUPPORTED_LANGUAGES, translate_to_english

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")

app = FastAPI(title="Triage ML Service", version="1.0")

model = None
vectorizer = None


@app.on_event("startup")
def load_model():
    global model, vectorizer
    model_path = os.path.join(MODEL_DIR, "triage_model.joblib")
    vectorizer_path = os.path.join(MODEL_DIR, "vectorizer.joblib")

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise RuntimeError(
            f"Модель не найдена в {MODEL_DIR}. Сначала запустите training/train_model.py"
        )

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)


class SymptomsRequest(BaseModel):
    symptoms_text: str
    language: str = "ru"  # ru | uz | en

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        if v not in SUPPORTED_LANGUAGES:
            raise ValueError(f"language должен быть одним из {sorted(SUPPORTED_LANGUAGES)}")
        return v


class TriagePrediction(BaseModel):
    predicted_disease: str
    specialist: str
    urgency: str
    confidence_score: float
    is_low_confidence: bool
    translated_text: str  # что реально ушло в модель — полезно для отладки и аудита
    model_type: str = "symptom_triage_v1"


# Ниже этого порога модель считается недостаточно уверенной — такие случаи
# должны обязательно эскалироваться на ручной осмотр врача, а не автоматически
# отдаваться пациенту как окончательная рекомендация.
LOW_CONFIDENCE_THRESHOLD = 0.5


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

    X = vectorizer.transform([english_text])
    predicted_disease = model.predict(X)[0]

    # calibrated LinearSVC даёт predict_proba — берём максимальную вероятность как confidence
    probabilities = model.predict_proba(X)[0]
    confidence = float(max(probabilities))

    info = get_specialist_info(predicted_disease, request.language)

    return TriagePrediction(
        predicted_disease=predicted_disease,
        specialist=info["specialist"],
        urgency=info["urgency"],
        confidence_score=round(confidence, 4),
        is_low_confidence=confidence < LOW_CONFIDENCE_THRESHOLD,
        translated_text=english_text,
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}