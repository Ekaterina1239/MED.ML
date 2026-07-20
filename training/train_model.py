"""
Обучение модели классификации симптомов -> диагноз.

Датасет: Symptom2Disease.csv (1200 строк, 24 сбалансированных класса по 50 примеров).
Модель: TF-IDF векторизация + калиброванный LinearSVC (даёт вероятности для confidence score).

Запуск:
    python train_model.py --data path/to/Symptom2Disease.csv --output ../model/

Результат: два файла в --output:
    triage_model.joblib   — обученный классификатор
    vectorizer.joblib     — TF-IDF векторизатор (нужен для препроцессинга новых текстов)
"""

import argparse
import json
import os

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


def train(data_path: str, output_dir: str, test_size: float = 0.2, random_state: int = 42):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    if "label" not in df.columns or "text" not in df.columns:
        raise ValueError(
            f"Ожидались колонки 'label' и 'text', получены: {df.columns.tolist()}"
        )

    X = df["text"]
    y = df["label"]

    print(f"Всего примеров: {len(df)}, классов: {y.nunique()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        stop_words="english",
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # LinearSVC не даёт вероятностей напрямую — оборачиваем в CalibratedClassifierCV,
    # чтобы получить confidence score для записи в MLPredictions.ConfidenceScore
    base_model = LinearSVC(C=1.0, max_iter=5000)
    model = CalibratedClassifierCV(base_model, cv=5)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\nAccuracy на тестовой выборке: {acc:.3f}\n")
    print(classification_report(y_test, y_pred))

    model_path = os.path.join(output_dir, "triage_model.joblib")
    vectorizer_path = os.path.join(output_dir, "vectorizer.joblib")
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    # Метрики сохраняем отдельно — пригодится и для диплома, и для мониторинга
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": acc,
                "n_samples": len(df),
                "n_classes": int(y.nunique()),
                "classes": sorted(y.unique().tolist()),
                "classification_report": report,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nSaved:")
    print(f"  {model_path}")
    print(f"  {vectorizer_path}")
    print(f"  {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a symptom-to-disease classification model.")
    parser.add_argument("--data", required=True, help="Link to Symptom2Disease.csv")
    parser.add_argument("--output", default="../model", help="Folder to save model and vectorizer")
    args = parser.parse_args()

    train(args.data, args.output)