"""
Обучение модели классификации: текст жалобы -> диагноз (24 класса Symptom2Disease).

Что делает скрипт:
  1. Удаляет точные дубликаты жалоб (в Symptom2Disease их 47).
  2. Оценивает модель групповой 5-кратной кросс-валидацией: похожие жалобы
     (косинусное сходство TF-IDF >= 0.8) всегда попадают в один фолд, поэтому
     почти одинаковые тексты не оказываются одновременно в train и test.
  3. На тех же фолдах подбирает порог «похожести» для детектора незнакомых
     жалоб (out-of-distribution): 5-й перцентиль сходства тестовых жалоб
     с обучающими. Жалобы ниже порога сервис отправляет к врачу.
  4. Обучает финальную модель на всех данных и сохраняет артефакты.

Признаки (--features):
  word       — TF-IDF по словам (1-2-граммы), исходная версия
  char       — TF-IDF по символьным 2-5-граммам, устойчив к опечаткам
  word+char  — объединение (по умолчанию: лучшая точность и устойчивость)

Запуск:
    python train_model.py --data path/to/Symptom2Disease.csv --output ../model

Результат в --output:
    triage_model.joblib    — калиброванный LinearSVC
    vectorizer.joblib      — векторизатор признаков для модели
    ood_reference.joblib   — символьный векторизатор, матрица обучающих жалоб и порог сходства
    training_metrics.json  — метрики кросс-валидации
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_union
from sklearn.svm import LinearSVC

GROUP_SIMILARITY = 0.8   # порог объединения похожих жалоб в группу для кросс-валидации
OOD_QUANTILE = 0.05      # доля «своих» жалоб, которые детектор допускает отклонить


def make_vectorizer(features: str):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english", sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    return {"word": word, "char": char, "word+char": make_union(word, char)}[features]


def make_ood_vectorizer():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)


def make_model():
    # LinearSVC не даёт вероятностей — калибровка нужна для confidence_score
    return CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=5000), cv=5)


def similarity_groups(texts: pd.Series, threshold: float) -> np.ndarray:
    matrix = TfidfVectorizer().fit_transform(texts.str.lower())
    _, groups = connected_components(cosine_similarity(matrix) >= threshold, directed=False)
    return groups


def train(data_path: str, output_dir: str, features: str = "word+char", random_state: int = 0):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    if "label" not in df.columns or "text" not in df.columns:
        raise ValueError(f"Ожидались колонки 'label' и 'text', получены: {df.columns.tolist()}")
    n_raw = len(df)
    df = df.loc[~df["text"].str.lower().str.strip().duplicated(), ["label", "text"]].reset_index(drop=True)
    print(f"Примеров: {n_raw}, после удаления дубликатов: {len(df)}, классов: {df['label'].nunique()}")

    groups = similarity_groups(df["text"], GROUP_SIMILARITY)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)

    y_true, y_pred, fold_acc, fold_f1, id_similarity = [], [], [], [], []
    for train_idx, test_idx in cv.split(df["text"], df["label"], groups):
        X_tr, X_te = df["text"].iloc[train_idx], df["text"].iloc[test_idx]
        y_tr, y_te = df["label"].iloc[train_idx], df["label"].iloc[test_idx]

        vec = make_vectorizer(features)
        model = make_model().fit(vec.fit_transform(X_tr), y_tr)
        pred = model.predict(vec.transform(X_te))
        y_true += y_te.tolist(); y_pred += pred.tolist()
        fold_acc.append(accuracy_score(y_te, pred)); fold_f1.append(f1_score(y_te, pred, average="macro"))

        ood_vec = make_ood_vectorizer().fit(X_tr)
        id_similarity += cosine_similarity(ood_vec.transform(X_te), ood_vec.transform(X_tr)).max(axis=1).tolist()

    similarity_threshold = float(np.quantile(id_similarity, OOD_QUANTILE))
    print(f"\nГрупповая 5-кратная CV: accuracy {np.mean(fold_acc):.3f} ± {np.std(fold_acc):.3f}, "
          f"macro-F1 {np.mean(fold_f1):.3f}")
    print(f"Порог сходства для детектора незнакомых жалоб: {similarity_threshold:.3f}\n")
    print(classification_report(y_true, y_pred))

    # финальная модель на всех данных
    vectorizer = make_vectorizer(features)
    model = make_model().fit(vectorizer.fit_transform(df["text"]), df["label"])
    ood_vec = make_ood_vectorizer().fit(df["text"])
    ood_reference = {
        "vectorizer": ood_vec,
        "train_matrix": ood_vec.transform(df["text"]),
        "similarity_threshold": similarity_threshold,
    }

    joblib.dump(model, os.path.join(output_dir, "triage_model.joblib"))
    joblib.dump(vectorizer, os.path.join(output_dir, "vectorizer.joblib"))
    joblib.dump(ood_reference, os.path.join(output_dir, "ood_reference.joblib"))
    with open(os.path.join(output_dir, "training_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "features": features,
            "n_samples_raw": n_raw,
            "n_samples_deduplicated": len(df),
            "n_classes": int(df["label"].nunique()),
            "evaluation": f"StratifiedGroupKFold(5), groups = TF-IDF cosine >= {GROUP_SIMILARITY}",
            "cv_accuracy_mean": float(np.mean(fold_acc)),
            "cv_accuracy_std": float(np.std(fold_acc)),
            "cv_macro_f1_mean": float(np.mean(fold_f1)),
            "ood_similarity_threshold": similarity_threshold,
            "classes": sorted(df["label"].unique().tolist()),
            "classification_report": classification_report(y_true, y_pred, output_dict=True),
        }, f, ensure_ascii=False, indent=2)
    print(f"Сохранено в {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a symptom-to-disease classification model.")
    parser.add_argument("--data", required=True, help="Путь к Symptom2Disease.csv")
    parser.add_argument("--output", default="../model", help="Папка для модели и векторизатора")
    parser.add_argument("--features", default="word+char", choices=["word", "char", "word+char"])
    args = parser.parse_args()
    train(args.data, args.output, args.features)
