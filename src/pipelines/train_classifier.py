# datasets/train_classifier.py

import csv
import logging
import os
import time
import joblib
import numpy as np
from collections import Counter

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MultiLabelBinarizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

DATASET_FILE = "../../datasets/classifier/questions.csv"
OUTPUT_DIR = "../../datasets/classifier/model"
REGULATIONS = ["MiCA", "AI Act", "GDPR", "DORA"]


def load_dataset(path: str) -> tuple[list[str], list[list[str]]]:
    log.info("Loading dataset from %s", path)
    questions, labels = [], []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row["question"].strip()
            regs = [r.strip() for r in row["regulation"].split(",")]
            regs = [r for r in regs if r in REGULATIONS]
            if q and regs:
                questions.append(q)
                labels.append(regs)
    log.info("Loaded %d questions", len(questions))
    return questions, labels


def print_dataset_stats(labels: list[list[str]]):
    log.info("--- Dataset stats ---")
    flat = [reg for regs in labels for reg in regs]
    for reg, count in sorted(Counter(flat).items()):
        log.info("  %-10s %d questions", reg, count)
    multi = sum(1 for l in labels if len(l) > 1)
    log.info("  Multi-label : %d (%.1f%%)", multi, multi / len(labels) * 100)
    log.info("  Total       : %d", len(labels))


def build_vectorizer() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=20000,
            sublinear_tf=True,
            min_df=2,
            analyzer="word",
        )),
        ("char", TfidfVectorizer(
            ngram_range=(3, 5),
            max_features=10000,
            sublinear_tf=True,
            min_df=3,
            analyzer="char_wb",
        )),
    ])


def evaluate(clf, X_test, y_test, mlb: MultiLabelBinarizer, name: str) -> float:
    log.info("Evaluating %s...", name)
    y_pred = clf.predict(X_test)
    report = classification_report(
        y_test, y_pred,
        target_names=mlb.classes_,
        zero_division=0
    )
    print(f"\n--- {name} ---")
    print(report)
    f1_micro = f1_score(y_test, y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    log.info("%s  →  F1 micro=%.3f  F1 macro=%.3f", name, f1_micro, f1_macro)
    return f1_micro


def tune_thresholds(clf, X_test, y_test, mlb: MultiLabelBinarizer) -> list[float]:
    log.info("Tuning decision thresholds per regulation...")
    probas = clf.predict_proba(X_test)
    best_thresholds = []

    for i, reg in enumerate(mlb.classes_):
        proba_pos = probas[i][:, 1]
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.1, 0.7, 0.05):
            preds = (proba_pos >= t).astype(int)
            f1 = f1_score(y_test[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = round(float(t), 2)
        best_thresholds.append(best_t)
        log.info("  %-10s threshold=%.2f  F1=%.3f", reg, best_t, best_f1)

    return best_thresholds


def run_grid_search(
    name: str,
    base_model,
    param_grid: dict,
    X_train, y_train
) -> MultiOutputClassifier:
    log.info("Starting GridSearchCV — %s", name)
    log.info("  Param grid: %s", param_grid)
    t0 = time.time()

    clf = MultiOutputClassifier(base_model, n_jobs=-1)
    wrapped_grid = {f"estimator__{k}": v for k, v in param_grid.items()}
    n_candidates = 1
    for v in param_grid.values():
        n_candidates *= len(v)
    log.info("  %d candidates × 3 folds = %d fits", n_candidates, n_candidates * 3)

    scorer = make_scorer(f1_score, average="micro", zero_division=0)
    search = GridSearchCV(
        clf,
        wrapped_grid,
        scoring=scorer,
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    search.fit(X_train, y_train)

    elapsed = time.time() - t0
    log.info("  Done in %.1fs", elapsed)
    log.info("  Best params : %s", search.best_params_)
    log.info("  Best CV F1  : %.3f", search.best_score_)
    return search.best_estimator_


def main():
    t_start = time.time()

    # Load
    questions, labels = load_dataset(DATASET_FILE)
    print_dataset_stats(labels)

    # Encode labels
    log.info("Encoding labels with MultiLabelBinarizer...")
    mlb = MultiLabelBinarizer(classes=REGULATIONS)
    y = mlb.fit_transform(labels)
    log.info("  Classes: %s", list(mlb.classes_))

    # Vectorize
    log.info("Fitting TF-IDF vectorizer (word ngrams + char ngrams)...")
    t0 = time.time()
    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(questions)
    log.info("  Done in %.1fs — feature matrix: %s", time.time() - t0, X.shape)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    log.info("Train: %d | Test: %d", X_train.shape[0], X_test.shape[0])

    # Random Forest
    log.info("=" * 50)
    log.info("MODEL 1: Random Forest")
    log.info("=" * 50)
    rf_params = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 20, 40],
        "min_samples_leaf": [1, 2],
        "class_weight": ["balanced"],
    }
    best_rf = run_grid_search("RandomForest", RandomForestClassifier(random_state=42), rf_params, X_train, y_train)
    rf_f1 = evaluate(best_rf, X_test, y_test, mlb, "Random Forest (tuned)")

    # Gradient Boosting
    log.info("=" * 50)
    log.info("MODEL 2: Gradient Boosting")
    log.info("=" * 50)
    gb_params = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
    }
    best_gb = run_grid_search("GradientBoosting", GradientBoostingClassifier(random_state=42), gb_params, X_train, y_train)
    gb_f1 = evaluate(best_gb, X_test, y_test, mlb, "Gradient Boosting (tuned)")

    # Compare
    log.info("=" * 50)
    log.info("RESULTS SUMMARY")
    log.info("=" * 50)
    log.info("  RandomForest    F1 micro=%.3f", rf_f1)
    log.info("  GradientBoosting F1 micro=%.3f", gb_f1)

    best_name = "GradientBoosting" if gb_f1 > rf_f1 else "RandomForest"
    best_clf = best_gb if gb_f1 > rf_f1 else best_rf
    log.info("  Winner: %s", best_name)

    # Threshold tuning
    log.info("=" * 50)
    thresholds = tune_thresholds(best_clf, X_test, y_test, mlb)

    # Save
    log.info("=" * 50)
    log.info("Saving model artifacts to %s", OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump(best_clf,   f"{OUTPUT_DIR}/classifier.joblib")
    joblib.dump(vectorizer, f"{OUTPUT_DIR}/vectorizer.joblib")
    joblib.dump(mlb,        f"{OUTPUT_DIR}/mlb.joblib")
    joblib.dump(thresholds, f"{OUTPUT_DIR}/thresholds.joblib")
    log.info("  classifier.joblib")
    log.info("  vectorizer.joblib")
    log.info("  mlb.joblib")
    log.info("  thresholds.joblib")

    log.info("Total training time: %.1fs", time.time() - t_start)


if __name__ == "__main__":
    main()