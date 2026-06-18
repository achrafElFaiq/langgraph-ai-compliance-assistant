# datasets/train_classifier.py

import csv
import os
import joblib
import numpy as np
from collections import Counter

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

DATASET_FILE = "datasets/classifier/questions.csv"
OUTPUT_DIR = "datasets/classifier/model"
REGULATIONS = ["MiCA", "AI Act", "GDPR", "DORA"]


def load_dataset(path: str) -> tuple[list[str], list[list[str]]]:
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
    return questions, labels


def print_dataset_stats(labels: list[list[str]]):
    print("\nDataset stats:")
    flat = [reg for regs in labels for reg in regs]
    for reg, count in sorted(Counter(flat).items()):
        print(f"  {reg}: {count} questions")
    multi = sum(1 for l in labels if len(l) > 1)
    print(f"  Multi-label: {multi} questions ({multi/len(labels)*100:.1f}%)")
    print(f"  Total: {len(labels)} questions")


def evaluate(clf, X_test, y_test, mlb: MultiLabelBinarizer, name: str):
    y_pred = clf.predict(X_test)
    print(f"\n--- {name} ---")
    print(classification_report(
        y_test, y_pred,
        target_names=mlb.classes_,
        zero_division=0
    ))
    f1_micro = f1_score(y_test, y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    print(f"  F1 micro: {f1_micro:.3f}")
    print(f"  F1 macro: {f1_macro:.3f}")
    return f1_micro


def grid_search(base_model, param_grid: dict, X_train, y_train) -> MultiOutputClassifier:
    clf = MultiOutputClassifier(base_model, n_jobs=-1)

    # Wrap param keys with estimator__ prefix for MultiOutputClassifier
    wrapped_grid = {
        f"estimator__{k}": v for k, v in param_grid.items()
    }

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
    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV F1 micro: {search.best_score_:.3f}")
    return search.best_estimator_


def main():
    print(f"Loading dataset from {DATASET_FILE}...")
    questions, labels = load_dataset(DATASET_FILE)
    print_dataset_stats(labels)

    # Encode labels
    mlb = MultiLabelBinarizer(classes=REGULATIONS)
    y = mlb.fit_transform(labels)

    # TF-IDF features
    print("\nFitting TF-IDF vectorizer...")
    from sklearn.pipeline import FeatureUnion

    vectorizer = FeatureUnion([
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
    X = vectorizer.fit_transform(questions)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # Random Forest grid search
    print("\nGrid search — Random Forest...")
    rf_params = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 20, 40],
        "min_samples_leaf": [1, 2],
        "class_weight": ["balanced"],
    }
    best_rf = grid_search(
        RandomForestClassifier(random_state=42),
        rf_params, X_train, y_train
    )
    rf_f1 = evaluate(best_rf, X_test, y_test, mlb, "Random Forest (tuned)")

    # Gradient Boosting grid search
    print("\nGrid search — Gradient Boosting...")
    gb_params = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
    }
    best_gb = grid_search(
        GradientBoostingClassifier(random_state=42),
        gb_params, X_train, y_train
    )
    gb_f1 = evaluate(best_gb, X_test, y_test, mlb, "Gradient Boosting (tuned)")

    # Pick best — prefer RF on tie (better recall for compliance)
    best_name = "GradientBoosting" if gb_f1 > rf_f1 else "RandomForest"
    best_clf = best_gb if gb_f1 > rf_f1 else best_rf
    print(f"\nBest model: {best_name} (F1 micro: {max(rf_f1, gb_f1):.3f})")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    joblib.dump(best_clf, f"{OUTPUT_DIR}/classifier.joblib")
    joblib.dump(vectorizer, f"{OUTPUT_DIR}/vectorizer.joblib")
    joblib.dump(mlb, f"{OUTPUT_DIR}/mlb.joblib")
    print(f"\nSaved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()