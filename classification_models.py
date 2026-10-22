"""AIML Practical 04: repeatable comparison of four binary classifiers.

Target encoding: 0 = benign; 1 = malignant. This is an educational
classification experiment, not a clinically validated diagnostic model.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # supports headless/script execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, confusion_matrix,
    f1_score, log_loss, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV, StratifiedKFold, cross_val_score, train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42
CV_FOLDS = 5
OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def load_data():
    """Load the bundled dataset and explicitly make malignant positive."""
    dataset = load_breast_cancer()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series((dataset.target == 0).astype(int), name="malignant")
    assert X.shape == (569, 30)
    assert y.value_counts().sort_index().to_dict() == {0: 357, 1: 212}
    assert X.notna().all().all()
    return X, y


def split_data(X, y):
    """Hold out a single stratified test split, never used during tuning."""
    return train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )


def model_specs():
    """Return separate estimators and small, explainable search spaces."""
    return {
        "SVM": (
            Pipeline([
                ("scale", StandardScaler()),
                ("model", SVC(probability=True, random_state=SEED)),
            ]),
            {
                "model__C": [0.1, 1, 10],
                "model__kernel": ["linear", "rbf"],
            },
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=SEED),
            {
                "criterion": ["gini", "entropy"],
                "max_depth": [3, 5, None],
                "min_samples_leaf": [1, 3, 5],
            },
        ),
        "KNN": (
            Pipeline([
                ("scale", StandardScaler()),
                ("model", KNeighborsClassifier()),
            ]),
            {
                "model__n_neighbors": [3, 5, 7, 9],
                "model__weights": ["uniform", "distance"],
            },
        ),
        "Naive Bayes": (
            GaussianNB(),
            {"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]},
        ),
    }


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Tune on training CV; evaluate each refitted model on held-out test."""
    cross_validation = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=SEED
    )
    rows, fitted = [], {}

    for name, (estimator, param_grid) in model_specs().items():
        baseline_scores = cross_val_score(
            estimator, X_train, y_train, cv=cross_validation,
            scoring="f1", n_jobs=-1,
        )
        search = GridSearchCV(
            estimator, param_grid=param_grid, scoring="f1",
            cv=cross_validation, n_jobs=-1, refit=True,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        fitted[name] = model

        y_pred = model.predict(X_test)
        positive_probability = model.predict_proba(X_test)[:, 1]
        tn, fp, fn, tp = confusion_matrix(
            y_test, y_pred, labels=[0, 1]
        ).ravel()
        rows.append({
            "Model": name,
            "Baseline_CV_F1": baseline_scores.mean(),
            "Tuned_CV_F1": search.best_score_,
            "CV_F1_SD": search.cv_results_["std_test_score"][search.best_index_],
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "Log_Loss": log_loss(y_test, positive_probability, labels=[0, 1]),
            "ROC_AUC": roc_auc_score(y_test, positive_probability),
            "Train_Accuracy": accuracy_score(y_train, model.predict(X_train)),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "Best_Params": str(search.best_params_),
        })
    results = pd.DataFrame(rows).sort_values(
        "Tuned_CV_F1", ascending=False
    ).reset_index(drop=True)
    return results, fitted


def save_plots(results, fitted, X_test, y_test, output_dir=OUTPUT_DIR):
    """Save clear, labeled figures as PNG and GitHub-friendly SVG."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})

    # Grouped performance metrics (note the truncated axis is labeled).
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    results.set_index("Model")[["Accuracy", "Precision", "Recall", "F1"]].plot.bar(
        ax=ax, width=0.76, rot=0,
        color=["#2864a5", "#35a49a", "#f0a646", "#8159bb"],
    )
    ax.set_ylim(0.75, 1.04)
    ax.set_ylabel("Score (0 to 1; displayed from 0.75)")
    ax.set_xlabel("")
    ax.set_title("Classifier performance on the held-out test set", weight="bold")
    ax.legend(loc="lower right", ncol=4)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"metrics_comparison.{suffix}", dpi=180)
    plt.close(fig)

    # Label the positive and negative class on every matrix.
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 9))
    for ax, (name, estimator) in zip(axes.flat, fitted.items()):
        matrix = confusion_matrix(
            y_test, estimator.predict(X_test), labels=[0, 1]
        )
        ConfusionMatrixDisplay(
            confusion_matrix=matrix, display_labels=["Benign", "Malignant"]
        ).plot(ax=ax, colorbar=False, cmap="Blues", values_format="d")
        ax.set_title(name, weight="bold")
    fig.suptitle("Confusion matrices (positive class: malignant)",
                 weight="bold", fontsize=13)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"confusion_matrices.{suffix}", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    colors = ["#2864a5", "#e39d35", "#35a49a", "#8159bb"]
    for (name, estimator), color in zip(fitted.items(), colors):
        RocCurveDisplay.from_estimator(
            estimator, X_test, y_test, ax=ax, name=name, color=color,
        )
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Random classifier")
    ax.set_title("ROC curves — malignant is the positive class", weight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"roc_curves.{suffix}", dpi=180)
    plt.close(fig)


def run_experiment(output_dir=OUTPUT_DIR, make_plots=True):
    """Run the full experiment and write all reproducible output artifacts."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    results, fitted = train_and_evaluate(X_train, X_test, y_train, y_test)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "classification_results.csv", index=False)

    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    summary = {
        "dataset": "Breast Cancer Wisconsin (Diagnostic) — sklearn bundled copy",
        "target_encoding": {"0": "benign", "1": "malignant"},
        "observations": len(y),
        "features": X.shape[1],
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "train_class_counts": {
            str(k): int(v) for k, v in y_train.value_counts().sort_index().items()
        },
        "test_class_counts": {
            str(k): int(v) for k, v in y_test.value_counts().sort_index().items()
        },
        "majority_baseline_test_accuracy": accuracy_score(
            y_test, dummy.predict(X_test)
        ),
        "majority_baseline_malignant_recall": recall_score(
            y_test, dummy.predict(X_test), zero_division=0
        ),
        "selection_metric": "mean stratified 5-fold training CV malignant F1",
        "recommended_model": results.iloc[0]["Model"],
    }
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    if make_plots:
        save_plots(results, fitted, X_test, y_test, output_dir)
    return results, summary, fitted


def main():
    results, summary, _ = run_experiment()
    pd.set_option("display.max_columns", None)
    print("Dataset:", summary["observations"], "rows;", summary["features"], "features")
    print("Class encoding: 0 = benign; 1 = malignant")
    print("Train / test:", summary["train_samples"], "/", summary["test_samples"])
    print("Test class counts:", summary["test_class_counts"])
    print("\nTest metrics / cross-validation results:\n")
    cols = ["Model", "Baseline_CV_F1", "Tuned_CV_F1", "CV_F1_SD", "Accuracy",
            "Precision", "Recall", "F1", "Log_Loss", "ROC_AUC",
            "Train_Accuracy", "TN", "FP", "FN", "TP"]
    print(results[cols].round(4).to_string(index=False))
    print("\nSelected hyperparameters:")
    for _, row in results.iterrows():
        print(f"{row['Model']}: {row['Best_Params']}")
    print("\nMajority-class baseline accuracy:",
          round(summary["majority_baseline_test_accuracy"], 4))
    print("Best by training cross-validation F1:", summary["recommended_model"])
    print("Outputs written to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()