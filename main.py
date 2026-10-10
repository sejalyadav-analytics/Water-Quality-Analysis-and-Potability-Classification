"""
Water Quality Analysis and Potability Classification
Run: python main.py
Place the Kaggle CSV at data/water_potability.csv
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "water_potability.csv"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
REPORTS = OUTPUTS / "reports"

for folder in (OUTPUTS, FIGURES, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)

TARGET = "Potability"
RANDOM_STATE = 42


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Download the Kaggle Water Potability CSV and save it as:\n"
            "data/water_potability.csv"
        )
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' was not found. "
            f"Available columns: {list(df.columns)}"
        )
    return df


def save_basic_analysis(df: pd.DataFrame) -> None:
    # Basic dataset information
    info = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values_by_column": {
            k: int(v) for k, v in df.isna().sum().items()
        },
        "target_distribution": {
            str(k): int(v) for k, v in df[TARGET].value_counts(dropna=False).items()
        },
        "numeric_summary": df.describe(include="all").transpose()
            .fillna("").astype(str).to_dict(orient="index"),
    }
    with open(REPORTS / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    df.head(20).to_csv(REPORTS / "first_20_rows.csv", index=False)
    df.isna().sum().rename("missing_count").to_csv(REPORTS / "missing_values.csv")
    df.describe().transpose().to_csv(REPORTS / "descriptive_statistics.csv")

    # Target distribution
    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x=TARGET)
    plt.title("Distribution of Water Potability Labels")
    plt.xlabel("Potability label (dataset-provided)")
    plt.ylabel("Number of samples")
    plt.tight_layout()
    plt.savefig(FIGURES / "target_distribution.png", dpi=160)
    plt.close()

    # Missing values
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if not missing.empty:
        plt.figure(figsize=(9, 5))
        missing.plot(kind="bar")
        plt.title("Missing Values by Feature")
        plt.xlabel("Feature")
        plt.ylabel("Missing count")
        plt.tight_layout()
        plt.savefig(FIGURES / "missing_values.png", dpi=160)
        plt.close()

    # Histograms
    numeric_features = df.select_dtypes(include=np.number).columns.tolist()
    features = [c for c in numeric_features if c != TARGET]
    df[features].hist(figsize=(14, 10), bins=25)
    plt.suptitle("Feature Distributions", y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES / "feature_distributions.png", dpi=160)
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(11, 8))
    sns.heatmap(df[numeric_features].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES / "correlation_heatmap.png", dpi=160)
    plt.close()


def build_models():
    # Scaling is important for Logistic Regression; tree models do not require it.
    # The same pipeline structure is used for all models for simplicity.
    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced",
                                         random_state=RANDOM_STATE))
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(class_weight="balanced",
                                             random_state=RANDOM_STATE,
                                             min_samples_leaf=5))
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=300, class_weight="balanced",
                min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
            ))
        ]),
        "Extra Trees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(
                n_estimators=300, class_weight="balanced",
                min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
            ))
        ]),
    }

    # Add XGBoost when it is installed; the other models still run without it.
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=250, max_depth=4, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.9,
                objective="binary:logistic", eval_metric="logloss",
                random_state=RANDOM_STATE, n_jobs=-1
            ))
        ])
    except ImportError:
        print("XGBoost is not installed; skipping it. Install with: pip install xgboost")

    return models


def evaluate_models(X_train, X_test, y_train, y_test):
    models = build_models()
    results = []
    fitted_models = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # ROC-AUC is calculated when the model exposes probabilities.
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        except (AttributeError, ValueError):
            y_prob = None
            auc = np.nan

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": auc,
        })
        fitted_models[name] = model

        report = classification_report(y_test, y_pred, zero_division=0)
        safe_name = name.lower().replace(" ", "_")
        (REPORTS / f"classification_report_{safe_name}.txt").write_text(
            report, encoding="utf-8"
        )

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Label 0", "Label 1"]
        )
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(f"Confusion Matrix — {name}")
        fig.tight_layout()
        fig.savefig(FIGURES / f"confusion_matrix_{safe_name}.png", dpi=160)
        plt.close(fig)

        if y_prob is not None:
            fig, ax = plt.subplots(figsize=(7, 5))
            RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
            ax.set_title(f"ROC Curve — {name}")
            fig.tight_layout()
            fig.savefig(FIGURES / f"roc_curve_{safe_name}.png", dpi=160)
            plt.close(fig)

    results_df = pd.DataFrame(results).sort_values("F1-score", ascending=False)
    results_df.to_csv(REPORTS / "model_comparison.csv", index=False)

    plt.figure(figsize=(10, 6))
    plot_df = results_df.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"],
        var_name="Metric", value_name="Score"
    )
    sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric")
    plt.ylim(0, 1)
    plt.title("Model Comparison")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(FIGURES / "model_comparison.png", dpi=160)
    plt.close()

    return results_df, fitted_models


def save_feature_importance(best_model, X_test, y_test, feature_names):
    # Permutation importance works with the full fitted pipeline and is model-agnostic.
    print("Calculating permutation feature importance...")
    result = permutation_importance(
        best_model, X_test, y_test, n_repeats=8,
        random_state=RANDOM_STATE, scoring="f1", n_jobs=-1
    )
    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance_Mean": result.importances_mean,
        "Importance_Std": result.importances_std,
    }).sort_values("Importance_Mean", ascending=False)
    importance.to_csv(REPORTS / "permutation_feature_importance.csv", index=False)

    plt.figure(figsize=(9, 6))
    sns.barplot(data=importance, x="Importance_Mean", y="Feature")
    plt.title("Permutation Feature Importance (F1-score)")
    plt.xlabel("Mean decrease in F1-score when feature is shuffled")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIGURES / "permutation_feature_importance.png", dpi=160)
    plt.close()
    return importance


def try_shap(best_model, X_train, X_test, feature_names):
    """Optional SHAP explanation. If SHAP is unavailable, the main workflow continues."""
    try:
        import shap
    except ImportError:
        print("SHAP is not installed. Skipping SHAP plots. Install with: pip install shap")
        return

    try:
        fitted_estimator = best_model.named_steps["model"]
        imputer = best_model.named_steps["imputer"]
        X_sample = X_test.sample(min(150, len(X_test)), random_state=RANDOM_STATE)
        X_transformed = imputer.transform(X_sample)
        explainer = shap.TreeExplainer(fitted_estimator)
        shap_values = explainer.shap_values(X_transformed)

        # SHAP output shape differs across versions and classifiers.
        if isinstance(shap_values, list):
            values_to_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        elif getattr(shap_values, "ndim", 0) == 3:
            values_to_plot = shap_values[:, :, 1]
        else:
            values_to_plot = shap_values

        plt.figure()
        shap.summary_plot(
            values_to_plot, X_transformed, feature_names=feature_names,
            show=False, plot_type="bar"
        )
        plt.tight_layout()
        plt.savefig(FIGURES / "shap_summary_bar.png", dpi=160, bbox_inches="tight")
        plt.close("all")

        plt.figure()
        shap.summary_plot(
            values_to_plot, X_transformed, feature_names=feature_names,
            show=False
        )
        plt.tight_layout()
        plt.savefig(FIGURES / "shap_summary_beeswarm.png", dpi=160, bbox_inches="tight")
        plt.close("all")
        print("Saved SHAP summary plots.")
    except Exception as exc:
        print(f"SHAP explanation could not be generated for this model/version: {exc}")
        print("Permutation feature importance is available as the alternative.")


def main():
    print("=" * 65)
    print("WATER QUALITY ANALYSIS AND POTABILITY CLASSIFICATION")
    print("=" * 65)

    df = load_data(DATA_PATH)
    print(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

    # Drop exact duplicate records, if any.
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        print(f"Removing {duplicate_count} exact duplicate rows.")
        df = df.drop_duplicates().reset_index(drop=True)

    # Rows without a target label cannot be used for supervised learning.
    missing_target = int(df[TARGET].isna().sum())
    if missing_target:
        print(f"Dropping {missing_target} rows with missing target labels.")
        df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    save_basic_analysis(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    # Ensure features are numeric; unexpected text columns are reported.
    X = X.apply(pd.to_numeric, errors="coerce")
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")
    print("Missing feature values are imputed inside each training pipeline.")

    results_df, fitted_models = evaluate_models(X_train, X_test, y_train, y_test)
    print("\nModel comparison:")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Select based on F1 score; do not select only by accuracy when classes are imbalanced.
    best_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_name]
    print(f"\nBest model by test F1-score: {best_name}")

    (REPORTS / "best_model.txt").write_text(
        f"Best model by test F1-score: {best_name}\n\n"
        + results_df.iloc[0].to_string(),
        encoding="utf-8"
    )

    importance = save_feature_importance(
        best_model, X_test, y_test, feature_names
    )
    print("\nTop features by permutation importance:")
    print(importance.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # SHAP is attempted for tree-based models only; other models still get permutation importance.
    if best_name in ["Decision Tree", "Random Forest", "Extra Trees", "XGBoost"]:
        try_shap(best_model, X_train, X_test, feature_names)
    else:
        print("Best model is not tree-based; using permutation importance for explainability.")

    print("\nFinished.")
    print(f"Reports saved in: {REPORTS}")
    print(f"Figures saved in: {FIGURES}")
    print("\nReminder: dataset labels are not a substitute for certified water testing.")


if __name__ == "__main__":
    main()
