"""
EduSense AI: Student Performance Predictor & Recommender
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_DATASET_PATH = Path(r"C:\Users\paras\Downloads\student_performance.csv")
DEFAULT_SAMPLE_SIZE = 20000
RANDOM_STATE = 42


def check_required_packages() -> Dict[str, object]:
    missing = []
    modules: Dict[str, object] = {}
    required = {
        "matplotlib.pyplot": "matplotlib",
        "seaborn": "seaborn",
        "sklearn": "scikit-learn",
    }
    for import_name, package_name in required.items():
        try:
            modules[import_name] = __import__(import_name, fromlist=["*"])
        except ModuleNotFoundError:
            missing.append(package_name)
    if missing:
        print("\nMissing required packages:", ", ".join(sorted(set(missing))))
        print("Recommended Python version: 3.11 or 3.12")
        print("Install with: pip install pandas numpy matplotlib seaborn scikit-learn xgboost streamlit")
        sys.exit(1)
    return modules


try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ModuleNotFoundError:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False


def create_synthetic_dataset(rows: int = 2000, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    study_hours = np.clip(rng.normal(15, 5, rows), 1, 40)
    attendance = np.clip(rng.normal(84, 10, rows), 45, 100)
    sleep_hours = np.clip(rng.normal(7, 1.1, rows), 4, 10)
    previous_scores = np.clip(
        35 + (study_hours * 1.8) + (attendance * 0.35) + rng.normal(0, 9, rows), 35, 100
    )
    extracurricular_activity = rng.choice(["Yes", "No"], size=rows, p=[0.58, 0.42])
    score_signal = (
        0.36 * previous_scores
        + 0.28 * attendance
        + 1.10 * study_hours
        + 2.40 * sleep_hours
        + np.where(extracurricular_activity == "Yes", 2.4, -1.2)
        + rng.normal(0, 6, rows)
    )
    score_signal = np.clip(score_signal / 1.45, 35, 100)
    performance = pd.cut(
        score_signal, bins=[0, 69, 84, 100], labels=["C", "B", "A"], include_lowest=True
    ).astype(str)
    return pd.DataFrame(
        {
            "study_hours": np.round(study_hours, 1),
            "attendance": np.round(attendance, 1),
            "sleep_hours": np.round(sleep_hours, 1),
            "previous_scores": np.round(previous_scores, 1),
            "extracurricular_activity": extracurricular_activity,
            "performance": performance,
        }
    )


def load_and_prepare_dataset(
    dataset_path: Optional[Path],
    sample_size: Optional[int] = DEFAULT_SAMPLE_SIZE,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    if dataset_path and dataset_path.exists():
        raw_df = pd.read_csv(dataset_path)
        print(f"\nLoaded dataset from: {dataset_path}")
        print(f"Original shape: {raw_df.shape}")
        if sample_size is not None and sample_size > 0 and len(raw_df) > sample_size:
            raw_df = raw_df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
            print(f"Sampled {sample_size:,} rows for faster training and visualization.")
        study_hours = raw_df["weekly_self_study_hours"].astype(float)
        attendance = raw_df["attendance_percentage"].astype(float)
        class_participation = raw_df["class_participation"].astype(float)
        total_score = raw_df["total_score"].astype(float)
        original_grade = raw_df["grade"].astype(str).str.upper().str.strip()
        sleep_hours = np.clip(
            7.8
            - 0.05 * np.maximum(study_hours - 18, 0)
            + 0.12 * (class_participation - class_participation.mean())
            + rng.normal(0, 0.6, len(raw_df)),
            4.0,
            9.5,
        )
        previous_scores = np.clip(
            total_score * 0.83 + attendance * 0.08 + study_hours * 0.25 + rng.normal(0, 6, len(raw_df)),
            35,
            100,
        )
        extracurricular_prob = np.clip(
            0.30 + 0.05 * class_participation + 0.002 * (attendance - 75), 0.20, 0.85
        )
        extracurricular_activity = np.where(
            rng.random(len(raw_df)) < extracurricular_prob, "Yes", "No"
        )
        performance = original_grade.replace({"D": "C", "F": "C"})
        df = pd.DataFrame(
            {
                "study_hours": study_hours.round(1),
                "attendance": attendance.round(1),
                "sleep_hours": np.round(sleep_hours, 1),
                "previous_scores": np.round(previous_scores, 1),
                "extracurricular_activity": extracurricular_activity,
                "performance": performance,
            }
        )
    else:
        print("\nDataset not found. Creating a realistic synthetic dataset instead.")
        df = create_synthetic_dataset(rows=3000, random_state=random_state)
    for column in ["study_hours", "attendance", "sleep_hours", "previous_scores"]:
        missing_mask = rng.random(len(df)) < 0.02
        df.loc[missing_mask, column] = np.nan
    cat_mask = rng.random(len(df)) < 0.015
    df.loc[cat_mask, "extracurricular_activity"] = np.nan
    print("\nPrepared EduSense dataset preview:")
    print(df.head().to_string(index=False))
    print("\nMissing values per column:")
    print(df.isnull().sum().to_string())
    print("\nTarget distribution:")
    print(df["performance"].value_counts().to_string())
    return df


def get_feature_lists() -> Tuple[List[str], List[str], str]:
    return (
        ["study_hours", "attendance", "sleep_hours", "previous_scores"],
        ["extracurricular_activity"],
        "performance",
    )


def create_preprocessor(scale_type: str = "standard"):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

    numeric_features, categorical_features, _ = get_feature_lists()
    scaler = StandardScaler() if scale_type == "standard" else MinMaxScaler()
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", scaler)]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def create_preprocessor_without_scaling():
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    numeric_features, categorical_features, _ = get_feature_lists()
    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )


def create_model(model_name: str):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    if model_name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)
    if model_name == "Random Forest":
        return RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1)
    if model_name == "XGBoost":
        if not XGBOOST_AVAILABLE:
            raise RuntimeError("XGBoost is not installed in the current environment.")
        return XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"Unsupported model name: {model_name}")


def plot_correlation_matrix(df: pd.DataFrame, plt_module, sns_module) -> None:
    plot_df = df.copy()
    plot_df["extracurricular_activity"] = plot_df["extracurricular_activity"].map({"Yes": 1, "No": 0})
    plot_df["performance"] = plot_df["performance"].map({"C": 0, "B": 1, "A": 2})
    corr = plot_df.corr(numeric_only=True)
    plt_module.figure(figsize=(8, 6))
    sns_module.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f")
    plt_module.title("Correlation Matrix")
    plt_module.tight_layout()


def run_rfe_feature_selection(X_train: pd.DataFrame, y_train: pd.Series, feature_names: List[str]) -> pd.DataFrame:
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression

    estimator = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    selector = RFE(estimator=estimator, n_features_to_select=min(4, len(feature_names)))
    selector.fit(X_train, y_train)
    return pd.DataFrame(
        {"feature": feature_names, "selected": selector.support_, "ranking": selector.ranking_}
    ).sort_values(["selected", "ranking"], ascending=[False, True])


@dataclass
class ModelRunResult:
    model_name: str
    scaler: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray
    y_pred: np.ndarray


def evaluate_model(y_true, y_pred) -> Dict[str, object]:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.metrics import precision_score, recall_score

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def compare_scalers(X_train, X_test, y_train, y_test) -> Tuple[pd.DataFrame, str]:
    from sklearn.pipeline import Pipeline

    scaler_results = []
    best_scaler = "standard"
    best_accuracy = -1.0
    for scaler_name in ["standard", "minmax"]:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", create_preprocessor(scale_type=scaler_name)),
                ("model", create_model("Logistic Regression")),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)
        scaler_results.append(
            {
                "scaler": scaler_name.title(),
                "model": "Logistic Regression",
                "accuracy": round(metrics["accuracy"], 4),
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1_score": round(metrics["f1_score"], 4),
            }
        )
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_scaler = scaler_name
    return pd.DataFrame(scaler_results), best_scaler


def transform_for_rfe(X_train: pd.DataFrame, X_test: pd.DataFrame):
    preprocessor = create_preprocessor(scale_type="standard")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    feature_names = list(preprocessor.get_feature_names_out())
    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)
    return X_train_df, X_test_df, feature_names


def train_supervised_models(X_train, X_test, y_train, y_test, best_scaler: str):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder

    # Encode target labels to numeric values for all models
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    model_names = ["Logistic Regression", "Decision Tree", "Random Forest"]
    if XGBOOST_AVAILABLE:
        model_names.append("XGBoost")
    results: List[ModelRunResult] = []
    for model_name in model_names:
        if model_name == "Decision Tree":
            preprocessor = create_preprocessor_without_scaling()
            scaler_label = "No Scaling"
        else:
            preprocessor = create_preprocessor(scale_type=best_scaler)
            scaler_label = best_scaler.title()
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", create_model(model_name))])
        pipeline.fit(X_train, y_train_encoded)
        y_pred_encoded = pipeline.predict(X_test)
        # Decode predictions back to original labels
        y_pred = le.inverse_transform(y_pred_encoded)
        metrics = evaluate_model(y_test, y_pred)
        results.append(
            ModelRunResult(
                model_name=model_name,
                scaler=scaler_label,
                accuracy=metrics["accuracy"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1_score=metrics["f1_score"],
                confusion_matrix=metrics["confusion_matrix"],
                y_pred=y_pred,
            )
        )
    results_df = pd.DataFrame(
        [
            {
                "Model": result.model_name,
                "Scaler": result.scaler,
                "Accuracy": round(result.accuracy, 4),
                "Precision": round(result.precision, 4),
                "Recall": round(result.recall, 4),
                "F1-Score": round(result.f1_score, 4),
            }
            for result in results
        ]
    ).sort_values(by="F1-Score", ascending=False)
    return results_df, results


def plot_confusion_matrices(results: List[ModelRunResult], labels: List[str], plt_module, sns_module):
    cols = 2
    rows = int(np.ceil(len(results) / cols))
    fig, axes = plt_module.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = np.array(axes).reshape(-1)
    for idx, result in enumerate(results):
        sns_module.heatmap(
            result.confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=axes[idx],
        )
        axes[idx].set_title(f"{result.model_name} Confusion Matrix")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("Actual")
    for idx in range(len(results), len(axes)):
        fig.delaxes(axes[idx])
    plt_module.tight_layout()


def find_optimal_k(X_cluster: np.ndarray) -> Tuple[List[int], List[float], List[float], List[float], int]:
    from sklearn.cluster import KMeans
    from sklearn.metrics import davies_bouldin_score, silhouette_score

    k_values = list(range(2, 9))
    inertias, silhouette_scores, davies_scores = [], [], []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(X_cluster)
        inertias.append(model.inertia_)
        silhouette_scores.append(silhouette_score(X_cluster, labels))
        davies_scores.append(davies_bouldin_score(X_cluster, labels))
    best_index = int(np.argmax(silhouette_scores))
    best_k = k_values[best_index]
    return k_values, inertias, silhouette_scores, davies_scores, best_k


def run_clustering(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import davies_bouldin_score, silhouette_score
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_features, categorical_features, _ = get_feature_lists()
    numeric_df = df[numeric_features].copy()
    categorical_df = df[categorical_features].copy()
    numeric_imputer = SimpleImputer(strategy="median")
    categorical_imputer = SimpleImputer(strategy="most_frequent")
    numeric_ready = numeric_imputer.fit_transform(numeric_df)
    categorical_ready = categorical_imputer.fit_transform(categorical_df)
    categorical_encoded = OneHotEncoder(handle_unknown="ignore").fit_transform(categorical_ready).toarray()
    combined = np.hstack([numeric_ready, categorical_encoded])
    scaled = StandardScaler().fit_transform(combined)
    k_values, inertias, silhouette_scores, davies_scores, best_k = find_optimal_k(scaled)
    final_model = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = final_model.fit_predict(scaled)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    components = pca.fit_transform(scaled)
    clustered_df = df.copy()
    clustered_df["cluster"] = cluster_labels
    clustered_df["pca_1"] = components[:, 0]
    clustered_df["pca_2"] = components[:, 1]
    cluster_metrics = {
        "k_values": k_values,
        "inertias": inertias,
        "silhouette_scores": silhouette_scores,
        "davies_scores": davies_scores,
        "best_k": best_k,
        "final_silhouette": silhouette_score(scaled, cluster_labels),
        "final_davies_bouldin": davies_bouldin_score(scaled, cluster_labels),
        "cluster_model": final_model,
    }
    return clustered_df, cluster_metrics


def plot_elbow(k_values, inertias, plt_module) -> None:
    plt_module.figure(figsize=(8, 5))
    plt_module.plot(k_values, inertias, marker="o")
    plt_module.title("Elbow Method for K-Means")
    plt_module.xlabel("Number of Clusters (K)")
    plt_module.ylabel("Inertia")
    plt_module.grid(True, alpha=0.3)
    plt_module.tight_layout()


def plot_cluster_scatter(clustered_df: pd.DataFrame, plt_module, sns_module) -> None:
    plt_module.figure(figsize=(8, 6))
    sns_module.scatterplot(
        data=clustered_df, x="study_hours", y="attendance", hue="cluster", palette="Set2", alpha=0.75
    )
    plt_module.title("K-Means Clusters (Study Hours vs Attendance)")
    plt_module.tight_layout()


def plot_pca_clusters(clustered_df: pd.DataFrame, plt_module, sns_module) -> None:
    plt_module.figure(figsize=(8, 6))
    sns_module.scatterplot(
        data=clustered_df,
        x="pca_1",
        y="pca_2",
        hue="cluster",
        style="performance",
        palette="tab10",
        alpha=0.75,
    )
    plt_module.title("PCA Projection of Student Clusters")
    plt_module.tight_layout()


def summarize_clusters(clustered_df: pd.DataFrame) -> pd.DataFrame:
    return clustered_df.groupby("cluster")[["study_hours", "attendance", "sleep_hours", "previous_scores"]].mean().round(2)


def generate_recommendation(student_row: pd.Series, cluster_summary: pd.DataFrame) -> str:
    cluster_id = int(student_row["cluster"])
    cluster_profile = cluster_summary.loc[cluster_id]
    advice: List[str] = []
    if student_row["study_hours"] < max(12, cluster_profile["study_hours"]):
        advice.append("Increase study hours with a fixed weekly timetable.")
    if student_row["attendance"] < max(80, cluster_profile["attendance"]):
        advice.append("Improve attendance and avoid missing important classes.")
    if student_row["sleep_hours"] < 6.5:
        advice.append("Sleep more consistently to improve focus and memory.")
    if student_row["previous_scores"] < max(70, cluster_profile["previous_scores"]):
        advice.append("Revise previous weak topics before learning new ones.")
    if student_row["extracurricular_activity"] == "No":
        advice.append("Join one extracurricular activity to strengthen engagement and balance.")
    prediction = student_row.get("predicted_performance", "")
    if prediction == "A":
        advice.append("Maintain your current habits and aim for advanced practice questions.")
    elif prediction == "B":
        advice.append("You are close to top performance; focus on consistency and mock tests.")
    else:
        advice.append("Meet a mentor or teacher weekly to track progress and close learning gaps.")
    if not advice:
        advice.append("Keep following your current routine and review progress every week.")
    return " ".join(advice)


def print_step_by_step_explanation() -> None:
    print("\n" + "=" * 80)
    print("STEP-BY-STEP EXPLANATION")
    print("=" * 80)
    print("1. Load the Kaggle CSV or generate a fallback dataset if the file is missing.")
    print("2. Transform the data into the required EduSense AI feature set.")
    print("3. Add a small amount of missing values so preprocessing can be demonstrated.")
    print("4. Handle missing values, encode categories, and compare two scalers.")
    print("5. Analyze feature relationships using a correlation heatmap and RFE.")
    print("6. Train multiple supervised learning models and compare their metrics.")
    print("7. Apply K-Means clustering, elbow method, PCA, and cluster evaluation scores.")
    print("8. Generate student-wise recommendations from predicted performance and cluster.")


def print_model_reports(results_df: pd.DataFrame, scaler_results_df: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("SCALER COMPARISON")
    print("=" * 80)
    print(scaler_results_df.to_string(index=False))
    print("\n" + "=" * 80)
    print("MODEL PERFORMANCE COMPARISON")
    print("=" * 80)
    print(results_df.to_string(index=False))


def prepare_example_students(df: pd.DataFrame) -> pd.DataFrame:
    return df.sample(n=min(5, len(df)), random_state=RANDOM_STATE).copy().reset_index(drop=True)


def run_main_workflow(dataset_path: Optional[Path], sample_size: Optional[int]) -> None:
    modules = check_required_packages()
    plt_module = modules["matplotlib.pyplot"]
    sns_module = modules["seaborn"]
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    sns_module.set_style("whitegrid")
    df = load_and_prepare_dataset(dataset_path=dataset_path, sample_size=sample_size)
    numeric_features, categorical_features, target_column = get_feature_lists()
    print_step_by_step_explanation()
    X = df[numeric_features + categorical_features]
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    scaler_results_df, best_scaler = compare_scalers(X_train, X_test, y_train, y_test)
    print(f"\nBest scaler selected for scaled models: {best_scaler.title()}")
    plot_correlation_matrix(df, plt_module, sns_module)
    X_train_rfe, X_test_rfe, feature_names = transform_for_rfe(X_train, X_test)
    rfe_results = run_rfe_feature_selection(X_train_rfe, y_train, feature_names)
    print("\n" + "=" * 80)
    print("RFE FEATURE SELECTION RESULTS")
    print("=" * 80)
    print(rfe_results.to_string(index=False))
    results_df, model_run_results = train_supervised_models(X_train, X_test, y_train, y_test, best_scaler)
    print_model_reports(results_df, scaler_results_df)
    labels = sorted(y.unique())
    plot_confusion_matrices(model_run_results, labels, plt_module, sns_module)
    best_model_name = results_df.iloc[0]["Model"]
    print(f"\nBest supervised model based on F1-score: {best_model_name}")
    
    # Encode labels for best_pipeline training
    from sklearn.preprocessing import LabelEncoder
    le_best = LabelEncoder()
    y_train_encoded_best = le_best.fit_transform(y_train)
    
    if best_model_name == "Decision Tree":
        best_pipeline = Pipeline(
            steps=[("preprocessor", create_preprocessor_without_scaling()), ("model", create_model(best_model_name))]
        )
    else:
        best_pipeline = Pipeline(
            steps=[("preprocessor", create_preprocessor(scale_type=best_scaler)), ("model", create_model(best_model_name))]
        )
    best_pipeline.fit(X_train, y_train_encoded_best)
    clustered_df, cluster_metrics = run_clustering(df)
    cluster_summary = summarize_clusters(clustered_df)
    print("\n" + "=" * 80)
    print("CLUSTERING EVALUATION")
    print("=" * 80)
    print(f"Optimal K from elbow/silhouette search: {cluster_metrics['best_k']}")
    print(f"Silhouette Score: {cluster_metrics['final_silhouette']:.4f}")
    print(f"Davies-Bouldin Index: {cluster_metrics['final_davies_bouldin']:.4f}")
    print("\nCluster summary:")
    print(cluster_summary.to_string())
    plot_elbow(cluster_metrics["k_values"], cluster_metrics["inertias"], plt_module)
    plot_cluster_scatter(clustered_df, plt_module, sns_module)
    plot_pca_clusters(clustered_df, plt_module, sns_module)
    examples = prepare_example_students(df)
    example_predictions = best_pipeline.predict(examples[numeric_features + categorical_features])
    # Decode predictions back to original labels
    example_predictions = le_best.inverse_transform(example_predictions)
    examples["predicted_performance"] = example_predictions
    clustered_lookup = clustered_df[["study_hours", "attendance", "sleep_hours", "previous_scores", "cluster"]]
    examples = examples.merge(
        clustered_lookup.drop_duplicates(subset=["study_hours", "attendance", "sleep_hours", "previous_scores"]),
        on=["study_hours", "attendance", "sleep_hours", "previous_scores"],
        how="left",
    )
    if examples["cluster"].isnull().any():
        fill_cluster = int(clustered_df["cluster"].mode().iloc[0])
        examples["cluster"] = examples["cluster"].fillna(fill_cluster)
    examples["recommendation"] = examples.apply(lambda row: generate_recommendation(row, cluster_summary), axis=1)
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS AND RECOMMENDATIONS")
    print("=" * 80)
    print(
        examples[
            [
                "study_hours",
                "attendance",
                "sleep_hours",
                "previous_scores",
                "extracurricular_activity",
                "predicted_performance",
                "cluster",
                "recommendation",
            ]
        ].to_string(index=False)
    )
    if not XGBOOST_AVAILABLE:
        print("\nNote: XGBoost is not installed in the current environment, so it was skipped.\nInstall it with: pip install xgboost")
    plt_module.show()


def run_streamlit_app(dataset_path: Optional[Path], sample_size: Optional[int]) -> None:
    try:
        import streamlit as st
    except ModuleNotFoundError:
        print("Streamlit is not installed. Install it with: pip install streamlit")
        sys.exit(1)
    modules = check_required_packages()
    _ = modules["matplotlib.pyplot"]
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    df = load_and_prepare_dataset(dataset_path=dataset_path, sample_size=sample_size)
    numeric_features, categorical_features, target_column = get_feature_lists()
    X = df[numeric_features + categorical_features]
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    scaler_results_df, best_scaler = compare_scalers(X_train, X_test, y_train, y_test)
    best_pipeline = Pipeline(
        steps=[("preprocessor", create_preprocessor(scale_type=best_scaler)), ("model", create_model("Random Forest"))]
    )
    best_pipeline.fit(X_train, y_train)
    clustered_df, _ = run_clustering(df)
    cluster_summary = summarize_clusters(clustered_df)
    st.title("EduSense AI: Student Performance Predictor & Recommender")
    st.write("Enter student details and get a predicted performance label with advice.")
    study_hours = st.slider("Study Hours", 0.0, 40.0, 15.0, 0.5)
    attendance = st.slider("Attendance (%)", 40.0, 100.0, 85.0, 0.5)
    sleep_hours = st.slider("Sleep Hours", 3.0, 10.0, 7.0, 0.5)
    previous_scores = st.slider("Previous Scores", 0.0, 100.0, 75.0, 0.5)
    extracurricular_activity = st.selectbox("Extracurricular Activity", ["Yes", "No"])
    input_df = pd.DataFrame(
        {
            "study_hours": [study_hours],
            "attendance": [attendance],
            "sleep_hours": [sleep_hours],
            "previous_scores": [previous_scores],
            "extracurricular_activity": [extracurricular_activity],
        }
    )
    if st.button("Predict Performance"):
        prediction = best_pipeline.predict(input_df)[0]
        closest_cluster = (
            (
                (clustered_df["study_hours"].fillna(clustered_df["study_hours"].median()) - study_hours) ** 2
                + (clustered_df["attendance"].fillna(clustered_df["attendance"].median()) - attendance) ** 2
            )
            .idxmin()
        )
        cluster_id = int(clustered_df.loc[closest_cluster, "cluster"])
        result_row = input_df.iloc[0].copy()
        result_row["predicted_performance"] = prediction
        result_row["cluster"] = cluster_id
        recommendation = generate_recommendation(result_row, cluster_summary)
        st.success(f"Predicted Performance: {prediction}")
        st.info(f"Assigned Cluster: {cluster_id}")
        st.write("Recommendation:")
        st.write(recommendation)
        st.write("Scaler comparison used during training:")
        st.dataframe(scaler_results_df)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EduSense AI: Student Performance Predictor & Recommender")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET_PATH), help="Path to the CSV dataset file.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of rows to sample from the dataset for faster execution. Use 0 for full data.",
    )
    parser.add_argument(
        "--streamlit",
        action="store_true",
        help="Run the optional Streamlit interface instead of the notebook-style workflow.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    dataset_path = Path(args.dataset) if args.dataset else None
    sample_size = None if args.sample_size == 0 else args.sample_size
    if args.streamlit:
        run_streamlit_app(dataset_path=dataset_path, sample_size=sample_size)
    else:
        run_main_workflow(dataset_path=dataset_path, sample_size=sample_size)


if __name__ == "__main__":
    main()
