"""
EduSense AI: Streamlit Web Application
Student Performance Predictor & Recommender
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score, accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")

# ==================== Configuration ====================
st.set_page_config(page_title="EduSense AI", layout="wide", initial_sidebar_state="expanded")
st.title("🎓 EduSense AI - Student Performance Predictor")
st.markdown("Predict student performance and get personalized recommendations using machine learning")

DEFAULT_DATASET_PATH = Path(__file__).parent / "student_performance.csv"
DEFAULT_SAMPLE_SIZE = 20000
RANDOM_STATE = 42

# ==================== Helper Functions ====================
@st.cache_data
def load_dataset(dataset_path: Optional[Path] = None, sample_size: Optional[int] = None):
    """Load and prepare dataset"""
    if sample_size is None:
        sample_size = DEFAULT_SAMPLE_SIZE
    
    try:
        df = pd.read_csv(dataset_path or DEFAULT_DATASET_PATH)
        df = df.sample(n=min(sample_size, len(df)), random_state=RANDOM_STATE)
        return df
    except FileNotFoundError:
        st.error(f"Dataset not found at {dataset_path or DEFAULT_DATASET_PATH}")
        return None


def get_feature_lists() -> Tuple[List[str], List[str], str]:
    """Define feature groups"""
    numeric_features = ["weekly_self_study_hours", "attendance_percentage", "class_participation", "total_score"]
    categorical_features = []
    target_column = "grade"
    return numeric_features, categorical_features, target_column


def handle_missing_values(df: pd.DataFrame, numeric_features: List[str]) -> pd.DataFrame:
    """Handle missing values"""
    df_processed = df.copy()
    
    # Drop student_id if present
    if "student_id" in df_processed.columns:
        df_processed = df_processed.drop("student_id", axis=1)
    
    # Fill numeric features with median
    for feature in numeric_features:
        if feature in df_processed.columns:
            df_processed[feature] = df_processed[feature].fillna(df_processed[feature].median())
    
    return df_processed


def create_model(model_name: str):
    """Create ML model"""
    try:
        from xgboost import XGBClassifier
        xgboost_available = True
    except ImportError:
        xgboost_available = False
    
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
    elif model_name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)
    elif model_name == "Random Forest":
        return RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1)
    elif model_name == "XGBoost":
        if not xgboost_available:
            return RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1)
        return XGBClassifier(n_estimators=200, max_depth=5, random_state=RANDOM_STATE, eval_metric="mlogloss")


def create_preprocessor(scale_type: str = "standard"):
    """Create preprocessing pipeline"""
    numeric_features = ["weekly_self_study_hours", "attendance_percentage", "class_participation", "total_score"]
    categorical_features = []
    
    scaler = StandardScaler() if scale_type == "standard" else MinMaxScaler()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", scaler, numeric_features),
        ]
    )
    return preprocessor


def generate_recommendation(row: pd.Series) -> str:
    """Generate personalized recommendation"""
    advice = []
    
    if row["weekly_self_study_hours"] < 10:
        advice.append("Increase study hours with a fixed weekly timetable.")
    if row["attendance_percentage"] < 85:
        advice.append("Improve attendance and avoid missing important classes.")
    if row["class_participation"] < 5:
        advice.append("Increase class participation and engage more actively in discussions.")
    
    prediction = row["predicted_performance"]
    if prediction == "A":
        advice.append("Maintain your current habits and aim for advanced practice questions.")
    elif prediction == "B":
        advice.append("You are close to top performance; focus on consistency and mock tests.")
    elif prediction == "C":
        advice.append("Meet a mentor or teacher weekly to track progress and close learning gaps.")
    else:
        advice.append("Seek immediate academic support to improve performance.")
    
    if not advice:
        advice.append("Keep following your current routine and review progress every week.")
    
    return " ".join(advice)


# ==================== Main Application ====================
def main():
    # Sidebar for navigation
    st.sidebar.header("📊 Navigation")
    page = st.sidebar.radio("Select a page:", 
                           ["Dataset Overview", "Model Performance", "Clustering Analysis", "Predictions"])
    
    # Load data
    with st.spinner("Loading dataset..."):
        df = load_dataset(DEFAULT_DATASET_PATH, DEFAULT_SAMPLE_SIZE)
    
    if df is None:
        st.stop()
    
    # Get features
    numeric_features, categorical_features, target_column = get_feature_lists()
    
    # Validate columns exist
    required_cols = numeric_features + categorical_features + [target_column]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"❌ Missing columns: {', '.join(missing_cols)}")
        st.info(f"Available columns: {', '.join(df.columns)}")
        st.stop()
    
    # Handle missing values
    df = handle_missing_values(df, numeric_features)
    
    if page == "Dataset Overview":
        st.header("📋 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Features", len(numeric_features) + len(categorical_features))
        with col3:
            st.metric("Target Classes", df[target_column].nunique())
        with col4:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        st.subheader("Sample Data")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.subheader("📊 Target Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        df[target_column].value_counts().sort_index().plot(kind="bar", ax=ax, color=["#FF6B6B", "#45B7D1", "#4ECDC4"])
        ax.set_title("Performance Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Performance Grade")
        ax.set_ylabel("Number of Students")
        plt.xticks(rotation=0)
        st.pyplot(fig)
        
        st.subheader("🔗 Correlation Matrix")
        plot_df = df.copy()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(plot_df[numeric_features].corr(numeric_only=True), annot=True, fmt=".2f", cmap="YlGnBu", ax=ax)
        ax.set_title("Correlation Matrix", fontsize=14, fontweight="bold")
        st.pyplot(fig)
        
        st.subheader("📈 Descriptive Statistics")
        st.dataframe(df[numeric_features].describe(), use_container_width=True)
    
    elif page == "Model Performance":
        st.header("🤖 Model Performance Analysis")
        
        # Prepare data
        X = df[numeric_features + categorical_features]
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        
        # Encode labels
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test)
        
        # Train models
        models_to_train = ["Logistic Regression", "Decision Tree", "Random Forest"]
        results = []
        
        with st.spinner("Training models..."):
            for model_name in models_to_train:
                preprocessor = create_preprocessor(scale_type="minmax")
                pipeline = Pipeline([
                    ("preprocessor", preprocessor),
                    ("model", create_model(model_name))
                ])
                
                pipeline.fit(X_train, y_train_encoded)
                y_pred_encoded = pipeline.predict(X_test)
                y_pred = le.inverse_transform(y_pred_encoded)
                
                results.append({
                    "Model": model_name,
                    "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                    "Precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
                    "Recall": round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
                    "F1-Score": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
                })
        
        results_df = pd.DataFrame(results).sort_values("F1-Score", ascending=False)
        
        st.subheader("Model Comparison")
        st.dataframe(results_df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            results_df.set_index("Model")["Accuracy"].plot(kind="bar", ax=ax, color="#4ECDC4")
            ax.set_title("Model Accuracy", fontsize=12, fontweight="bold")
            ax.set_ylabel("Accuracy")
            ax.set_xlabel("")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            results_df.set_index("Model")["F1-Score"].plot(kind="bar", ax=ax, color="#FF6B6B")
            ax.set_title("Model F1-Score", fontsize=12, fontweight="bold")
            ax.set_ylabel("F1-Score")
            ax.set_xlabel("")
            plt.xticks(rotation=45)
            st.pyplot(fig)
    
    elif page == "Clustering Analysis":
        st.header("📊 Clustering Analysis")
        
        # Prepare data for clustering
        X = df[numeric_features + categorical_features].copy()
        preprocessor = create_preprocessor(scale_type="minmax")
        X_processed = preprocessor.fit_transform(X)
        
        # Find optimal K
        with st.spinner("Finding optimal clusters..."):
            k_values = range(2, 9)
            silhouette_scores = []
            inertias = []
            
            for k in k_values:
                kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
                labels = kmeans.fit_predict(X_processed)
                silhouette_scores.append(silhouette_score(X_processed, labels))
                inertias.append(kmeans.inertia_)
            
            optimal_k = k_values[np.argmax(silhouette_scores)]
        
        # Train final model
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=10)
        df["cluster"] = kmeans_final.fit_predict(X_processed)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Optimal K", optimal_k)
        with col2:
            st.metric("Silhouette Score", f"{silhouette_scores[optimal_k - 2]:.4f}")
        with col3:
            db_score = davies_bouldin_score(X_processed, kmeans_final.labels_)
            st.metric("Davies-Bouldin Index", f"{db_score:.4f}")
        
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(list(k_values), inertias, "bo-", linewidth=2, markersize=8)
            ax.axvline(x=optimal_k, color="r", linestyle="--", label=f"Optimal K = {optimal_k}")
            ax.set_xlabel("Number of Clusters (K)")
            ax.set_ylabel("Inertia")
            ax.set_title("Elbow Method", fontsize=12, fontweight="bold")
            ax.legend()
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(list(k_values), silhouette_scores, "go-", linewidth=2, markersize=8)
            ax.axvline(x=optimal_k, color="r", linestyle="--", label=f"Optimal K = {optimal_k}")
            ax.set_xlabel("Number of Clusters (K)")
            ax.set_ylabel("Silhouette Score")
            ax.set_title("Silhouette Analysis", fontsize=12, fontweight="bold")
            ax.legend()
            st.pyplot(fig)
        
        # Cluster visualization
        st.subheader("K-Means Clustering (Study Hours vs Attendance)")
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(df["weekly_self_study_hours"], df["attendance_percentage"], c=df["cluster"], 
                           cmap="viridis", alpha=0.6, s=50)
        ax.set_xlabel("Weekly Study Hours")
        ax.set_ylabel("Attendance Percentage (%)")
        ax.set_title("Student Clusters")
        plt.colorbar(scatter, ax=ax, label="Cluster")
        st.pyplot(fig)
        
        # Cluster summary
        st.subheader("Cluster Summary")
        cluster_summary = df.groupby("cluster")[numeric_features].mean()
        st.dataframe(cluster_summary, use_container_width=True)
    
    elif page == "Predictions":
        st.header("🎯 Student Predictions & Recommendations")
        
        # Prepare and train model
        X = df[numeric_features + categorical_features]
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        
        preprocessor = create_preprocessor(scale_type="minmax")
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", create_model("Random Forest"))
        ])
        pipeline.fit(X_train, y_train_encoded)
        
        # Get sample predictions
        st.subheader("📋 Sample Student Predictions")
        sample_students = df.sample(n=min(5, len(df)), random_state=RANDOM_STATE).copy()
        sample_predictions = pipeline.predict(sample_students[numeric_features + categorical_features])
        sample_students["predicted_performance"] = le.inverse_transform(sample_predictions)
        sample_students["recommendation"] = sample_students.apply(generate_recommendation, axis=1)
        
        for idx, student in sample_students.iterrows():
            with st.expander(f"📚 Student {idx + 1} - Predicted Grade: **{student['predicted_performance']}**"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Weekly Study Hours", f"{student['weekly_self_study_hours']:.1f}h")
                    st.metric("Attendance", f"{student['attendance_percentage']:.1f}%")
                    st.metric("Class Participation", f"{student['class_participation']:.1f}")
                with col2:
                    st.metric("Total Score", f"{student['total_score']:.1f}")
                
                st.markdown("### 📝 Personalized Recommendation")
                st.info(student['recommendation'])
        
        # Interactive prediction
        st.markdown("---")
        st.subheader("🔮 Interactive Prediction Tool")
        st.markdown("Adjust the sliders to predict performance for different study scenarios")
        
        col1, col2 = st.columns(2)
        with col1:
            study_hours = st.slider("📚 Weekly Study Hours", 0.0, 30.0, 10.0, step=0.5)
            attendance = st.slider("📍 Attendance (%)", 0.0, 100.0, 85.0, step=1.0)
        with col2:
            participation = st.slider("💬 Class Participation Score", 0.0, 10.0, 5.0, step=0.5)
            total_score = st.slider("📊 Total Score", 0.0, 100.0, 70.0, step=1.0)
        
        if st.button("🎯 Get Prediction", key="predict_btn", use_container_width=True):
            user_data = pd.DataFrame({
                "weekly_self_study_hours": [study_hours],
                "attendance_percentage": [attendance],
                "class_participation": [participation],
                "total_score": [total_score]
            })
            
            pred_encoded = pipeline.predict(user_data)
            prediction = le.inverse_transform(pred_encoded)[0]
            
            # Display prediction
            grade_colors = {"A": "🟢", "B": "🟡", "C": "🟠", "D": "🔴"}
            grade_emoji = grade_colors.get(prediction, "❓")
            st.success(f"### {grade_emoji} Predicted Grade: **{prediction}**")
            
            # Generate recommendation
            rec_dict = {
                "weekly_self_study_hours": study_hours,
                "attendance_percentage": attendance,
                "class_participation": participation,
                "predicted_performance": prediction
            }
            recommendation = generate_recommendation(pd.Series(rec_dict))
            st.info(f"### 📝 Recommendation:\n{recommendation}")


if __name__ == "__main__":
    main()
