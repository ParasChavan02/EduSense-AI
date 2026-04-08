# 🎓 EduSense AI - Student Performance Predictor

A machine learning project that predicts student grades and provides personalized recommendations using classification, clustering, and feature analysis.

## ✨ Features

- **Data Processing**: Handles missing values and feature engineering
- **Scaler Comparison**: Tests StandardScaler vs MinMaxScaler
- **Feature Selection**: Uses RFE and correlation analysis
- **Multiple Models**: Logistic Regression, Decision Tree, Random Forest, XGBoost
- **Clustering**: K-Means with elbow method and silhouette analysis
- **Recommendations**: Personalized advice based on predictions
- **Web Dashboard**: Interactive Streamlit interface for visualization

## 📊 Dataset

Uses `student_performance.csv` with:
- `weekly_self_study_hours`
- `attendance_percentage`
- `class_participation`
- `total_score`
- `grade` (target)

## 🚀 Quick Start

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run desktop ML pipeline:**
```bash
python edusense_ai.py
```

**Run web dashboard:**
```bash
streamlit run app.py
```

## 📈 Output

- Dataset statistics and visualization
- Model performance comparison
- Confusion matrices
- Clustering analysis with metrics
- Sample student predictions & recommendations
- Interactive prediction tool

## 🛠️ Requirements

- Python 3.11+
- See `requirements.txt` for dependencies

## 📝 Notes

- The script automatically handles missing values
- Models are trained on 80% of data, tested on 20%
- Clustering uses silhouette score to find optimal K
- All visualizations are generated automatically

- elbow plot
- clustering scatter plot
- PCA plot

## Workflow Summary

1. Load dataset or create synthetic fallback data.
2. Build the required student feature set.
3. Handle missing values and encode the categorical column.
4. Compare scaling approaches.
5. Train and evaluate supervised models.
6. Perform clustering and PCA.
7. Generate human-readable recommendations.
