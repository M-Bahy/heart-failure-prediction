# Heart Failure Prediction

Binary classification of heart disease using multiple ML models with a focus on high recall and strong F1 performance.

- Active notebook: [notebook.ipynb](notebook.ipynb)
- Dataset: [heart.csv](heart.csv)
- Results snapshot: [all_results_sorted.csv](all_results_sorted.csv)

## Project Structure
- heart.csv — dataset (918 rows × 12 columns)
- notebook.ipynb — full workflow: EDA, preprocessing, modeling, evaluation
- all_results_sorted.csv — example model comparison results
- README.md — this file

## Overview
This project predicts the presence of heart disease (HeartDisease: 1/0) from clinical data. It benchmarks:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Classifier (SVC)
- XGBoost

Evaluation emphasizes Recall and F1 to minimize false negatives, which is critical in medical settings.

## Dataset
Source columns (lowercased with underscores in the workflow):
- age, sex, chestpaintype, restingbp, cholesterol, fastingbs, restingecg, maxhr, exerciseangina, oldpeak, st_slope, heartdisease

Targets:
- heartdisease (1: disease, 0: normal)

## Quickstart
1) Requirements
- Python 3.9+
- VS Code with Jupyter extension (recommended)

2) Install dependencies
- Option A (inside the notebook): the first cell installs all packages
- Option B (terminal):
  - pip install numpy pandas scikit-learn matplotlib seaborn statsmodels mlxtend xgboost scipy plotly lightgbm

3) Run
- Open [notebook.ipynb](notebook.ipynb) in VS Code
- Run all cells
- Model comparison table will be saved as all_results.csv

## What the Notebook Does
1) EDA
- Shapes, dtypes, unique counts, distributions, QQ plots, and outliers

2) Preprocessing
- Column cleanup: lowercase and underscores
- Imputation/corrections:
  - cholesterol: zeros → median of non-zero; Box-Cox; outliers capped via IQR median replacement; MinMax scaling
  - restingbp: zeros → median of non-zero; Box-Cox; outliers capped; MinMax scaling
  - oldpeak: negatives → median; reciprocal transform 1/(x+1); MinMax scaling
  - age, maxhr: Box-Cox; MinMax scaling
- Encoding:
  - sex, exerciseangina: label encoding (binary)
  - chestpaintype, restingecg, st_slope: one-hot encoding; one dummy column dropped per group to avoid multicollinearity

3) Feature Analysis
- Correlation heatmap and pairplots
- PCA (2 components) for visualization
- Clustering exploration: KMeans (elbow) and Hierarchical (Ward/Complete/Average/Single, silhouette)

4) Modeling and Evaluation
- Baselines and Grid Search per model (with/without PCA)
- Unified evaluation via:
  - [`evaluate_model`](notebook.ipynb): confusion matrices, classification reports, ROC-AUC, ROC curves
  - [`train_and_evaluate`](notebook.ipynb): trains an estimator and evaluates it
  - [`grid_search_and_evaluate`](notebook.ipynb): GridSearchCV + evaluation
  - [`train_sfs_and_evaluate`](notebook.ipynb): Sequential Forward Selection (mlxtend) + evaluation
  - [`draw_decision_boundary`](notebook.ipynb): decision boundary on PCA(2) space
- Metrics aggregated via [`expand_results`](notebook.ipynb) and exported to all_results.csv

## Results Summary
- Primary objective: maximize Test Recall and keep strong F1 and ROC-AUC
- Best performing choice in the notebook summary: XGBoost (Grid Search)
  - High Test Recall, top ROC-AUC, and best/near-best F1
  - Balanced train vs. test metrics (good generalization)
- A comparative results snapshot is provided in [all_results_sorted.csv](all_results_sorted.csv)

## Reproducibility
- Fixed random_state/seed = 42
- Stratified train/test split: 80/20
- Consistent preprocessing fitted on train, applied to test
- Deterministic grid searches (given seed and environment)