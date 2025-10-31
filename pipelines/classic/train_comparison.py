import os
import warnings

import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)

df = pd.read_sql("SELECT * FROM f1_results_transformed", conn)
conn.close()

print(f"Loaded {len(df)} rows from database")
print(f"Columns: {df.columns.tolist()}")

df["win"] = (df["position"] == 1).astype(int)

df = df.sort_values(["driverId", "year", "round"])
df["races_so_far"] = df.groupby("driverId").cumcount()
df["wins_so_far"] = df.groupby("driverId")["win"].cumsum() - df["win"]
df["win_rate"] = (df["wins_so_far"] / (df["races_so_far"] + 1)).fillna(0)

df["points_so_far"] = df.groupby("driverId")["points"].cumsum() - df["points"]
df["avg_points"] = (df["points_so_far"] / (df["races_so_far"] + 1)).fillna(0)

df_model = df[df["races_so_far"] > 5].copy()

print(f"After filtering for drivers with history: {len(df_model)} rows")
print(f"Win distribution: {df_model['win'].value_counts().to_dict()}")

X = df_model[["year", "round", "win_rate", "avg_points"]]
y = df_model["win"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n{'=' * 70}")
print(f"TRAINING MODELS ON {len(X_train)} SAMPLES, TESTING ON {len(X_test)} SAMPLES")
print(f"{'=' * 70}\n")

print("Training Random Forest classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_pred_proba = rf_model.predict_proba(X_test)

rf_acc = accuracy_score(y_test, rf_preds)
rf_auc = roc_auc_score(y_test, rf_pred_proba[:, 1])

print(f"\n{'=' * 70}")
print("RANDOM FOREST RESULTS")
print(f"{'=' * 70}")
print(f"Accuracy: {rf_acc:.3f}")
print(f"ROC-AUC Score: {rf_auc:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, rf_preds, target_names=["No Win", "Win"]))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_preds))
print("\nFeature Importances:")
for feature, importance in zip(X.columns, rf_model.feature_importances_):
    print(f"  {feature}: {importance:.3f}")
print(f"{'=' * 70}\n")

print("Training XGBoost classifier with a compact grid search...")

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 7, 9],
    "learning_rate": [0.05, 0.1, 0.15],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
}

scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

xgb_base = XGBClassifier(
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=-1,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
)

total_combinations = 3 ** len(param_grid)
total_fits = total_combinations * 3

print("Grid Search configuration:")
print(f"  Total combinations: {total_combinations}")
print("  Cross-validation folds: 3")
print(f"  Total fits: {total_fits}")

grid_search = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    n_jobs=-1,
    verbose=2,
)

grid_search.fit(X_train, y_train)

print("\nBest parameters:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"Best cross-validation score: {grid_search.best_score_:.3f}")

xgb_model = grid_search.best_estimator_
xgb_preds = xgb_model.predict(X_test)
xgb_pred_proba = xgb_model.predict_proba(X_test)

xgb_acc = accuracy_score(y_test, xgb_preds)
xgb_auc = roc_auc_score(y_test, xgb_pred_proba[:, 1])

print(f"\n{'=' * 70}")
print("XGBOOST RESULTS")
print(f"{'=' * 70}")
print(f"Accuracy: {xgb_acc:.3f}")
print(f"ROC-AUC Score: {xgb_auc:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, xgb_preds, target_names=["No Win", "Win"]))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, xgb_preds))
print("\nFeature Importances:")
for feature, importance in zip(X.columns, xgb_model.feature_importances_):
    print(f"  {feature}: {importance:.3f}")
print(f"{'=' * 70}\n")

print(f"\n{'=' * 70}")
print("MODEL COMPARISON")
print(f"{'=' * 70}")
print(f"Random Forest Accuracy: {rf_acc:.3f} | ROC-AUC: {rf_auc:.3f}")
print(f"XGBoost Accuracy:       {xgb_acc:.3f} | ROC-AUC: {xgb_auc:.3f}")
print(
    f"Best model: {'Random Forest' if rf_acc > xgb_acc else 'XGBoost' if xgb_acc > rf_acc else 'Tie'}"
)
print(f"{'=' * 70}\n")

engine = create_engine(
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:5432/{os.getenv('DB_NAME')}"
)

rf_results = X_test.copy()
rf_results["actual"] = y_test.values
rf_results["predicted"] = rf_preds
rf_results["model"] = "RandomForest"
rf_results.to_sql("f1_predictions_rf", engine, if_exists="replace", index=False)
print("Random Forest predictions saved to f1_predictions_rf table.")

xgb_results = X_test.copy()
xgb_results["actual"] = y_test.values
xgb_results["predicted"] = xgb_preds
xgb_results["model"] = "XGBoost"
xgb_results.to_sql("f1_predictions_xgb", engine, if_exists="replace", index=False)
print("XGBoost predictions saved to f1_predictions_xgb table.")

comparison_df = pd.DataFrame(
    {
        "model": ["RandomForest", "XGBoost"],
        "accuracy": [rf_acc, xgb_acc],
        "roc_auc": [rf_auc, xgb_auc],
        "train_samples": [len(X_train), len(X_train)],
        "test_samples": [len(X_test), len(X_test)],
    }
)
comparison_df.to_sql("f1_model_comparison", engine, if_exists="replace", index=False)
print("Model comparison saved to f1_model_comparison table.")

print("Training complete! Both models trained and predictions saved to database.")
