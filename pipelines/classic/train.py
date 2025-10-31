import os

import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

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

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)
preds = model.predict(X_test)
pred_proba = model.predict_proba(X_test)

acc = accuracy_score(y_test, preds)

print(f"\n{'=' * 50}")
print("MODEL EVALUATION RESULTS")
print(f"{'=' * 50}")
print(f"Accuracy: {acc:.3f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, pred_proba[:, 1]):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, preds, target_names=["No Win", "Win"]))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))
print("\nFeature Importances:")
for feature, importance in zip(X.columns, model.feature_importances_):
    print(f"  {feature}: {importance:.3f}")
print(f"{'=' * 50}\n")
print(
    f"Training complete! Model trained on {len(X_train)} samples, tested on {len(X_test)} samples."
)

results = X_test.copy()
results["actual"] = y_test.values
results["predicted"] = preds

engine = create_engine(
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:5432/{os.getenv('DB_NAME')}"
)
results.to_sql("f1_predictions", engine, if_exists="replace", index=False)
print("Predictions saved to database.")
