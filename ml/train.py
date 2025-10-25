import os
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    dbname=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD')
)

df = pd.read_sql("SELECT * FROM f1_results_transformed", conn)
conn.close()

print(f"Loaded {len(df)} rows from database")
print(f"Columns: {df.columns.tolist()}")

# Create target variable: 1 if won (position == 1), 0 otherwise
df['win'] = (df['position'] == 1).astype(int)

# Feature engineering - use only features that would be available BEFORE the race result
# For a real model, you'd want: qualifying position, driver stats, constructor, weather, etc.
# For this demo, we'll use aggregated historical performance

# Calculate driver's historical win rate (excluding current race)
df = df.sort_values(['driverId', 'year', 'round'])
df['races_so_far'] = df.groupby('driverId').cumcount()
df['wins_so_far'] = df.groupby('driverId')['win'].cumsum() - df['win']  # Exclude current race
df['win_rate'] = df['wins_so_far'] / (df['races_so_far'] + 1)  # Avoid division by zero
df['win_rate'] = df['win_rate'].fillna(0)

# Add average points per race (historical, excluding current)
df['points_so_far'] = df.groupby('driverId')['points'].cumsum() - df['points']
df['avg_points'] = df['points_so_far'] / (df['races_so_far'] + 1)
df['avg_points'] = df['avg_points'].fillna(0)

# Use features that are predictive but not leaked from the outcome
# Only keep records where driver has some history
df_model = df[df['races_so_far'] > 5].copy()  # Need at least 5 races of history

print(f"After filtering for drivers with history: {len(df_model)} rows")
print(f"Win distribution: {df_model['win'].value_counts().to_dict()}")

X = df_model[['year', 'round', 'win_rate', 'avg_points']]
y = df_model['win']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
preds = model.predict(X_test)
pred_proba = model.predict_proba(X_test)

# Calculate metrics
acc = accuracy_score(y_test, preds)
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

print(f"\n{'='*50}")
print(f"MODEL EVALUATION RESULTS")
print(f"{'='*50}")
print(f"✅ Accuracy: {acc:.3f}")
print(f"✅ ROC-AUC Score: {roc_auc_score(y_test, pred_proba[:, 1]):.3f}")
print(f"\nClassification Report:")
print(classification_report(y_test, preds, target_names=['No Win', 'Win']))
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))
print(f"\nFeature Importances:")
for feature, importance in zip(X.columns, model.feature_importances_):
    print(f"  {feature}: {importance:.3f}")
print(f"{'='*50}\n")
print(f"✅ Training complete! Model trained on {len(X_train)} samples, tested on {len(X_test)} samples.")

results = X_test.copy()
results['actual'] = y_test.values
results['predicted'] = preds

engine = create_engine(f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:5432/{os.getenv('DB_NAME')}")
results.to_sql("f1_predictions", engine, if_exists='replace', index=False)
print("Predictions saved to database.")