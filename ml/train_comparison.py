import os
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Database connection
conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    dbname=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD')
)

df = pd.read_sql("SELECT * FROM f1_results", conn)
conn.close()

print(f"Loaded {len(df)} rows from database")
print(f"Columns: {df.columns.tolist()}")

# Create target variable: 1 if won (position == 1), 0 otherwise
df['win'] = (df['position'] == 1).astype(int)

# Feature engineering - use only features that would be available BEFORE the race result
df = df.sort_values(['driverId', 'year', 'round'])
df['races_so_far'] = df.groupby('driverId').cumcount()
df['wins_so_far'] = df.groupby('driverId')['win'].cumsum() - df['win']  # Exclude current race
df['win_rate'] = df['wins_so_far'] / (df['races_so_far'] + 1)  # Avoid division by zero
df['win_rate'] = df['win_rate'].fillna(0)

# Add average points per race (historical, excluding current)
df['points_so_far'] = df.groupby('driverId')['points'].cumsum() - df['points']
df['avg_points'] = df['points_so_far'] / (df['races_so_far'] + 1)
df['avg_points'] = df['avg_points'].fillna(0)

# Only keep records where driver has some history
df_model = df[df['races_so_far'] > 5].copy()

print(f"After filtering for drivers with history: {len(df_model)} rows")
print(f"Win distribution: {df_model['win'].value_counts().to_dict()}")

X = df_model[['year', 'round', 'win_rate', 'avg_points']]
y = df_model['win']

# Use the same train-test split for both models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n{'='*70}")
print(f"TRAINING MODELS ON {len(X_train)} SAMPLES, TESTING ON {len(X_test)} SAMPLES")
print(f"{'='*70}\n")

# ==================== Random Forest Model ====================
print("🌲 Training Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_pred_proba = rf_model.predict_proba(X_test)

rf_acc = accuracy_score(y_test, rf_preds)
rf_auc = roc_auc_score(y_test, rf_pred_proba[:, 1])

print(f"\n{'='*70}")
print(f"RANDOM FOREST RESULTS")
print(f"{'='*70}")
print(f"✅ Accuracy: {rf_acc:.3f}")
print(f"✅ ROC-AUC Score: {rf_auc:.3f}")
print(f"\nClassification Report:")
print(classification_report(y_test, rf_preds, target_names=['No Win', 'Win']))
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_preds))
print(f"\nFeature Importances:")
for feature, importance in zip(X.columns, rf_model.feature_importances_):
    print(f"  {feature}: {importance:.3f}")
print(f"{'='*70}\n")

# ==================== XGBoost Model with Hyperparameter Tuning ====================
print("🚀 Training XGBoost Classifier with Hyperparameter Tuning...")

from sklearn.model_selection import GridSearchCV

# Define focused parameter grid for tuning (optimized for quality and speed)
param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [7, 9],
    'learning_rate': [0.05, 0.1],
    'min_child_weight': [1, 3],
    'subsample': [0.9],
    'colsample_bytree': [0.9],
    'gamma': [0, 0.1]
}

# Calculate class weight
scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])

# Base XGBoost model
xgb_base = XGBClassifier(
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    tree_method='hist',  # Faster training
    n_jobs=-1
)

total_combinations = (len(param_grid['n_estimators']) * len(param_grid['max_depth']) * 
                     len(param_grid['learning_rate']) * len(param_grid['min_child_weight']) * 
                     len(param_grid['subsample']) * len(param_grid['colsample_bytree']) * 
                     len(param_grid['gamma']))

print(f"Starting Grid Search with {total_combinations} combinations (optimized for speed)...")

# Grid search with cross-validation (using n_jobs=1 for Docker compatibility)
grid_search = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    cv=3,  # 3-fold cross-validation
    scoring='accuracy',
    n_jobs=1,  # Sequential processing (more stable in Docker)
    verbose=2
)

grid_search.fit(X_train, y_train)

print(f"\n✅ Best Parameters Found:")
for param, value in grid_search.best_params_.items():
    print(f"   {param}: {value}")
print(f"\n✅ Best Cross-Validation Score: {grid_search.best_score_:.3f}")

# Use best model
xgb_model = grid_search.best_estimator_
xgb_preds = xgb_model.predict(X_test)
xgb_pred_proba = xgb_model.predict_proba(X_test)

xgb_acc = accuracy_score(y_test, xgb_preds)
xgb_auc = roc_auc_score(y_test, xgb_pred_proba[:, 1])

print(f"\n{'='*70}")
print(f"XGBOOST RESULTS")
print(f"{'='*70}")
print(f"✅ Accuracy: {xgb_acc:.3f}")
print(f"✅ ROC-AUC Score: {xgb_auc:.3f}")
print(f"\nClassification Report:")
print(classification_report(y_test, xgb_preds, target_names=['No Win', 'Win']))
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, xgb_preds))
print(f"\nFeature Importances:")
for feature, importance in zip(X.columns, xgb_model.feature_importances_):
    print(f"  {feature}: {importance:.3f}")
print(f"{'='*70}\n")

# ==================== Comparison ====================
print(f"\n{'='*70}")
print(f"MODEL COMPARISON")
print(f"{'='*70}")
print(f"Random Forest Accuracy: {rf_acc:.3f} | ROC-AUC: {rf_auc:.3f}")
print(f"XGBoost Accuracy:       {xgb_acc:.3f} | ROC-AUC: {xgb_auc:.3f}")
print(f"\nBest Model: {'Random Forest' if rf_acc > xgb_acc else 'XGBoost' if xgb_acc > rf_acc else 'Tie'}")
print(f"{'='*70}\n")

# ==================== Save Results to Database ====================
engine = create_engine(f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:5432/{os.getenv('DB_NAME')}")

# Save Random Forest predictions
rf_results = X_test.copy()
rf_results['actual'] = y_test.values
rf_results['predicted'] = rf_preds
rf_results['model'] = 'RandomForest'
rf_results.to_sql("f1_predictions_rf", engine, if_exists='replace', index=False)
print("✅ Random Forest predictions saved to f1_predictions_rf table.")

# Save XGBoost predictions
xgb_results = X_test.copy()
xgb_results['actual'] = y_test.values
xgb_results['predicted'] = xgb_preds
xgb_results['model'] = 'XGBoost'
xgb_results.to_sql("f1_predictions_xgb", engine, if_exists='replace', index=False)
print("✅ XGBoost predictions saved to f1_predictions_xgb table.")

# Save comparison metrics
comparison_df = pd.DataFrame({
    'model': ['RandomForest', 'XGBoost'],
    'accuracy': [rf_acc, xgb_acc],
    'roc_auc': [rf_auc, xgb_auc],
    'train_samples': [len(X_train), len(X_train)],
    'test_samples': [len(X_test), len(X_test)]
})
comparison_df.to_sql("f1_model_comparison", engine, if_exists='replace', index=False)
print("✅ Model comparison saved to f1_model_comparison table.")

print("\n🎉 Training complete! Both models trained and predictions saved to database.")
