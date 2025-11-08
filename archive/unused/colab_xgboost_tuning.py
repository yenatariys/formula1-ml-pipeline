# ============================================================================
# F1 XGBoost Hyperparameter Tuning - Google Colab
# ============================================================================
# Instructions:
# 1. Go to https://colab.research.google.com/
# 2. Create a new notebook
# 3. Copy-paste this entire script into a cell
# 4. Run the cell
# 5. Upload f1_results_transformed.csv when prompted
# 6. Wait ~10-20 minutes for results (much faster than local!)
# 7. Copy the best parameters printed at the end
# ============================================================================

# Install required packages
print("📦 Installing packages...")
# !pip install -q xgboost scikit-learn pandas scipy tqdm

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from scipy.stats import randint, uniform
from tqdm.auto import tqdm
import warnings
import json
import time
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: Upload the CSV file
# ============================================================================
print("\n📁 Please upload f1_results_transformed.csv...")
from google.colab import files
uploaded = files.upload()

# ============================================================================
# STEP 2: Load and prepare data
# ============================================================================
print("\n🔄 Loading and preparing data...")
df = pd.read_csv('f1_results_transformed.csv')
print(f"✅ Loaded {len(df)} rows")

# Create target variable
df['win'] = (df['position'] == 1).astype(int)

# Feature engineering - same as local
df = df.sort_values(['driverId', 'year', 'round'])
df['races_so_far'] = df.groupby('driverId').cumcount()
df['wins_so_far'] = df.groupby('driverId')['win'].cumsum() - df['win']
df['win_rate'] = df['wins_so_far'] / (df['races_so_far'] + 1)
df['win_rate'] = df['win_rate'].fillna(0)

df['points_so_far'] = df.groupby('driverId')['points'].cumsum() - df['points']
df['avg_points'] = df['points_so_far'] / (df['races_so_far'] + 1)
df['avg_points'] = df['avg_points'].fillna(0)

# Filter for drivers with history
df_model = df[df['races_so_far'] > 5].copy()
print(f"✅ After filtering: {len(df_model)} rows")
print(f"   Win distribution: {df_model['win'].value_counts().to_dict()}")

# Prepare features
X = df_model[['year', 'round', 'win_rate', 'avg_points']]
y = df_model['win']

# Train-test split (same random_state as local for consistency)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✅ Training samples: {len(X_train)}")
print(f"✅ Testing samples: {len(X_test)}")

# ============================================================================
# STEP 3: XGBoost Hyperparameter Tuning
# ============================================================================
print("\n" + "="*70)
print("🚀 STARTING XGBOOST HYPERPARAMETER TUNING")
print("="*70)

# Define parameter distributions (same as local)
param_distributions = {
    'n_estimators': randint(100, 800),
    'max_depth': randint(5, 15),
    'learning_rate': uniform(0.01, 0.29),
    'min_child_weight': randint(1, 10),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'gamma': uniform(0, 0.5),
    'reg_alpha': uniform(0, 0.5),
    'reg_lambda': uniform(0.5, 2.5)
}

# Calculate class weight
scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
print(f"Class weight (scale_pos_weight): {scale_pos_weight:.2f}")

# Base XGBoost model with GPU support
xgb_base = XGBClassifier(
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    tree_method='gpu_hist',  # Use GPU for faster training!
    n_jobs=-1
)

n_iter = 2500

print(f"\n⚙️  Randomized Search Configuration:")
print(f"   - Combinations to try: {n_iter}")
print(f"   - Cross-validation folds: 5")
print(f"   - Total fits: {n_iter * 5} = 12,500")
print(f"   - Using all CPU cores for parallel processing")
print(f"\n⏱️  Estimated time: 5-10 minutes on Colab with GPU!")
print(f"   💡 Make sure to enable GPU: Runtime -> Change runtime type -> GPU")
print("\n🏃 Starting training...\n")

# Randomized search with custom callback for progress tracking
class ProgressCallback:
    def __init__(self, n_iter, cv):
        self.n_iter = n_iter
        self.cv = cv
        self.total = n_iter * cv
        self.pbar = tqdm(total=self.total, desc="Training Progress", 
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        self.start_time = time.time()
        
    def __call__(self, *args, **kwargs):
        self.pbar.update(1)
        if self.pbar.n % 100 == 0:  # Update every 100 fits
            elapsed = time.time() - self.start_time
            rate = self.pbar.n / elapsed if elapsed > 0 else 0
            remaining = (self.total - self.pbar.n) / rate if rate > 0 else 0
            print(f"✓ Completed {self.pbar.n}/{self.total} fits | "
                  f"Speed: {rate:.1f} fits/sec | "
                  f"ETA: {remaining/60:.1f} min")
    
    def close(self):
        self.pbar.close()

# Create callback
progress = ProgressCallback(n_iter, 5)

# Randomized search
random_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=param_distributions,
    n_iter=n_iter,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0,  # Turn off default verbose
    random_state=42,
    error_score='raise'
)

# FIT THE MODEL with progress tracking
try:
    # Monitor training with a timer thread
    import threading
    
    fit_complete = False
    start_time = time.time()
    
    def monitor_progress():
        while not fit_complete:
            time.sleep(10)  # Update every 10 seconds
            if not fit_complete:
                elapsed = time.time() - start_time
                print(f"⏱️  Training in progress... Elapsed time: {elapsed/60:.1f} minutes")
    
    monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
    monitor_thread.start()
    
    random_search.fit(X_train, y_train)
    fit_complete = True
    
    total_time = time.time() - start_time
    print(f"\n✅ Training completed in {total_time/60:.1f} minutes!")
    
finally:
    fit_complete = True
    time.sleep(0.5)  # Let monitor thread finish

# ============================================================================
# STEP 4: Results
# ============================================================================
print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)

# Best parameters
best_params = random_search.best_params_
print("\n🏆 BEST PARAMETERS FOUND:")
print("="*70)
for param, value in best_params.items():
    print(f"   {param}: {value}")

print(f"\n✅ Best Cross-Validation Score: {random_search.best_score_:.3f}")

# Test the best model
xgb_model = random_search.best_estimator_
xgb_preds = xgb_model.predict(X_test)
xgb_pred_proba = xgb_model.predict_proba(X_test)

xgb_acc = accuracy_score(y_test, xgb_preds)
xgb_auc = roc_auc_score(y_test, xgb_pred_proba[:, 1])

print("\n" + "="*70)
print("📊 TEST SET RESULTS")
print("="*70)
print(f"✅ Accuracy: {xgb_acc:.3f}")
print(f"✅ ROC-AUC Score: {xgb_auc:.3f}")
print(f"\nClassification Report:")
print(classification_report(y_test, xgb_preds, target_names=['No Win', 'Win']))
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, xgb_preds))

print(f"\nFeature Importances:")
for feature, importance in zip(X.columns, xgb_model.feature_importances_):
    print(f"  {feature}: {importance:.3f}")

# ============================================================================
# STEP 5: Download best parameters as JSON
# ============================================================================
print("\n" + "="*70)
print("💾 SAVING BEST PARAMETERS")
print("="*70)

# Save as JSON
output = {
    'best_parameters': best_params,
    'best_cv_score': float(random_search.best_score_),
    'test_accuracy': float(xgb_acc),
    'test_roc_auc': float(xgb_auc),
    'scale_pos_weight': float(scale_pos_weight)
}

with open('xgboost_best_params.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\n✅ Best parameters saved to: xgboost_best_params.json")
print("\n📥 Downloading file...")
files.download('xgboost_best_params.json')

# Also print Python code to copy-paste
print("\n" + "="*70)
print("📋 COPY-PASTE THIS INTO YOUR LOCAL CODE:")
print("="*70)
print(f"""
xgb_model = XGBClassifier(
    n_estimators={best_params['n_estimators']},
    max_depth={best_params['max_depth']},
    learning_rate={best_params['learning_rate']:.6f},
    min_child_weight={best_params['min_child_weight']},
    subsample={best_params['subsample']:.6f},
    colsample_bytree={best_params['colsample_bytree']:.6f},
    gamma={best_params['gamma']:.6f},
    reg_alpha={best_params['reg_alpha']:.6f},
    reg_lambda={best_params['reg_lambda']:.6f},
    random_state=42,
    scale_pos_weight={scale_pos_weight:.6f},
    eval_metric='logloss',
    tree_method='hist',  # Use 'hist' for CPU on local machine
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
""")

print("\n" + "="*70)
print("🎉 ALL DONE!")
print("="*70)
print("\nNext steps:")
print("1. Copy the parameters above")
print("2. Paste them into your local train_comparison.py")
print("3. Remove the RandomizedSearchCV code")
print("4. Run locally - training will take only ~5 seconds!")
print("="*70)
