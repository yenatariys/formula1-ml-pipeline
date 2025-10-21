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

df = pd.read_sql("SELECT * FROM f1_results", conn)

df['win'] = (df['position'] == 1).astype(int)
X = df[['grid', 'points']]
y = df['win']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)

acc = accuracy_score(y_test, preds)
print(f"✅ Model Accuracy: {acc:.3f}")

results = X_test.copy()
results['actual'] = y_test.values
results['predicted'] = preds

engine = create_engine(f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:5432/{os.getenv('DB_NAME')}")
results.to_sql("f1_predictions", engine, if_exists='replace', index=False)
print("Predictions saved to database.")