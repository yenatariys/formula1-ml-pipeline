FROM python:3.10-slim-bookworm

WORKDIR /app

# Install PostgreSQL client for wait-for-postgres script
RUN apt-get update && apt-get install -y postgresql-client && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install pandas scikit-learn psycopg2-binary sqlalchemy xgboost

# Copy wait-for-postgres script
COPY wait-for-postgres.sh .
RUN chmod +x wait-for-postgres.sh

# Copy train scripts
COPY ml/train.py .
COPY ml/train_comparison.py .
COPY ml/train_with_best_params.py .

CMD ["./wait-for-postgres.sh", "f1_postgres", "python", "train_comparison.py"]