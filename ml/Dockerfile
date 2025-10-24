FROM python:3.10-slim

WORKDIR /app

# Install PostgreSQL client for wait-for-postgres script
RUN apt-get update && apt-get install -y postgresql-client && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install pandas scikit-learn psycopg2-binary sqlalchemy

# Copy wait-for-postgres script
COPY wait-for-postgres.sh .
RUN chmod +x wait-for-postgres.sh

# Copy train script
COPY ml/train.py .

CMD ["./wait-for-postgres.sh", "f1_postgres", "python", "train.py"]