FROM python:3.11

WORKDIR /app

# install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements dan install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install PostgreSQL client di ETL Container
RUN apt-get update && apt-get install -y postgresql-client  

# Copy seluruh project termasuk folder data
COPY . .

# Copy wait-for-postgres script
COPY wait-for-postgres.sh .
RUN chmod +x wait-for-postgres.sh

# Jalankan ETL
CMD ["python", "etl/etl_pipeline.py"]