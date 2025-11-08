# 01 – Environment Setup

Use this guide to prepare local tooling and container services before running any pipelines.

## 1. Install prerequisites
- Docker Desktop with Compose V2 enabled.
- Python 3.10+ (for optional host execution of classic pipeline scripts).
- `git` for cloning and version control.

## 2. Clone and configure the repository
```powershell
# Windows PowerShell example
git clone https://github.com/yenatariys/formula1-ml-pipeline.git
cd formula1-ml-pipeline

# Copy sample environment file if you use one (optional)
# cp .env.example .env
```

## 3. Build base images
The project relies on several custom images. Build them once so subsequent `docker-compose up` runs start quickly.
```powershell
docker-compose build postgres etl ml_train dashboard_classic dashboard_bigdata spark-master spark-worker-1 spark-worker-2
```

## 4. Start core infrastructure
Bring up Postgres and the Spark cluster; leave dashboards and ETL off until you need them.
```powershell
docker-compose up -d postgres spark-master spark-worker-1 spark-worker-2
```

## 5. Verify connectivity
```powershell
docker-compose ps
```
Healthy containers should show a `running` state. If Postgres is not healthy yet, wait for the health check to pass or inspect logs with `docker-compose logs postgres`.
