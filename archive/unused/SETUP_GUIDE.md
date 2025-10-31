# Formula 1 ML Pipeline - Complete Setup Guide

## 📋 Prerequisites
- Docker Desktop installed and running
- Git (to clone the repository)
- Minimum 8GB RAM, 20GB free disk space

---

## 🚀 Fresh Installation (From Scratch)

### Step 1: Clean Docker Environment (if needed)
```powershell
# Remove all containers, images, volumes, and networks
docker system prune -a --volumes -f

# Verify everything is clean
docker ps -a
docker images
docker volume ls
```

### Step 2: Clone/Navigate to Project
```powershell
cd D:\Work\Github\formula1-ml-pipeline
```

### Step 3: Build All Containers
```powershell
# Build all Docker images (takes 3-5 minutes)
docker-compose build
```

### Step 4: Start Core Services
```powershell
# Start PostgreSQL, Spark, ETL, Dashboard, and pgAdmin
docker-compose up -d postgres spark-master etl dashboard pgadmin
```

**Wait ~30-60 seconds** for ETL to complete data transformation.

You can monitor ETL progress:
```powershell
docker logs etl_service -f
```

Look for: `"Transformation complete and saved to PostgreSQL"`

### Step 5: Train ML Models
```powershell
# Train both Random Forest and XGBoost models (~10 seconds)
docker-compose run --rm ml_train python train_with_best_params.py
```

**Expected Output:**
```
✅ Random Forest Accuracy: 0.914 | ROC-AUC: 0.855
✅ XGBoost Accuracy: 0.894 | ROC-AUC: 0.825
✅ Best Model: Random Forest
🎉 Training complete! Both models trained and predictions saved to database.
```

### Step 6: Access the Dashboard
Open your browser: **http://localhost:8501**

You should see:
- ✅ Formula 1 Race Results Dashboard
- ✅ Model Comparison Charts
- ✅ Race Win Predictions
- ✅ Historical Data Visualizations

---

## 🔄 Daily Workflow (After Initial Setup)

### Start Services
```powershell
# Start all services
docker-compose up -d postgres spark-master etl dashboard pgadmin

# Wait 30 seconds for ETL, then train models
docker-compose run --rm ml_train python train_with_best_params.py
```

### Stop Services
```powershell
docker-compose down
```

### Restart Everything
```powershell
docker-compose down
docker-compose up -d postgres spark-master etl dashboard pgadmin
docker-compose run --rm ml_train python train_with_best_params.py
```

---

## 🔍 Verification & Troubleshooting

### Check Running Containers
```powershell
docker ps
```

**Expected containers:**
- `f1_postgres` (PostgreSQL database)
- `spark-master` (Spark cluster)
- `etl_service` (ETL pipeline)
- `f1_dashboard` (Streamlit dashboard)
- `f1_pgadmin` (Database admin interface)

### Check Logs
```powershell
# ETL logs
docker logs etl_service

# Dashboard logs
docker logs f1_dashboard

# ML training logs
docker logs ml_train
```

### Verify Database Tables
```powershell
# Connect to PostgreSQL and check tables
docker exec -it f1_postgres psql -U admin -d f1_data

# Inside psql:
\dt                                    # List all tables
SELECT COUNT(*) FROM f1_results_transformed;   # Should return 15806
SELECT * FROM f1_model_comparison;    # Check model results
\q                                     # Exit
```

### Access pgAdmin (Database GUI)
- **URL**: http://localhost:5050
- **Email**: admin@admin.com
- **Password**: admin

**Add Server Connection:**
1. Right-click Servers → Register → Server
2. General tab: Name = "F1 Database"
3. Connection tab:
   - Host: `f1_postgres`
   - Port: `5432`
   - Database: `f1_data`
   - Username: `admin`
   - Password: `admin123`
4. Save

---

## 📊 What Gets Created

### Database Tables
| Table Name | Rows | Description |
|------------|------|-------------|
| `f1_results_transformed` | 15,806 | Transformed race results from ETL |
| `f1_predictions_rf` | 2,630 | Random Forest predictions |
| `f1_predictions_xgb` | 2,630 | XGBoost predictions |
| `f1_model_comparison` | 2 | Model accuracy comparison |

### ML Model Performance
- **Random Forest**: 91.4% accuracy, 0.855 ROC-AUC (Winner 🏆)
- **XGBoost**: 89.4% accuracy, 0.825 ROC-AUC

---

## 🛠️ Common Issues & Solutions

### Issue: Dashboard shows "No data available"
**Solution:**
```powershell
# Restart dashboard to clear cache
docker-compose restart dashboard
```

### Issue: ETL hasn't completed
**Solution:**
```powershell
# Check ETL logs
docker logs etl_service -f

# Wait for: "Transformation complete and saved to PostgreSQL"
# Then run ML training
```

### Issue: "Can't connect to f1_postgres"
**Solution:**
```powershell
# Ensure PostgreSQL is running
docker ps | grep f1_postgres

# Restart if needed
docker-compose restart postgres

# Wait 10 seconds, then retry
```

### Issue: Port already in use
**Solution:**
```powershell
# Check what's using the port (e.g., 5432, 8501, 5050)
netstat -ano | findstr :5432

# Stop the conflicting service or change port in docker-compose.yml
```

### Issue: Out of disk space
**Solution:**
```powershell
# Clean unused Docker resources
docker system prune -a --volumes

# Keep only what you need
```

---

## 🔧 Advanced: Retrain Models with Different Parameters

### Option 1: Quick Local Tuning (5-10 minutes)
```powershell
# Edit ml/train_comparison.py to adjust grid search parameters
docker-compose build ml_train
docker-compose run --rm ml_train python train_comparison.py
```

### Option 2: Google Colab Tuning (Faster, Free GPU)
1. Upload `f1_results_transformed.csv` to Colab
2. Run `colab_xgboost_tuning.py` script
3. Download best parameters
4. Update `ml/train_with_best_params.py` with new values
5. Rebuild and run:
```powershell
docker-compose build ml_train
docker-compose run --rm ml_train python train_with_best_params.py
```

---

## 📂 Project Structure

```
formula1-ml-pipeline/
├── data/                    # Raw CSV files
├── db/                      # Database init scripts
│   └── init.sql
├── etl/                     # ETL pipeline
│   ├── etl_pipeline.py
│   ├── extract_data.py
│   ├── load_data.py
│   └── transform_data.py
├── ml/                      # Machine Learning
│   ├── train_comparison.py         # Grid search tuning
│   ├── train_with_best_params.py   # Fast training (use this!)
│   └── Dockerfile
├── dashboard/               # Streamlit dashboard
│   ├── dashboard_app.py
│   ├── Dockerfile
│   └── requirements.txt
├── docker/                  # Helper scripts
│   └── wait_for_table.py
├── docker-compose.yml       # Main orchestration
├── colab_xgboost_tuning.py # Google Colab script
└── SETUP_GUIDE.md          # This file
```

---

## 🎯 Quick Reference Commands

```powershell
# 🏁 FRESH START (everything from scratch)
docker system prune -a --volumes -f
docker-compose build
docker-compose up -d postgres spark-master etl dashboard pgadmin
# Wait 30 seconds
docker-compose run --rm ml_train python train_with_best_params.py

# 📊 DAILY USE (services already built)
docker-compose up -d postgres spark-master etl dashboard pgadmin
docker-compose run --rm ml_train python train_with_best_params.py

# 🔍 CHECK STATUS
docker ps                                    # Running containers
docker logs etl_service                      # ETL logs
docker logs f1_dashboard                     # Dashboard logs
docker exec -it f1_postgres psql -U admin -d f1_data  # Database access

# 🛑 STOP EVERYTHING
docker-compose down

# 🧹 CLEAN UP
docker-compose down -v                       # Stop and remove volumes
docker system prune -a --volumes -f          # Deep clean
```

---

## 📞 Need Help?

### Check Container Health
```powershell
docker ps -a
docker logs <container_name>
```

### Database Connection Issues
```powershell
# Test PostgreSQL connection
docker exec -it f1_postgres psql -U admin -d f1_data -c "SELECT COUNT(*) FROM f1_results_transformed;"
```

### Reset Everything
```powershell
# Nuclear option: delete everything and start fresh
docker-compose down -v
docker system prune -a --volumes -f
# Then follow "Fresh Installation" steps
```

---

## ✅ Success Checklist

- [ ] All containers running: `docker ps` shows 5 containers
- [ ] Database populated: 15,806 rows in `f1_results_transformed`
- [ ] ML models trained: Accuracy ~91% (RF) and ~89% (XGB)
- [ ] Dashboard accessible: http://localhost:8501 loads correctly
- [ ] No errors in logs: `docker logs f1_dashboard` shows no errors

---

## 🎉 You're All Set!

**Access Points:**
- 📊 **Dashboard**: http://localhost:8501
- 🗄️ **pgAdmin**: http://localhost:5050
- 💾 **PostgreSQL**: localhost:5432

**Next Steps:**
1. Explore the dashboard visualizations
2. Try different year filters
3. Examine model predictions
4. Compare Random Forest vs XGBoost performance

Enjoy your Formula 1 ML Pipeline! 🏎️💨
