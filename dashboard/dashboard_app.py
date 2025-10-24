import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(page_title="F1 Race Results Dashboard", layout="wide")

# Database connection
engine = create_engine("postgresql+psycopg2://admin:admin123@f1_postgres:5432/f1_data")

# LOAD Race Results 
@st.cache_data
def load_results():
    return pd.read_sql("SELECT * FROM f1_results", engine)


df = load_results()

# Dashboard Race Results
st.title("🏎️ Formula 1 Race Results Dashboard")

# Show total records
st.metric("Total Race Results", len(df))

# Year filter
year = st.selectbox("Select Season", sorted(df["year"].unique(), reverse=True))
filtered = df[df["year"] == year]

col1, col2 = st.columns(2)

with col1:
    # Wins by driver (position 1)
    winners = filtered[filtered["position"] == 1].groupby("surname")["raceId"].count().reset_index()
    winners.columns = ["Driver", "Wins"]
    fig1 = px.bar(winners, x="Driver", y="Wins", title=f"Race Wins by Driver ({year})")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    # Points distribution
    points_by_driver = filtered.groupby("surname")["points"].sum().reset_index().sort_values("points", ascending=False).head(10)
    points_by_driver.columns = ["Driver", "Total Points"]
    fig2 = px.pie(points_by_driver, values="Total Points", names="Driver", title=f"Top 10 Drivers by Points ({year})")
    st.plotly_chart(fig2, use_container_width=True)

# Detailed results table
st.subheader(f"Detailed Results - {year}")
st.dataframe(filtered[["name", "round", "surname", "position", "points"]].sort_values(["round", "position"]), use_container_width=True)

# LOAD ML Predictions
@st.cache_data(ttl=60)  # Cache for 60 seconds, then reload
def load_predictions():
    try:
        return pd.read_sql("SELECT * FROM f1_predictions", engine)
    except:
        return pd.DataFrame()  # Return empty if table doesn't exist

df_preds = load_predictions()

# Dashboard Predictions
if not df_preds.empty:
    st.subheader("🤖 Race Win Predictions")
    
    # Show overall stats
    st.info(f"Total predictions in database: {len(df_preds)} across years {df_preds['year'].min():.0f}-{df_preds['year'].max():.0f}")
    
    # Filter predictions by season first
    pred_filtered = df_preds[df_preds["year"] == year].copy()
    
    if len(pred_filtered) > 0:
        # Merge predictions with race results to get driver names
        # Match on year, round, and points to identify the driver
        pred_with_drivers = pred_filtered.merge(
            df[['year', 'round', 'points', 'surname', 'name']],
            on=['year', 'round', 'points'],
            how='left'
        )
        
        # Driver Filter
        available_drivers = sorted(pred_with_drivers['surname'].dropna().unique())
        if len(available_drivers) > 0:
            selected_driver = st.selectbox("Select Driver for Predictions", ["All Drivers"] + available_drivers)
            
            if selected_driver != "All Drivers":
                pred_filtered_display = pred_with_drivers[pred_with_drivers['surname'] == selected_driver].copy()
            else:
                pred_filtered_display = pred_with_drivers.copy()
        else:
            st.warning("Could not match predictions to drivers. Showing all predictions.")
            pred_filtered_display = pred_filtered.copy()
            selected_driver = "All Drivers"
        
        # Get predictions for selected filter
        y_test = pred_filtered_display['actual'].values
        preds = pred_filtered_display['predicted'].values
        
        # Calculate metrics
        correct = (y_test == preds).sum()
        total = len(y_test)
        acc = accuracy_score(y_test, preds)
        
        # Show filter context
        filter_text = f" - {selected_driver}" if selected_driver != "All Drivers" else ""
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(f"Total Predictions{filter_text}", total)
            st.metric("Correct Predictions", correct)
            st.metric("Accuracy", f"{acc:.1%}")
        
        with col2:
            st.subheader(f"📊 Predictions Breakdown - {year}{filter_text}")
            breakdown_df = pd.DataFrame({
                'Category': [
                    'True Negatives (Predicted No Win, Actual No Win)',
                    'True Positives (Predicted Win, Actual Win)',
                    'False Negatives (Predicted No Win, Actual Win)',
                    'False Positives (Predicted Win, Actual No Win)'
                ],
                'Count': [
                    ((preds == 0) & (y_test == 0)).sum(),
                    ((preds == 1) & (y_test == 1)).sum(),
                    ((preds == 0) & (y_test == 1)).sum(),
                    ((preds == 1) & (y_test == 0)).sum()
                ]
            })
            st.dataframe(breakdown_df, use_container_width=True)
        
        st.subheader(f"✅ Classification Report - {year}{filter_text}")
        report_dict = classification_report(y_test, preds, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df, use_container_width=True)
        
        st.subheader(f"🟦 Confusion Matrix - {year}{filter_text}")
        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted Win")
        ax.set_ylabel("Actual Win")
        ax.set_title(f"Confusion Matrix - {year}")
        st.pyplot(fig)
        
        # Show sample predictions
        st.subheader(f"📋 Sample Predictions{' - ' + selected_driver if selected_driver != 'All Drivers' else ''}")
        
        # Prepare display columns
        display_cols = ['year', 'round']
        if 'surname' in pred_filtered_display.columns and 'name' in pred_filtered_display.columns:
            display_cols.extend(['surname', 'name'])
        display_cols.extend(['points', 'actual', 'predicted'])
        
        st.dataframe(pred_filtered_display[display_cols].head(20), use_container_width=True)
    else:
        st.info(f"No predictions available for {year}")
else:
    st.info("No ML predictions available yet. Run the ML training service first.")