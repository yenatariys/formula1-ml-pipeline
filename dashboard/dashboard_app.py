import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(page_title="F1 Race Results Dashboard", layout="wide")

# Database connection
engine = create_engine("postgresql+psycopg2://admin:admin123@f1_postgres:5432/f1_data")

# LOAD Race Results 
@st.cache_data
def load_results():
    return pd.read_sql("SELECT * FROM f1_results_transformed", engine)

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
@st.cache_data
def load_predictions():
    return pd.read_sql("SELECT * FROM f1_predictions", engine)

df_preds = load_predictions()

# Dashboard Predictions
st.subheader("🤖 Race Win Predictions")

# Merge predictions dengan info driver/race
df_preds_display = df_preds.merge(
    df[["raceId","name","surname","year"]], on=["surname"], how="left"
)

# Filter predictions by season
pred_filtered = df_preds_display[df_preds_display["year"] == year]

# Driver Filter
drivers = sorted(pred_filtered["surname"].unique())
selected_driver = st.selectbox("Select Driver for Predictions", ["All"] + drivers)

if selected_driver != "All":
    pred_filtered_driver = pred_filtered[pred_filtered["surname"] == selected_driver]
else:
    pred_filtered_driver = pred_filtered

y_test = pred_filtered_driver['actual'].values
preds = pred_filtered_driver['predicted'].values

st.subheader(f"✅ Classification Report - {year} - {selected_driver}")
if len(y_test) > 0:
    report_dict = classification_report(y_test, preds, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df)

    st.subheader(f"🟦 Confusion Matrix - {year} - {selected_driver}")
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
else:
    st.write("No data available for this driver in the selected season")