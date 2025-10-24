import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px

st.set_page_config(page_title="F1 Race Results Dashboard", layout="wide")

engine = create_engine("postgresql+psycopg2://admin:admin123@f1_postgres:5432/f1_data")

@st.cache_data
def load_data():
    return pd.read_sql("SELECT * FROM f1_results", engine)

df = load_data()

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