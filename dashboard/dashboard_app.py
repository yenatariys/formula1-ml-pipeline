import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px

st.set_page_config(page_title="F1 Race Results Dashboard", layout="wide")

engine = create_engine("postgresql+psycopg2://admin:admin123@postgres:5432/f1_data")

@st.cache_data
def load_data():
    return pd.read_sql("SELECT * FROM race_results", engine)

df = load_data()

st.title("🏎️ Formula 1 Race Results Dashboard")

year = st.selectbox("Select Season", sorted(df["season"].unique(), reverse=True))
filtered = df[df["season"] == year]

col1, col2 = st.columns(2)

with col1:
    fig1 = px.bar(filtered, x="race_name", y="laps", title=f"Number of Laps ({year})")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    winners = filtered.groupby("winner")["race_id"].count().reset_index()
    fig2 = px.pie(winners, values="race_id", names="winner", title="Wins by Driver")
    st.plotly_chart(fig2, use_container_width=True)