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

# ==================== NEW FEATURES ====================

st.divider()
st.header("📊 Advanced Analytics")

# Create tabs for different analytics
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Constructor Championship", 
    "📈 Performance Trends", 
    "⚔️ Head-to-Head",
    "🎯 Driver Statistics",
    "🗓️ Season Timeline"
])

with tab1:
    st.subheader(f"Constructor Championship - {year}")
    
    # Get podium finishes (top 3)
    if len(filtered) > 0:
        podium_counts = filtered[filtered["position"].isin([1, 2, 3])].groupby("surname").size().reset_index()
        podium_counts.columns = ["Driver", "Podium Finishes"]
        podium_counts = podium_counts.sort_values("Podium Finishes", ascending=False).head(10)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_podium = px.bar(
                podium_counts, 
                x="Driver", 
                y="Podium Finishes",
                title=f"Top 10 Podium Finishers ({year})",
                color="Podium Finishes",
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig_podium, use_container_width=True)
        
        with col2:
            # Average finishing position by driver
            avg_pos = filtered.groupby("surname")["position"].mean().reset_index()
            avg_pos.columns = ["Driver", "Avg Position"]
            avg_pos = avg_pos.sort_values("Avg Position").head(10)
            
            fig_avg = px.bar(
                avg_pos,
                x="Driver",
                y="Avg Position",
                title=f"Top 10 Average Finishing Position ({year})",
                color="Avg Position",
                color_continuous_scale="Reds_r"
            )
            fig_avg.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_avg, use_container_width=True)

with tab2:
    st.subheader(f"Performance Trends - {year}")
    
    # Points progression throughout the season
    if len(filtered) > 0:
        # Get top 10 drivers by total points
        top_drivers = filtered.groupby("surname")["points"].sum().nlargest(10).index.tolist()
        
        # Filter for top drivers only
        trend_data = filtered[filtered["surname"].isin(top_drivers)].copy()
        trend_data = trend_data.sort_values(["surname", "round"])
        
        # Calculate cumulative points
        trend_data["cumulative_points"] = trend_data.groupby("surname")["points"].cumsum()
        
        fig_trend = px.line(
            trend_data,
            x="round",
            y="cumulative_points",
            color="surname",
            title=f"Championship Points Progression ({year})",
            labels={"round": "Race Round", "cumulative_points": "Cumulative Points", "surname": "Driver"},
            markers=True
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Position trends
        st.subheader("Finishing Position Trends")
        selected_drivers = st.multiselect(
            "Select drivers to compare",
            options=top_drivers,
            default=top_drivers[:3] if len(top_drivers) >= 3 else top_drivers
        )
        
        if selected_drivers:
            position_data = filtered[filtered["surname"].isin(selected_drivers)].copy()
            position_data = position_data.sort_values(["surname", "round"])
            
            fig_pos = px.line(
                position_data,
                x="round",
                y="position",
                color="surname",
                title="Finishing Positions Throughout Season",
                labels={"round": "Race Round", "position": "Finishing Position", "surname": "Driver"},
                markers=True
            )
            fig_pos.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_pos, use_container_width=True)

with tab3:
    st.subheader(f"Head-to-Head Comparison - {year}")
    
    drivers_list = sorted(filtered["surname"].unique())
    
    col1, col2 = st.columns(2)
    with col1:
        driver1 = st.selectbox("Select Driver 1", drivers_list, key="h2h_driver1")
    with col2:
        driver2 = st.selectbox("Select Driver 2", drivers_list, index=min(1, len(drivers_list)-1), key="h2h_driver2")
    
    if driver1 and driver2:
        d1_data = filtered[filtered["surname"] == driver1]
        d2_data = filtered[filtered["surname"] == driver2]
        
        # Create comparison metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(f"{driver1} - Total Points", f"{d1_data['points'].sum():.0f}")
            st.metric(f"{driver2} - Total Points", f"{d2_data['points'].sum():.0f}")
        
        with col2:
            st.metric(f"{driver1} - Wins", len(d1_data[d1_data["position"] == 1]))
            st.metric(f"{driver2} - Wins", len(d2_data[d2_data["position"] == 1]))
        
        with col3:
            st.metric(f"{driver1} - Podiums", len(d1_data[d1_data["position"] <= 3]))
            st.metric(f"{driver2} - Podiums", len(d2_data[d2_data["position"] <= 3]))
        
        with col4:
            st.metric(f"{driver1} - Avg Pos", f"{d1_data['position'].mean():.1f}")
            st.metric(f"{driver2} - Avg Pos", f"{d2_data['position'].mean():.1f}")
        
        # Points comparison chart
        comparison_df = pd.DataFrame({
            "Driver": [driver1, driver2],
            "Total Points": [d1_data['points'].sum(), d2_data['points'].sum()],
            "Wins": [len(d1_data[d1_data["position"] == 1]), len(d2_data[d2_data["position"] == 1])],
            "Podiums": [len(d1_data[d1_data["position"] <= 3]), len(d2_data[d2_data["position"] <= 3])]
        })
        
        fig_comparison = px.bar(
            comparison_df.melt(id_vars="Driver", var_name="Metric", value_name="Count"),
            x="Metric",
            y="Count",
            color="Driver",
            barmode="group",
            title=f"{driver1} vs {driver2} - Season Comparison"
        )
        st.plotly_chart(fig_comparison, use_container_width=True)

with tab4:
    st.subheader(f"Driver Statistics - {year}")
    
    # Calculate various statistics
    stats_data = []
    for driver in filtered["surname"].unique():
        driver_df = filtered[filtered["surname"] == driver]
        
        stats_data.append({
            "Driver": driver,
            "Races": len(driver_df),
            "Wins": len(driver_df[driver_df["position"] == 1]),
            "Podiums": len(driver_df[driver_df["position"] <= 3]),
            "Top 5": len(driver_df[driver_df["position"] <= 5]),
            "Top 10": len(driver_df[driver_df["position"] <= 10]),
            "Total Points": driver_df["points"].sum(),
            "Avg Position": driver_df["position"].mean(),
            "Best Finish": driver_df["position"].min(),
            "Worst Finish": driver_df["position"].max()
        })
    
    stats_df = pd.DataFrame(stats_data)
    stats_df = stats_df.sort_values("Total Points", ascending=False)
    
    # Add formatting
    st.dataframe(
        stats_df.style.background_gradient(subset=["Total Points"], cmap="Greens")
                     .background_gradient(subset=["Wins"], cmap="Blues")
                     .format({
                         "Total Points": "{:.0f}",
                         "Avg Position": "{:.2f}",
                     }),
        use_container_width=True
    )
    
    # Win rate calculation
    stats_df["Win Rate %"] = (stats_df["Wins"] / stats_df["Races"] * 100).round(1)
    stats_df["Podium Rate %"] = (stats_df["Podiums"] / stats_df["Races"] * 100).round(1)
    
    col1, col2 = st.columns(2)
    
    with col1:
        top_win_rate = stats_df.nlargest(10, "Win Rate %")
        fig_win_rate = px.bar(
            top_win_rate,
            x="Driver",
            y="Win Rate %",
            title="Top 10 Win Rate %",
            color="Win Rate %",
            color_continuous_scale="RdYlGn"
        )
        st.plotly_chart(fig_win_rate, use_container_width=True)
    
    with col2:
        top_podium_rate = stats_df.nlargest(10, "Podium Rate %")
        fig_podium_rate = px.bar(
            top_podium_rate,
            x="Driver",
            y="Podium Rate %",
            title="Top 10 Podium Rate %",
            color="Podium Rate %",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_podium_rate, use_container_width=True)

with tab5:
    st.subheader(f"Season Timeline - {year}")
    
    # Race winners timeline
    winners_timeline = filtered[filtered["position"] == 1].copy()
    winners_timeline = winners_timeline.sort_values("round")
    
    if len(winners_timeline) > 0:
        fig_timeline = px.scatter(
            winners_timeline,
            x="round",
            y="surname",
            size="points",
            color="surname",
            title=f"Race Winners by Round ({year})",
            labels={"round": "Race Round", "surname": "Winner"},
            hover_data=["name", "points"]
        )
        fig_timeline.update_traces(marker=dict(size=20, line=dict(width=2, color='DarkSlateGrey')))
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Show race-by-race results
        st.subheader("Race-by-Race Winners")
        winners_table = winners_timeline[["round", "name", "surname", "points"]].copy()
        winners_table.columns = ["Round", "Race", "Winner", "Points"]
        st.dataframe(winners_table, use_container_width=True)

st.divider()

# ==================== END NEW FEATURES ====================

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
        # Note: Predictions table has year, round, win_rate, avg_points, actual, predicted
        # We can't easily match back to specific drivers without driverId in predictions table
        # For now, show aggregated statistics
        
        # Get predictions for selected year
        y_test = pred_filtered['actual'].values
        preds = pred_filtered['predicted'].values
        
        # Calculate metrics
        correct = (y_test == preds).sum()
        total = len(y_test)
        acc = accuracy_score(y_test, preds)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(f"Total Predictions", total)
            st.metric("Correct Predictions", correct)
            st.metric("Accuracy", f"{acc:.1%}")
        
        with col2:
            st.subheader(f"📊 Predictions Breakdown - {year}")
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
        
        st.subheader(f"✅ Classification Report - {year}")
        report_dict = classification_report(y_test, preds, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df, use_container_width=True)
        
        st.subheader(f"🟦 Confusion Matrix - {year}")
        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted Win")
        ax.set_ylabel("Actual Win")
        ax.set_title(f"Confusion Matrix - {year}")
        st.pyplot(fig)
        
        # Show sample predictions
        st.subheader(f"📋 Sample Predictions")
        st.dataframe(pred_filtered[['year', 'round', 'win_rate', 'avg_points', 'actual', 'predicted']].head(20), use_container_width=True)
    else:
        st.info(f"No predictions available for {year}")
else:
    st.info("No ML predictions available yet. Run the ML training service first.")