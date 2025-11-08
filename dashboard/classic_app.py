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

@st.cache_data(ttl=60)
def load_rf_predictions():
    try:
        return pd.read_sql("SELECT * FROM f1_predictions_rf", engine)
    except:
        return pd.DataFrame()

@st.cache_data(ttl=60)
def load_xgb_predictions():
    try:
        return pd.read_sql("SELECT * FROM f1_predictions_xgb", engine)
    except:
        return pd.DataFrame()

df_preds = load_predictions()
rf_preds_df = load_rf_predictions()
xgb_preds_df = load_xgb_predictions()

# Dashboard Predictions - Show Both Models
st.subheader("🤖 Race Win Predictions - Model Comparison")

if not rf_preds_df.empty and not xgb_preds_df.empty:
    # Filter predictions by season
    rf_filtered = rf_preds_df[rf_preds_df["year"] == year].copy()
    xgb_filtered = xgb_preds_df[xgb_preds_df["year"] == year].copy()
    
    if len(rf_filtered) > 0 and len(xgb_filtered) > 0:
        # Show metrics for both models side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🌲 Random Forest")
            rf_y_test = rf_filtered['actual'].values
            rf_preds = rf_filtered['predicted'].values
            rf_correct = (rf_y_test == rf_preds).sum()
            rf_total = len(rf_y_test)
            rf_acc = accuracy_score(rf_y_test, rf_preds)
            
            st.metric("Total Predictions", rf_total)
            st.metric("Correct Predictions", rf_correct)
            st.metric("Accuracy", f"{rf_acc:.1%}")
            
            # Confusion matrix
            rf_cm = confusion_matrix(rf_y_test, rf_preds)
            fig_rf = px.imshow(
                rf_cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['No Win', 'Win'],
                y=['No Win', 'Win'],
                title=f"Random Forest Confusion Matrix ({year})",
                color_continuous_scale='Greens',
                text_auto=True
            )
            st.plotly_chart(fig_rf, use_container_width=True, key="rf_cm_predictions")
        
        with col2:
            st.markdown("### 🚀 XGBoost")
            xgb_y_test = xgb_filtered['actual'].values
            xgb_preds = xgb_filtered['predicted'].values
            xgb_correct = (xgb_y_test == xgb_preds).sum()
            xgb_total = len(xgb_y_test)
            xgb_acc = accuracy_score(xgb_y_test, xgb_preds)
            
            st.metric("Total Predictions", xgb_total)
            st.metric("Correct Predictions", xgb_correct)
            st.metric("Accuracy", f"{xgb_acc:.1%}")
            
            # # Confusion matrix
            # xgb_cm = confusion_matrix(xgb_y_test, xgb_preds)
            # fig_xgb = px.imshow(
            #     xgb_cm,
            #     labels=dict(x="Predicted", y="Actual", color="Count"),
            #     x=['No Win', 'Win'],
            #     y=['No Win', 'Win'],
            #     title=f"XGBoost Confusion Matrix ({year})",
            #     color_continuous_scale='Reds',
            #     text_auto=True
            # )
            # st.plotly_chart(fig_xgb, use_container_width=True, key="xgb_cm_predictions")
        
        # Classification Reports
        st.subheader(f"📊 Detailed Classification Reports - {year}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Random Forest**")
            rf_report = classification_report(rf_y_test, rf_preds, output_dict=True, zero_division=0)
            rf_report_df = pd.DataFrame(rf_report).transpose()
            st.dataframe(rf_report_df, use_container_width=True)
        
        with col2:
            st.markdown("**XGBoost**")
            xgb_report = classification_report(xgb_y_test, xgb_preds, output_dict=True, zero_division=0)
            xgb_report_df = pd.DataFrame(xgb_report).transpose()
            st.dataframe(xgb_report_df, use_container_width=True)
        
        # Sample Predictions Comparison
        st.subheader(f"Sample Predictions Comparison - {year}")
        
        # Merge predictions to show side by side
        comparison_all = rf_filtered.merge(
            xgb_filtered,
            on=['year', 'round', 'win_rate', 'avg_points', 'actual'],
            suffixes=('_rf', '_xgb')
        )
        
        # Show a diverse sample: actual wins first, then other predictions
        wins = comparison_all[comparison_all['actual'] == 1].head(10)
        no_wins = comparison_all[comparison_all['actual'] == 0].head(10)
        comparison_samples = pd.concat([wins, no_wins]).head(20)
        
        comparison_samples_display = comparison_samples[['year', 'round', 'win_rate', 'avg_points', 'actual', 'predicted_rf', 'predicted_xgb']].copy()
        comparison_samples_display.columns = ['Year', 'Round', 'Win Rate', 'Avg Points', 'Actual Win', 'RF Prediction', 'XGB Prediction']
        
        # Convert to int for cleaner display
        comparison_samples_display['Actual Win'] = comparison_samples_display['Actual Win'].astype(int)
        comparison_samples_display['RF Prediction'] = comparison_samples_display['RF Prediction'].astype(int)
        comparison_samples_display['XGB Prediction'] = comparison_samples_display['XGB Prediction'].astype(int)
        
        st.dataframe(comparison_samples_display, use_container_width=True)
        
        # Show statistics about the sample
        total_in_sample = len(comparison_samples)
        wins_in_sample = (comparison_samples['actual'] == 1).sum()
        rf_correct = (comparison_samples['predicted_rf'] == comparison_samples['actual']).sum()
        xgb_correct = (comparison_samples['predicted_xgb'] == comparison_samples['actual']).sum()
        
        st.caption(f"📊 Sample shows {wins_in_sample} actual wins out of {total_in_sample} predictions. RF got {rf_correct} correct, XGB got {xgb_correct} correct.")
        
        # Agreement statistics
        merged_all = comparison_all
        
        agree = (merged_all['predicted_rf'] == merged_all['predicted_xgb']).sum()
        total_merged = len(merged_all)
        st.info(f"**Models Agreement**: Both models agree on {agree}/{total_merged} predictions ({agree/total_merged*100:.1f}%)")
        
    else:
        st.info(f"No predictions available for {year} from both models")
elif not df_preds.empty:
    # Fallback to old single model display
    st.info("Showing legacy predictions. Run train_comparison.py to see both models.")
    pred_filtered = df_preds[df_preds["year"] == year].copy()
    
    if len(pred_filtered) > 0:
        y_test = pred_filtered['actual'].values
        preds = pred_filtered['predicted'].values
        acc = accuracy_score(y_test, preds)
        
        st.metric("Accuracy", f"{acc:.1%}")
        st.dataframe(pred_filtered[['year', 'round', 'win_rate', 'avg_points', 'actual', 'predicted']].head(20), use_container_width=True)
else:
    st.info("No ML predictions available yet. Run the ML training service first.")

# ==================== MODEL COMPARISON SECTION ====================
st.divider()
st.header("🤖 Model Comparison: Random Forest vs XGBoost")

# Load model comparison data
@st.cache_data(ttl=60)
def load_model_comparison():
    try:
        return pd.read_sql("SELECT * FROM f1_model_comparison", engine)
    except:
        return pd.DataFrame()

@st.cache_data(ttl=60)
def load_rf_predictions():
    try:
        return pd.read_sql("SELECT * FROM f1_predictions_rf", engine)
    except:
        return pd.DataFrame()

@st.cache_data(ttl=60)
def load_xgb_predictions():
    try:
        return pd.read_sql("SELECT * FROM f1_predictions_xgb", engine)
    except:
        return pd.DataFrame()

comparison_df = load_model_comparison()
rf_preds_df = load_rf_predictions()
xgb_preds_df = load_xgb_predictions()

if not comparison_df.empty:
    st.subheader("📊 Overall Model Performance")
    
    # Display comparison metrics
    col1, col2, col3 = st.columns(3)
    
    rf_data = comparison_df[comparison_df['model'] == 'RandomForest'].iloc[0]
    xgb_data = comparison_df[comparison_df['model'] == 'XGBoost'].iloc[0]
    
    with col1:
        st.metric("Random Forest Accuracy", f"{rf_data['accuracy']:.2%}")
        st.metric("Random Forest ROC-AUC", f"{rf_data['roc_auc']:.3f}")
    
    with col2:
        st.metric("XGBoost Accuracy", f"{xgb_data['accuracy']:.2%}")
        st.metric("XGBoost ROC-AUC", f"{xgb_data['roc_auc']:.3f}")
    
    with col3:
        # Determine winner
        best_model = "Random Forest" if rf_data['accuracy'] > xgb_data['accuracy'] else "XGBoost"
        acc_diff = abs(rf_data['accuracy'] - xgb_data['accuracy']) * 100
        st.metric("Best Model", best_model)
        st.metric("Accuracy Difference", f"{acc_diff:.2f}%")
    
    # Bar chart comparison
    st.subheader("📈 Performance Comparison")
    
    comp_melted = comparison_df.melt(
        id_vars=['model'], 
        value_vars=['accuracy', 'roc_auc'],
        var_name='Metric',
        value_name='Score'
    )
    
    fig_comp = px.bar(
        comp_melted,
        x='Metric',
        y='Score',
        color='model',
        barmode='group',
        title='Model Performance Comparison',
        labels={'Score': 'Score', 'Metric': 'Metric', 'model': 'Model'},
        color_discrete_map={'RandomForest': '#2ecc71', 'XGBoost': '#e74c3c'}
    )
    st.plotly_chart(fig_comp, use_container_width=True, key="model_comp_bar")
    
    # Year-by-year comparison
    if not rf_preds_df.empty and not xgb_preds_df.empty:
        st.subheader("📅 Year-by-Year Comparison")
        
        # Calculate accuracy by year for both models
        rf_by_year = rf_preds_df.groupby('year').apply(
            lambda x: accuracy_score(x['actual'], x['predicted'])
        ).reset_index()
        rf_by_year.columns = ['year', 'accuracy']
        rf_by_year['model'] = 'Random Forest'
        
        xgb_by_year = xgb_preds_df.groupby('year').apply(
            lambda x: accuracy_score(x['actual'], x['predicted'])
        ).reset_index()
        xgb_by_year.columns = ['year', 'accuracy']
        xgb_by_year['model'] = 'XGBoost'
        
        yearly_comp = pd.concat([rf_by_year, xgb_by_year])
        
        fig_yearly = px.line(
            yearly_comp,
            x='year',
            y='accuracy',
            color='model',
            title='Model Accuracy Over Years',
            labels={'year': 'Year', 'accuracy': 'Accuracy', 'model': 'Model'},
            markers=True,
            color_discrete_map={'Random Forest': '#2ecc71', 'XGBoost': '#e74c3c'}
        )
        st.plotly_chart(fig_yearly, use_container_width=True, key="yearly_comp_line")
        
        # Detailed comparison for selected year
        st.subheader(f"🔍 Detailed Comparison for {year}")
        
        rf_year = rf_preds_df[rf_preds_df['year'] == year]
        xgb_year = xgb_preds_df[xgb_preds_df['year'] == year]
        
        if len(rf_year) > 0 and len(xgb_year) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Random Forest**")
                rf_acc = accuracy_score(rf_year['actual'], rf_year['predicted'])
                st.metric("Accuracy", f"{rf_acc:.2%}")
                
                # Confusion matrix
                rf_cm = confusion_matrix(rf_year['actual'], rf_year['predicted'])
                fig_rf_cm = px.imshow(
                    rf_cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['No Win', 'Win'],
                    y=['No Win', 'Win'],
                    title=f"Random Forest Confusion Matrix ({year})",
                    color_continuous_scale='Greens',
                    text_auto=True
                )
                st.plotly_chart(fig_rf_cm, use_container_width=True, key="rf_cm_comparison")
            
            with col2:
                st.write("**XGBoost**")
                xgb_acc = accuracy_score(xgb_year['actual'], xgb_year['predicted'])
                st.metric("Accuracy", f"{xgb_acc:.2%}")
                
                # Confusion matrix
                xgb_cm = confusion_matrix(xgb_year['actual'], xgb_year['predicted'])
                fig_xgb_cm = px.imshow(
                    xgb_cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['No Win', 'Win'],
                    y=['No Win', 'Win'],
                    title=f"XGBoost Confusion Matrix ({year})",
                    color_continuous_scale='Reds',
                    text_auto=True
                )
                st.plotly_chart(fig_xgb_cm, use_container_width=True, key="xgb_cm_comparison")
            
            # Agreement analysis
            st.subheader("🤝 Model Agreement Analysis")
            
            # Merge predictions on year and round
            merged = rf_year.merge(
                xgb_year, 
                on=['year', 'round', 'win_rate', 'avg_points', 'actual'],
                suffixes=('_rf', '_xgb')
            )
            
            # Calculate agreement
            merged['both_correct'] = (merged['predicted_rf'] == merged['actual']) & (merged['predicted_xgb'] == merged['actual'])
            merged['both_wrong'] = (merged['predicted_rf'] != merged['actual']) & (merged['predicted_xgb'] != merged['actual'])
            merged['rf_correct_only'] = (merged['predicted_rf'] == merged['actual']) & (merged['predicted_xgb'] != merged['actual'])
            merged['xgb_correct_only'] = (merged['predicted_xgb'] == merged['actual']) & (merged['predicted_rf'] != merged['actual'])
            
            agreement_stats = pd.DataFrame({
                'Category': [
                    'Both Models Correct',
                    'Both Models Wrong',
                    'Only Random Forest Correct',
                    'Only XGBoost Correct'
                ],
                'Count': [
                    merged['both_correct'].sum(),
                    merged['both_wrong'].sum(),
                    merged['rf_correct_only'].sum(),
                    merged['xgb_correct_only'].sum()
                ]
            })
            
            fig_agreement = px.pie(
                agreement_stats,
                values='Count',
                names='Category',
                title=f'Model Agreement Analysis ({year})',
                color_discrete_sequence=['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
            )
            st.plotly_chart(fig_agreement, use_container_width=True, key="agreement_pie")
            
            # Show agreement percentage
            total = len(merged)
            agree = (merged['predicted_rf'] == merged['predicted_xgb']).sum()
            st.info(f"Models agree on {agree}/{total} predictions ({agree/total*100:.1f}%)")
        else:
            st.warning(f"No predictions available for {year} from both models")
else:
    st.info("No model comparison data available. Run the ML training with train_comparison.py first.")
