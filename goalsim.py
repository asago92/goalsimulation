import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import poisson

# Title for the Streamlit App
st.title('Football Goal Predictions')

# Load the data
data = pd.read_csv('epl_streamlit.csv')

# Create tabs for different analyses
tab1, tab2, tab3 = st.tabs(["Linear Regression", "Simulation", "Poisson Distribution"])

with tab1:
    # Mapping the results to numeric codes
    result_mapping = {'W': 2, 'D': 1, 'L': 0}
    data['Home Result Code'] = data['Home Result'].map(result_mapping)
    data['Away Result Code'] = data['Away Result'].map(result_mapping)

    def main():
        st.subheader('Goal Prediction using Linear Regression')
        st.write("""
        This model predicts the number of goals using a linear regression approach based on past performance.
        """)
    
        # Dropdowns to select home and away teams
        team_list = sorted(list(set(data['Home Team']).union(set(data['Away Team']))))
        cols = st.columns(2)
        with cols[0]:
            selected_home_team = st.selectbox('Select home team', team_list)
        with cols[1]:
            selected_away_team = st.selectbox('Select away team', team_list)
    
        # Filter data for selected teams
        home_data = data[data['Home Team'] == selected_home_team]
        away_data = data[data['Away Team'] == selected_away_team]
    
        # Prepare data for model training
        if not home_data.empty and not away_data.empty:
            home_goals = home_data['Home Goals'].values
            home_features = home_data[['Match', 'Home Result Code']]
            home_features['Opponent Avg Conceded'] = away_data['Away Conceded'].mean()
    
            away_goals = away_data['Away Goals'].values
            away_features = away_data[['Match', 'Away Result Code']]
            away_features['Opponent Avg Conceded'] = home_data['Home Conceded'].mean()
    
            # Fit models
            model_home = LinearRegression()
            model_home.fit(home_features, home_goals)
            predicted_home_goals = model_home.predict([[len(home_data) + 1, result_mapping['W'], home_features['Opponent Avg Conceded'].mean()]])
    
            model_away = LinearRegression()
            model_away.fit(away_features, away_goals)
            predicted_away_goals = model_away.predict([[len(away_data) + 1, result_mapping['W'], away_features['Opponent Avg Conceded'].mean()]])
    
            # Display predictions
            cols = st.columns(2)
            with cols[0]:
                st.metric(label="Predicted Home Goals", value=f"{predicted_home_goals[0]:.2f}")
            with cols[1]:
                st.metric(label="Predicted Away Goals", value=f"{predicted_away_goals[0]:.2f}")

            st.markdown("###")
            with st.expander("Goal Statistics"):
                total_matches = len(home_data) + len(away_data)
                if total_matches > 0:
                    avg_goals_per_match = (home_data['Home Goals'].sum() + away_data['Away Goals'].sum()) / total_matches
                    avg_goals_conceded = (home_data['Home Conceded'].sum() + away_data['Away Conceded'].sum()) / total_matches
                    avg_home_goals = home_data['Home Goals'].mean()
                    avg_away_goals = away_data['Away Goals'].mean()

                    # Display statistics
                    st.write(f"Average Goals per Match: {avg_goals_per_match:.2f}")
                    st.write(f"Average Goals Conceded per Match: {avg_goals_conceded:.2f}")
                    st.write(f"Average Home Goals: {avg_home_goals:.2f}")
                    st.write(f"Average Away Goals: {avg_away_goals:.2f}")
        else:
            st.write("Insufficient data available for predictions.")

    if __name__ == "__main__":
        main()

with tab2:
    st.subheader('Goal Simulation')
    st.write("""
    Simulate the number of goals based on a user-defined scoring probability.
    """)
    
    # Input sliders for probabilities and number of simulations
    cols = st.columns(3)
    with cols[0]:
        prob_team1 = st.number_input('Home Team Score Prob', min_value=0.00, value=0.05, step=0.01)
    with cols[1]:
        prob_team2 = st.number_input('Away Team Score Prob', min_value=0.00, value=0.05, step=0.01)
    with cols[2]:
        num_simulations = st.number_input('Number of Simulations', min_value=1, value=10000)
    
    st.markdown("---")
    
    def simulate_goals(prob, minutes=90):
        return np.random.binomial(minutes, prob)
    
    def run_simulations(prob1, prob2, num_simulations):
        results = []
        for _ in range(num_simulations):
            goals_team1 = simulate_goals(prob1)
            goals_team2 = simulate_goals(prob2)
            results.append((goals_team1, goals_team2))
        return results
    
    # Run simulations
    results = run_simulations(prob_team1, prob_team2, num_simulations)
    
    # Create a DataFrame to display results
    df_results = pd.DataFrame(results, columns=['Home Goals', 'Away Goals'])
    
    # Calculate win statistics
    team1_wins = np.sum(df_results['Home Goals'] > df_results['Away Goals'])
    team2_wins = np.sum(df_results['Away Goals'] > df_results['Home Goals'])
    draws = np.sum(df_results['Home Goals'] == df_results['Away Goals'])
    
    # Calculate winning percentages
    total_games = len(df_results)
    team1_win_percentage = (team1_wins / total_games) * 100
    team2_win_percentage = (team2_wins / total_games) * 100
    draw_percentage = (draws / total_games) * 100
    
    # Display results
    st.subheader('Winning Statistics:')
    cols = st.columns(3)
    with cols[0]:
        st.metric(label="Home Team Win", value=f"{team1_win_percentage:.2f}%")
    with cols[1]:
        st.metric(label="Away Team Win", value=f"{team2_win_percentage:.2f}%")
    with cols[2]:
        st.metric(label="Draw", value=f"{draw_percentage:.2f}%")
    
    st.markdown("---")
    
    # Display average goals
    average_goals_team1 = df_results['Home Goals'].mean()
    average_goals_team2 = df_results['Away Goals'].mean()

    st.subheader('Predicted Average Goals:')
    cols = st.columns(2)
    with cols[0]:
        st.metric(label="Home Team Avg Goals", value=f"{average_goals_team1:.2f}")
    with cols[1]:
        st.metric(label="Away Team Avg Goals", value=f"{average_goals_team2:.2f}")

    st.markdown("###")

    with st.expander("Simulation Results"):
        st.dataframe(df_results)

with tab3:
    st.subheader("Goal Prediction using Poisson Distribution")
    st.write("""
    This tab uses the Poisson distribution to predict the number of goals based on average goals scored and conceded.
    """)
    
    # Calculate average goals scored and conceded
    team_stats = data.groupby('Home Team').agg({
        'Home Goals': 'mean',
        'Away Goals': 'mean'
    }).rename(columns={'Home Goals': 'home_goals_avg', 'Away Goals': 'away_goals_avg'})
    
    team_stats['away_goals_avg'] = data.groupby('Away Team')['Away Goals'].mean()
    team_stats['home_goals_conceded_avg'] = data.groupby('Home Team')['Away Goals'].mean()
    team_stats['away_goals_conceded_avg'] = data.groupby('Away Team')['Home Goals'].mean()
    
    team_stats = team_stats.fillna(0)
    
    st.header("Input Match Details")
    home_team = st.selectbox('Home Team', team_stats.index)
    away_team = st.selectbox('Away Team', team_stats.index)
    
    if st.button("Predict"):
        home_goals_avg = team_stats.loc[home_team, 'home_goals_avg']
        away_goals_avg = team_stats.loc[away_team, 'away_goals_avg']
    
        home_goals_conceded_avg = team_stats.loc[away_team, 'away_goals_conceded_avg']
        away_goals_conceded_avg = team_stats.loc[home_team, 'home_goals_conceded_avg']
    
        # Expected goals for each team
        home_goals_expected = (home_goals_avg + away_goals_conceded_avg) / 2
        away_goals_expected = (away_goals_avg + home_goals_conceded_avg) / 2
    
        st.write(f"Expected goals for {home_team}: {home_goals_expected:.2f}")
        st.write(f"Expected goals for {away_team}: {away_goals_expected:.2f}")
    
        # Predict the distribution of goals
        home_goal_prob = [poisson.pmf(i, home_goals_expected) for i in range(6)]
        away_goal_prob = [poisson.pmf(i, away_goals_expected) for i in range(6)]
        
        st.bar_chart(pd.DataFrame({
            f"{'Home Team'} Goal Probability": home_goal_prob,
            f"{'Away Team'} Goal Probability": away_goal_prob
        }, index=list(range(6))))






    
    


