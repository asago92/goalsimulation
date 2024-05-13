import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.title('Soccer Game Goal Simulation')

# Load the data
data = pd.read_csv('epl_streamlit_13May.csv')

# Input sliders for the probabilities
prob_team1 = st.sidebar.number_input('Probability of Team 1 scoring a goal per minute', min_value=0.0, value=0.0)
prob_team2 = st.sidebar.number_input('Probability of Team 2 scoring a goal per minute', min_value=0.0, value=0.0)

# Input for number of simulations
num_simulations = st.sidebar.number_input('Number of Simulations', min_value=1, value=2000)

tab1, tab2 = st.tabs(["Analysis", "Simulation"])
with tab1:        
    # Encode the results
    result_mapping = {'W': 2, 'D': 1, 'L': 0}
    data['Home Result Code'] = data['Home Result'].map(result_mapping)
    data['Away Result Code'] = data['Away Result'].map(result_mapping)
    
    def main():
        st.title('Football Match Score Prediction')
    
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
        # Home model
        if not home_data.empty:
            home_goals = home_data['Home Goals'].values
            home_features = home_data[['Match Number', 'Home Result Code']]
            home_conceded_by_opponent = data[data['Home Team'] == selected_away_team]['Away Conceded'].mean()
            home_features['Opponent Avg Conceded'] = home_conceded_by_opponent
    
            # Fit model for home team
            model_home = LinearRegression()
            model_home.fit(home_features, home_goals)
            predicted_home_goals = model_home.predict([[len(home_data) + 1, result_mapping['W'], home_conceded_by_opponent]])  # Example: Next match assumed win
            st.metric(label="Predicted Home Goals", value=f"{predicted_home_goals[0]:.2f}")
        else:
            st.write("No home data available for predictions.")
    
        # Away model
        if not away_data.empty:
            away_goals = away_data['Away Goals'].values
            away_features = away_data[['Match Number', 'Away Result Code']]
            away_conceded_by_opponent = data[data['Away Team'] == selected_home_team]['Home Conceded'].mean()
            away_features['Opponent Avg Conceded'] = away_conceded_by_opponent
    
            # Fit model for away team
            model_away = LinearRegression()
            model_away.fit(away_features, away_goals)
            predicted_away_goals = model_away.predict([[len(away_data) + 1, result_mapping['W'], away_conceded_by_opponent]])  # Example: Next match assumed win
            st.write(f"Predicted Away Goals: {predicted_away_goals[0]:.2f}")
        else:
            st.write("No away data available for predictions.")
    
    if __name__ == "__main__":
        main()

with tab2:
    def simulate_goals(prob, minutes=90):
        # Simulates goals for one team in a single game
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
    df_results = pd.DataFrame(results, columns=['Team 1 Goals', 'Team 2 Goals'])
    
    # Calculate win statistics
    team1_wins = np.sum(df_results['Team 1 Goals'] > df_results['Team 2 Goals'])
    team2_wins = np.sum(df_results['Team 2 Goals'] > df_results['Team 1 Goals'])
    draws = np.sum(df_results['Team 1 Goals'] == df_results['Team 2 Goals'])
    
    # Calculate winning percentages
    total_games = len(df_results)
    team1_win_percentage = (team1_wins / total_games) * 100
    team2_win_percentage = (team2_wins / total_games) * 100
    draw_percentage = (draws / total_games) * 100
    
    # Display results
    st.write('Simulation Results:')
    st.dataframe(df_results)
    st.write('Winning Statistics:')
    st.write(f"Team 1 Win Percentage: {team1_win_percentage:.2f}%")
    st.write(f"Team 2 Win Percentage: {team2_win_percentage:.2f}%")
    st.write(f"Draw Percentage: {draw_percentage:.2f}%")

    # Calculate average goals
    average_goals_team1 = df_results['Team 1 Goals'].mean()
    average_goals_team2 = df_results['Team 2 Goals'].mean()

    # Display the predicted average goals in Streamlit
    st.write('Predicted Average Goals:')
    st.write(f"Average Goals Scored by Team 1: {average_goals_team1:.2f}")
    st.write(f"Average Goals Scored by Team 2: {average_goals_team2:.2f}")

    
    


