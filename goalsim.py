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
    def main():
        st.title('Football Team Statistics')
    
        # Dropdown to select a team
        team_list = list(set(data['Home Team']).union(set(data['Away Team'])))
        selected_team = st.selectbox('Select a team', team_list)
    
        # Dropdown to select result type
        result_options = ['All', 'W', 'L', 'D']
        selected_result = st.selectbox('Select result type', result_options)
    
        # Filter data for selected team
        if selected_result == 'All':
            home_data = data[data['Home Team'] == selected_team]
            away_data = data[data['Away Team'] == selected_team]
        else:
            home_data = data[(data['Home Team'] == selected_team) & (data['Home Result'] == selected_result)]
            away_data = data[(data['Away Team'] == selected_team) & (data['Away Result'] == selected_result)]
    
        # Calculate averages
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
            st.write("No matches found for the selected filters.")

        # Prepare data for model training
        X = home_data[['Match Number']].values
        y = home_data['Home Goals'].values
        model = LinearRegression()
        model.fit(X, y)
        predicted_home_goals = model.predict([[total_matches + 1]])  # Predict next match home goals

        X_away = away_data[['Match Number']].values
        y_away = away_data['Away Goals'].values
        model_away = LinearRegression()
        model_away.fit(X_away, y_away)
        predicted_away_goals = model_away.predict([[total_matches + 1]])  # Predict next match away goals

        st.write(f"Predicted Home Goals in next match: {predicted_home_goals[0]:.2f}")
        st.write(f"Predicted Away Goals in next match: {predicted_away_goals[0]:.2f}")
    
        # Plotting the count of goals per match number
        goals_per_match = pd.concat([home_data.assign(Goals=home_data['Home Goals']),
                                     away_data.assign(Goals=away_data['Away Goals'])])
    
        fig, ax = plt.subplots()
        goals_per_match.groupby('Match Number')['Goals'].sum().plot(kind='bar', ax=ax)
        ax.set_title('Count of Goals per Match Number')
        ax.set_ylabel('Total Goals')
        st.pyplot(fig)
    
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

    
    


