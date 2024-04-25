import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title('Soccer Game Goal Simulation')

# Input sliders for the probabilities
prob_team1 = st.sidebar.number_input('Probability of Team 1 scoring a goal per minute', min_value=0.0, value=0.0)
prob_team2 = st.sidebar.number_input('Probability of Team 2 scoring a goal per minute', min_value=0.0, value=0.0)

# Input for number of simulations
num_simulations = st.sidebar.number_input('Number of Simulations', min_value=1, value=1000)

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

# Visualize the distribution of goals
st.write('Goal Distribution for Team 1')
sns.histplot(df_results['Team 1 Goals'], kde=True)
st.pyplot(plt)

st.write('Goal Distribution for Team 2')
sns.histplot(df_results['Team 2 Goals'], kde=True)
st.pyplot(plt)


