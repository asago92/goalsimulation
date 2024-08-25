import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import poisson

st.title('Goal Predictions')

# Load the data
data = pd.read_csv('epl_streamlit_13May.csv')

tab1, tab2, tab3 = st.tabs(["Linear Regression", "Simulation", "Poisson Distribution"])
with tab1:        
    # Encode the results
    result_mapping = {'W': 2, 'D': 1, 'L': 0}
    data['Home Result Code'] = data['Home Result'].map(result_mapping)
    data['Away Result Code'] = data['Away Result'].map(result_mapping)
    
    def main():
        st.subheader('Goal Prediction')
        st.write("""
        The linear regression model looks at all previous matches (how many goals were scored and conceded, and whether the matches were won, lost, or drawn).
        It uses this information to find patterns or trends. Based on these trends, it makes a prediction for the next match.
        """)
        st.markdown("###")
    
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
            #st.metric(label="Predicted Home Goals", value=f"{predicted_home_goals[0]:.2f}")
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
            #st.metric(label="Predicted Away Goals", value=f"{predicted_away_goals[0]:.2f}")
        else:
            st.write("No away data available for predictions.")

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
    
    if __name__ == "__main__":
        main()

with tab2:
    st.subheader('Goal Simulation')
    st.write("""
    This simulation predicts how many goals two soccer teams might score in a 90-minute game based on a calculated probablity of scoring against the strength of their opponent.
    """)
    st.markdown("###")
    # Input sliders for the probabilities
    cols = st.columns(3)
    with cols[0]:   
        prob_team1 = st.number_input('Home Team Score Prob', min_value=0.00, value=0.0)
    with cols[1]:
        prob_team2 = st.number_input('Away Team Score Prob', min_value=0.00, value=0.0)
    with cols[2]:
        num_simulations = st.number_input('Number of Simulations', min_value=1, value=10000)

    st.markdown("---")
        
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
    st.subheader('Winning Statistics:')
    cols = st.columns(3)
    with cols[0]:
        st.metric(label="Home Team Win", value=f"{team1_win_percentage:.2f}%")
    with cols[1]:
        st.metric(label="Away Team Win", value=f"{team2_win_percentage:.2f}%")
    with cols[2]:
        st.metric(label="Draw", value=f"{draw_percentage:.2f}%")
    st.markdown("---")
    # Calculate average goals
    average_goals_team1 = df_results['Team 1 Goals'].mean()
    average_goals_team2 = df_results['Team 2 Goals'].mean()

    # Display the predicted average goals in Streamlit
    st.subheader('Predicted Average Goals:')
    cols = st.columns(2)
    with cols[0]:
        st.metric(label="Home Team Avg Goals", value=f"{average_goals_team1:.2f}")
    with cols[1]:
        st.metric(label="Away Team Avg Goals", value=f"{average_goals_team2:.2f}")

    st.markdown("###")

    # Create a DataFrame to display results
    with st.expander("Simulation Results"):
        st.dataframe(df_results)

with tab3:
    data = pd.read_csv('epl_streamlit_13May.csv')
    # Calculate average goals scored and conceded
    team_stats = data.groupby('Home Team').agg({
        'Home Goals': 'mean',
        'Away Goals': 'mean'
    }).rename(columns={'Home Goals': 'home_goals_avg', 'Away Goals': 'away_goals_avg'})
    
    team_stats['away_goals_avg'] = data.groupby('Away Team')['Away Goals'].mean()
    team_stats['home_goals_conceded_avg'] = data.groupby('Home Team')['Away Goals'].mean()
    team_stats['away_goals_conceded_avg'] = data.groupby('Away Team')['Home Goals'].mean()
    
    team_stats = team_stats.fillna(0)
    st.title("Soccer Goal Prediction using Poisson Distribution")

    st.header("Input Match Details")
    home_team = st.selectbox("Home Team", team_stats.index)
    away_team = st.selectbox("Away Team", team_stats.index)
    
    if st.button("Predict"):
        home_goals_avg = team_stats.loc["Home Team", 'home_goals_avg']
        away_goals_avg = team_stats.loc["Away Team", 'away_goals_avg']
    
        home_goals_conceded_avg = team_stats.loc["Away Team", 'away_goals_conceded_avg']
        away_goals_conceded_avg = team_stats.loc["Home Team", 'home_goals_conceded_avg']
    
        # Expected goals for each team
        home_goals_expected = (home_goals_avg + away_goals_conceded_avg) / 2
        away_goals_expected = (away_goals_avg + home_goals_conceded_avg) / 2
    
        st.write(f"Expected goals for {"Home Team"}: {home_goals_expected:.2f}")
        st.write(f"Expected goals for {"Away Team"}: {away_goals_expected:.2f}")
    
        # Predict the distribution of goals
        home_goal_prob = [poisson.pmf(i, home_goals_expected) for i in range(6)]
        away_goal_prob = [poisson.pmf(i, away_goals_expected) for i in range(6)]
    
        st.bar_chart(pd.DataFrame({
            f'{Home Team} Goal Probability': home_goal_prob,
            f'{Away Team} Goal Probability': away_goal_prob
        }, index=list(range(6))))





    
    


