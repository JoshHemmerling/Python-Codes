import os
import requests
import numpy as np
from scipy.stats import norm

# Function to get forecasted high temperature from the API
def get_forecasted_high(location, date):
    API_KEY = os.getenv("API_KEY", "default_api_key")  # Use environment variable for API key
    BASE_URL = "https://api.weatherapi.com/v1/forecast.json"
    
    params = {
        "key": API_KEY,
        "q": location,
        "dt": date
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        data = response.json()
        forecasted_high = data["forecast"]["forecastday"][0]["day"]["maxtemp_f"]
        return forecasted_high
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")
    except KeyError as key_err:
        print(f"Unexpected response format: missing key {key_err}")
    except Exception as err:
        print(f"An error occurred: {err}")
    return None

# Function to simulate historical temperature data
def simulate_historical_temperatures(mean, std_dev, num_samples=1000):
    return np.random.normal(mean, std_dev, num_samples)

# Function to calculate the probability of temperature being over/under a threshold
def calculate_probabilities(forecasted_high, historical_data):
    mean_temp = np.mean(historical_data)
    std_temp = np.std(historical_data)
    
    # Calculate z-score for the forecasted temperature
    z_score = (forecasted_high - mean_temp) / std_temp
    
    # Calculate probabilities for over and under
    prob_over = 1 - norm.cdf(z_score)
    prob_under = norm.cdf(z_score)
    
    return prob_over, prob_under

# Function to calculate odds from probabilities
def calculate_odds(probability):
    return 1 / probability if probability > 0 else float('inf')

# Function to calculate implied probabilities
def calculate_implied_probability(odds):
    return 1 / odds if odds != 0 else 0

# Function to print betting statement
def print_betting_statement(bet_choice, forecasted_high, historical_average_high):
    if bet_choice == "over":
        print(f"You are betting that the forecasted high temperature of {forecasted_high}°F will be OVER the historical average of {historical_average_high}°F.")
    elif bet_choice == "under":
        print(f"You are betting that the forecasted high temperature of {forecasted_high}°F will be UNDER the historical average of {historical_average_high}°F.")
    else:
        print("Invalid bet choice. Please choose 'over' or 'under'.")

# Function to calculate potential payout
def calculate_potential_payout(bet_amount, payout_multiplier):
    try:
        return bet_amount * payout_multiplier
    except Exception as err:
        print(f"An error occurred while calculating payout: {err}")
        return 0

# Main function to use the API for forecasted high and internal algorithm for betting odds
def betting_logic_combined():
    try:
        historical_average_high = 79.5  # in degrees Fahrenheit
        std_dev = 5.0  # Assumed standard deviation for the historical data

        location = "Buffalo, NY"
        date = "2024-07-30"
        
        # Get forecasted high temperature from the API
        forecasted_high = get_forecasted_high(location, date)
        if forecasted_high is None:
            print("Failed to retrieve forecasted high temperature. Please try again later.")
            return

        print(f"Forecasted high temperature for {location} on {date} is {forecasted_high}°F.")

        bet_choice = input("Do you bet on 'over' or 'under'? ").strip().lower()
        bet_amount = float(input("Enter the amount of money you are betting: "))

        # Validate user inputs
        if bet_choice not in ["over", "under"]:
            print("Invalid bet choice. Please enter 'over' or 'under'.")
            return
        
        if bet_amount <= 0:
            print("Bet amount must be greater than zero.")
            return

        # Simulate historical data
        historical_data = simulate_historical_temperatures(historical_average_high, std_dev)
        
        # Calculate probabilities for over and under
        prob_over, prob_under = calculate_probabilities(forecasted_high, historical_data)
        
        # Calculate odds based on probabilities
        odds_over = calculate_odds(prob_over)
        odds_under = calculate_odds(prob_under)
        
        # Print betting statement
        print_betting_statement(bet_choice, forecasted_high, historical_average_high)
        
        # Determine the user's choice and calculate the potential payout
        if bet_choice == "over":
            payout_multiplier = odds_over
        elif bet_choice == "under":
            payout_multiplier = odds_under
        
        implied_probability = calculate_implied_probability(payout_multiplier)
        potential_payout = calculate_potential_payout(bet_amount, payout_multiplier)
        
        # Output the prediction and odds details
        print(f"The forecasted temperature of {forecasted_high}°F is expected to be {'over' if forecasted_high > historical_average_high else 'under'} the historical average of {historical_average_high}°F.")
        print(f"Odds for betting on {bet_choice}: {payout_multiplier}")
        print(f"Implied probability of this outcome: {implied_probability:.2%}")
        print(f"If you bet ${bet_amount} on {bet_choice}, the potential payout is ${potential_payout:.2f}.")
    except ValueError as val_err:
        print(f"Input error: {val_err}")
    except Exception as err:
        print(f"An unexpected error occurred: {err}")

if __name__ == "__main__":
    betting_logic_combined()
