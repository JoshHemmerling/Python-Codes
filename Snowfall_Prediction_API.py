import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import requests

# Tomorrow.io API key and base URL
API_KEY = "2A7Hfxm52TymGFYxXJ4Z956KJJrYkopy"
BASE_URL = "https://api.tomorrow.io/v4/weather/forecast"

# Towns in Western New York with their coordinates
TOWNS = {
    "Buffalo": (42.8864, -78.8784),
    "Orchard Park": (42.7675, -78.7431),
    "Boston": (42.6256, -78.7375),
    "Colden": (42.6387, -78.6889),
    "Concord": (42.4928, -78.5372),
    "Ellicottville": (42.2759, -78.6692),
    "Springville": (42.5084, -78.6642),
    "East Aurora": (42.767, -78.6136),
    "Williamsville": (42.9634, -78.735),
    "Clarence": (42.9776, -78.5778),
    "Amherst": (42.9784, -78.7998),
    "Cheektowaga": (42.9025, -78.7446),
    "Lancaster": (42.9006, -78.6703),
    "Tonawanda": (43.0203, -78.8803),
    "Kenmore": (42.9656, -78.8717),
    "Grand Island": (43.0205, -78.9612)
}

# Function to fetch weather data from Tomorrow.io API
def fetch_weather_data(lat, lon):
    url = f"{BASE_URL}?location={lat},{lon}&apikey={API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        # Extract relevant details from API response
        temperature_c = data['timelines']['hourly'][0]['values'].get('temperature', None)
        humidity = data['timelines']['hourly'][0]['values'].get('humidity', None)
        wind_direction = data['timelines']['hourly'][0]['values'].get('windDirection', None)
        wind_speed = data['timelines']['hourly'][0]['values'].get('windSpeed', None)

        # Convert temperature to Fahrenheit
        if temperature_c is not None:
            temperature_f = (temperature_c * 9/5) + 32
        else:
            temperature_f = None

        # Handle cases where wind direction or speed might not be available
        wind_direction = wind_direction if wind_direction is not None else "N/A"
        wind_speed = wind_speed if wind_speed is not None else "N/A"

        return round(temperature_f, 1) if temperature_f is not None else None, humidity, wind_direction, wind_speed
    else:
        return None, None, None, None

# Function to calculate SLR dynamically based on season, temperature, and humidity
def calculate_slr(temp_f, humidity, season):
    if season == "Beginning Season":
        if temp_f >= 32:
            return 5  # Warmer temperatures, higher liquid content
        elif temp_f >= 30:
            return 10
        elif temp_f >= 18:
            return 12
        else:
            return 15
    elif season == "Average Season":
        if temp_f >= 32:
            return 5
        elif temp_f >= 30:
            return 10
        elif temp_f >= 18:
            return 13
        else:
            return 18
    elif season == "Late Season":
        if temp_f >= 32:
            return 5
        elif temp_f >= 30:
            return 10
        elif temp_f >= 18:
            return 15
        else:
            return 20
    return 10  # Default fallback

# Function to calculate snowfall
def calculate_snowfall(temp_f, humidity, precip_in, season):
    slr = calculate_slr(temp_f, humidity, season)
    return precip_in * slr

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Snowfall Prediction Dashboard"

# Layout of the Dash app
app.layout = html.Div([
    html.H1("Snowfall Prediction Dashboard", style={"textAlign": "center"}),

    html.Div([
        html.Label("Select Town:"),
        dcc.Dropdown(
            id="town-selector",
            options=[{"label": town, "value": town} for town in TOWNS.keys()],
            value="Buffalo",
        ),

        html.Label("Select Season:"),
        dcc.RadioItems(
            id="season-selector",
            options=[
                {"label": "Beginning Season", "value": "Beginning Season"},
                {"label": "Average Season", "value": "Average Season"},
                {"label": "Late Season", "value": "Late Season"},
            ],
            value="Average Season",
            inline=True
        ),

        html.Label("Adjust Humidity (%):"),
        dcc.Slider(
            id="humidity-slider",
            min=0,
            max=100,
            step=1,
            value=80,
            marks={i: f"{i}%" for i in range(0, 101, 10)},
        ),

        html.Label("Adjust Temperature (°F):"),
        dcc.Slider(
            id="temperature-slider",
            min=-10,
            max=40,
            step=1,
            value=25,
            marks={i: f"{i}°F" for i in range(-10, 41, 10)},
        ),

        html.Label("Liquid Precipitation (in):"),
        dcc.Input(id="precipitation-input", type="number", value=0.1, step=0.0001),

        html.Button("Submit", id="submit-button", n_clicks=0),
    ], style={"marginBottom": "20px"}),

    html.Div(id="output-div", style={"marginTop": "20px"}),

    dcc.Graph(id="snowfall-graph"),
])

# Callback to update the snowfall prediction
@app.callback(
    [Output("output-div", "children"), Output("snowfall-graph", "figure")],
    [Input("submit-button", "n_clicks")],
    [Input("town-selector", "value"),
     Input("season-selector", "value"),
     Input("humidity-slider", "value"),
     Input("temperature-slider", "value"),
     Input("precipitation-input", "value")]
)
def update_dashboard(n_clicks, town, season, manual_humidity, manual_temp_f, precip_in):
    if n_clicks > 0:
        lat, lon = TOWNS[town]
        temp_f, humidity, wind_direction, wind_speed = fetch_weather_data(lat, lon)

        # Use manual overrides if API data is unavailable
        if temp_f is None or humidity is None:
            temp_f, humidity = manual_temp_f, manual_humidity

        # Calculate snowfall for different time periods
        snowfall_1hr = calculate_snowfall(temp_f, humidity, precip_in, season)
        snowfall_6hr = calculate_snowfall(temp_f, humidity, precip_in * 6, season)
        snowfall_24hr = calculate_snowfall(temp_f, humidity, precip_in * 24, season)
        snowfall_48hr = calculate_snowfall(temp_f, humidity, precip_in * 48, season)

        # Create a DataFrame for plotting
        periods = ["1 Hour", "6 Hours", "24 Hours", "48 Hours"]
        snowfall_values = [snowfall_1hr, snowfall_6hr, snowfall_24hr, snowfall_48hr]
        df = pd.DataFrame({
            "Period": periods,
            "Snowfall (inches)": snowfall_values
        })

        # Generate a bar chart of snowfall predictions
        fig = px.bar(df, x="Period", y="Snowfall (inches)",
                     title=f"Predicted Snowfall for {town}", labels={"Period": "Time Period", "Snowfall (inches)": "Snowfall"})

        return (
            f"Predicted Snowfall for {town}: \n\n"
            f"Temperature: {temp_f} °F, Humidity: {humidity}%, Wind Direction: {wind_direction}°",
            fig
        )

    return "Adjust parameters and click Submit to calculate snowfall.", dash.no_update

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)