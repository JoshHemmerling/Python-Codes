import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Snowfall Prediction Dashboard"

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

# Layout of the Dash app
app.layout = html.Div([
    html.H1("Snowfall Prediction Dashboard", style={"textAlign": "center"}),

    html.Div([
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
        dcc.Input(id="precipitation-input", type="number", value=0.1, step=0.01),

        html.Button("Submit", id="submit-button", n_clicks=0),
    ], style={"marginBottom": "20px"}),

    html.Div(id="output-div", style={"marginTop": "20px"}),

    dcc.Graph(id="snowfall-graph"),
])

# Callback to update the snowfall prediction
@app.callback(
    [Output("output-div", "children"), Output("snowfall-graph", "figure")],
    [Input("submit-button", "n_clicks")],
    [Input("season-selector", "value"),
     Input("humidity-slider", "value"),
     Input("temperature-slider", "value"),
     Input("precipitation-input", "value")]
)
def update_dashboard(n_clicks, season, humidity, temp_f, precip_in):
    if n_clicks > 0:
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
                     title="Predicted Snowfall", labels={"Period": "Time Period", "Snowfall (inches)": "Snowfall"})

        return f"Predicted Snowfall based on current conditions:", fig

    return "Adjust parameters and click Submit to calculate snowfall.", dash.no_update

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
