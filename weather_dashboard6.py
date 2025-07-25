import dash 
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
from dash import Dash
import plotly.graph_objects as go
# Initialize the Dash app
app = Dash(__name__)

# Calculate wind chill function
def calculate_wind_chill(temp_f, wind_speed):
    return 35.74 + 0.6215 * temp_f - 35.75 * (wind_speed ** 0.16) + 0.4275 * temp_f * (wind_speed ** 0.16)

# Temperature weight function
def temperature_weight(temp_c):
    if temp_c >= 0:
        return 0.35
    elif -10 < temp_c < 0:
        return 0.58
    else:
        return 0.78

# Risk factor calculation for schools
def calculate_school_risk(temp_c, wind_speed, snow_accum, precip_rate, humidity, visibility, include_humidity, include_visibility):
    temp_f = temp_c * 9 / 5 + 32  # Convert to Fahrenheit for wind chill
    wind_chill = calculate_wind_chill(temp_f, wind_speed)
    temp_weight = temperature_weight(temp_c)
    risk = (
        (100 - wind_chill) * temp_weight * 0.45 +
        snow_accum * 0.33 +
        precip_rate * 10 * 0.85
    )
    if include_humidity:
        risk += humidity * 0.05
    if include_visibility:
        risk += (100 - visibility) * 0.03
    return risk

# Risk factor calculation for manufacturing businesses
def calculate_business_risk(temp_c, wind_speed, snow_accum, precip_rate, humidity, visibility, include_humidity, include_visibility):
    temp_f = temp_c * 9 / 5 + 32  # Convert to Fahrenheit for wind chill
    wind_chill = calculate_wind_chill(temp_f, wind_speed)
    temp_weight = temperature_weight(temp_c)
    risk = (
        (100 - wind_chill) * temp_weight * 0.25 +
        snow_accum * 0.8 +
        precip_rate * 10 * 0.95
    )
    if include_humidity:
        risk += humidity * 0.05
    if include_visibility:
        risk += (100 - visibility) * 0.03
    return risk

# App layout
app.layout = html.Div(
    style={
        "font-family": "Arial, sans-serif",
        "margin": "20px",
        "background": "linear-gradient(to bottom, #87CEEB, #E3F2FD)",
        "padding": "20px",
        "border-radius": "10px",
        "box-shadow": "0px 4px 6px rgba(0, 0, 0, 0.1)"
    },
    children=[
        html.H1("Weather Risk Dashboard", style={"text-align": "center", "color": "#4CAF50"}),
        
        # Schools Dashboard
        html.Div([
            html.H2("Schools Dashboard", style={"text-align": "center", "color": "#2196F3"}),
            html.Div([
                html.Label("Temperature (°C):", style={"font-weight": "bold"}),
                dcc.Input(id="school-temp", type="number", step=0.1, placeholder="Enter temperature...", style={"width": "100%", "margin-bottom": "10px"}),
                html.Label("Wind Speed (mph):", style={"font-weight": "bold"}),
                dcc.Input(id="school-wind", type="number", step=0.1, placeholder="Enter wind speed...", style={"width": "100%", "margin-bottom": "10px"}),
                html.Label("Snow Accumulation (in):", style={"font-weight": "bold"}),
                dcc.Input(id="school-snow", type="number", step=0.1, placeholder="Enter snow accumulation...", style={"width": "100%", "margin-bottom": "10px"}),
                html.Label("Precipitation Rate (in/hr):", style={"font-weight": "bold"}),
                dcc.Input(id="school-precip", type="number", step=0.01, placeholder="Enter precipitation rate...", style={"width": "100%", "margin-bottom": "10px"}),
                html.Label("Include Humidity?", style={"font-weight": "bold"}),
                dcc.Checklist(id="school-include-humidity", options=[{"label": "Include Humidity", "value": "humidity"}], style={"margin-bottom": "10px"}),
                html.Label("Include Visibility?", style={"font-weight": "bold"}),
                dcc.Checklist(id="school-include-visibility", options=[{"label": "Include Visibility", "value": "visibility"}], style={"margin-bottom": "20px"}),
            ], style={"padding": "10px", "border": "1px solid #2196F3", "border-radius": "5px", "background-color": "#E3F2FD"}),
            html.H3("School Risk Factor:", style={"margin-top": "20px"}),
            html.Div(id="school-risk", style={"font-size": "20px", "color": "red", "text-align": "center"}),
        ], style={"margin-bottom": "40px"}),

        # Businesses Dashboard
        html.Div([
            html.H2("Manufacturing Businesses Dashboard", style={"text-align": "center", "color": "#FF9800"}),
            html.Div([
                html.Label("Temperature (°C):", style={"font-weight": "bold"}),
                dcc.Input(id="business-temp", type="number", step=0.1, placeholder="Enter temperature...", style={"width": "100%", "margin-bottom": "10px"}),
                html.Label("Wind Speed (mph):", style={"font-weight": "bold"}),
                dcc.Input(id="business-wind", type="number", step=0.1, placeholder="Enter wind speed...", style={"width": "100%", "margin-bottom": "10px"}),
                html.Label("Include Humidity?", style={"font-weight": "bold"}),
                dcc.Checklist(id="business-include-humidity", options=[{"label": "Include Humidity", "value": "humidity"}], style={"margin-bottom": "10px"}),
                html.Label("Include Visibility?", style={"font-weight": "bold"}),
                dcc.Checklist(id="business-include-visibility", options=[{"label": "Include Visibility", "value": "visibility"}], style={"margin-bottom": "20px"}),
            ], style={"padding": "10px", "border": "1px solid #FF9800", "border-radius": "5px", "background-color": "#FFF3E0"}),
            html.H3("Business Risk Factor:", style={"margin-top": "20px"}),
            html.Div(id="business-risk", style={"font-size": "20px", "color": "blue", "text-align": "center"}),
        ])
    ]
)

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True, port=8055)