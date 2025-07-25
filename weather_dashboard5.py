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
def calculate_school_risk(temp_c, wind_speed, snow_accum, precip_rate, humidity, visibility):
    temp_f = temp_c * 9 / 5 + 32  # Convert to Fahrenheit for wind chill
    wind_chill = calculate_wind_chill(temp_f, wind_speed)
    temp_weight = temperature_weight(temp_c)
    return (
        (100 - wind_chill) * temp_weight * 0.45 +
        snow_accum * 0.33 +
        precip_rate * 10 * 0.85 +
        (100 - visibility) * 0.03 +
        humidity * 0.05
    )

# Risk factor calculation for manufacturing businesses
def calculate_business_risk(temp_c, wind_speed, snow_accum, precip_rate, humidity, visibility):
    temp_f = temp_c * 9 / 5 + 32  # Convert to Fahrenheit for wind chill
    wind_chill = calculate_wind_chill(temp_f, wind_speed)
    temp_weight = temperature_weight(temp_c)
    return (
        (100 - wind_chill) * temp_weight * 0.25 +
        snow_accum * 0.8 +
        precip_rate * 10 * 0.95 +
        (100 - visibility) * 0.03 +
        humidity * 0.05
    )

# Recommendations for businesses
def business_recommendation(risk_score):
    if risk_score < 20:
        return "No disruptions expected. Proceed with normal operations."
    elif 20 <= risk_score < 50:
        return (
            "Moderate risk. Monitor weather conditions and optimize shifts or "
            "adjust logistics to prevent delays."
        )
    elif 50 <= risk_score < 80:
        return (
            "High risk. Prepare for potential delays or operational halts. "
            "Ensure worker safety protocols are implemented."
        )
    else:
        return (
            "Extreme risk! Halt operations immediately. Implement all emergency safety measures and inform stakeholders."
        )

# Recommendations for schools
def school_recommendation(risk_score):
    if risk_score < 20:
        return "Normal operations expected. No weather-related issues."
    elif 20 <= risk_score < 50:
        return (
            "Moderate risk. Monitor weather conditions and prepare contingency plans."
        )
    elif 50 <= risk_score < 80:
        return (
            "High risk. Prepare for possible closures. Inform parents and staff about potential disruptions."
        )
    else:
        return (
            "Extreme risk! Close schools immediately and activate remote learning protocols."
        )

# App layout
app.layout = html.Div(
    style={"font-family": "Arial, sans-serif", "margin": "20px"},
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
                html.Label("Humidity (%):", style={"font-weight": "bold"}),
                dcc.Input(id="school-humidity", type="number", step=1, min=0, max=100, placeholder="Enter humidity...", style={"width": "100%", "margin-bottom": "10px"}),
                html.Label("Visibility (miles):", style={"font-weight": "bold"}),
                dcc.Input(id="school-visibility", type="number", step=0.1, placeholder="Enter visibility...", style={"width": "100%", "margin-bottom": "20px"}),
            ], style={"padding": "10px", "border": "1px solid #2196F3", "border-radius": "5px", "background-color": "#E3F2FD"}),
            html.H3("School Risk Factor:", style={"margin-top": "20px"}),
            html.Div(id="school-risk", style={"font-size": "20px", "color": "red", "text-align": "center"}),
            dcc.Graph(id="school-risk-graph"),
            html.H3("Recommendation:", style={"margin-top": "20px"}),
            html.Div(id="school-recommendation", style={"font-size": "18px", "color": "green", "text-align": "center"})
        ], style={"margin-bottom": "40px"}),

        # Businesses Dashboard
        html.Div([
            html.H2("Manufacturing Businesses Dashboard", style={"text-align": "center", "color": "#FF9800"}),
            html.Div([
                html.Label("Temperature (°C):", style={"font-weight": "bold"}),
                dcc.Input(id="business-temp", type="number", step=0.1, placeholder="Enter temperature...", style={"width": "100%", "margin-bottom": "10px"}),
                html.Label("Wind Speed (mph):", style={"font-weight": "bold"}),
                dcc.Input(id="business-wind", type="number", step=0.1, placeholder="Enter wind speed...", style={"width": "100%", "margin-bottom": "10px"}),
                html.Label("Snow Accumulation (in):", style={"font-weight": "bold"}),
                dcc.Input(id="business-snow", type="number", step=0.1, placeholder="Enter snow accumulation...", style={"width": "100%", "margin-bottom": "10px"}),
                html.Label("Precipitation Rate (in/hr):", style={"font-weight": "bold"}),
                dcc.Input(id="business-precip", type="number", step=0.01, placeholder="Enter precipitation rate...", style={"width": "100%", "margin-bottom": "10px"}),
                html.Label("Humidity (%):", style={"font-weight": "bold"}),
                dcc.Input(id="business-humidity", type="number", step=1, min=0, max=100, placeholder="Enter humidity...", style={"width": "100%", "margin-bottom": "10px"}),
                html.Label("Visibility (miles):", style={"font-weight": "bold"}),
                dcc.Input(id="business-visibility", type="number", step=0.1, placeholder="Enter visibility...", style={"width": "100%", "margin-bottom": "20px"}),
            ], style={"padding": "10px", "border": "1px solid #FF9800", "border-radius": "5px", "background-color": "#FFF3E0"}),
            html.H3("Business Risk Factor:", style={"margin-top": "20px"}),
            html.Div(id="business-risk", style={"font-size": "20px", "color": "blue", "text-align": "center"}),
            dcc.Graph(id="business-risk-graph"),
            html.H3("Recommendation:", style={"margin-top": "20px"}),
            html.Div(id="business-recommendation", style={"font-size": "18px", "color": "green", "text-align": "center"})
        ])
    ]
)

# Callbacks remain unchanged as the logic is the same
@app.callback(
    [Output("school-risk", "children"),
     Output("school-risk-graph", "figure"),
     Output("school-recommendation", "children")],
    [Input("school-temp", "value"),
     Input("school-wind", "value"),
     Input("school-snow", "value"),
     Input("school-precip", "value"),
     Input("school-humidity", "value"),
     Input("school-visibility", "value")]
)
def update_school_dashboard(temp, wind, snow, precip, humidity, visibility):
    if None in [temp, wind, snow, precip, humidity, visibility]:
        return "Please fill in all inputs.", {}, ""
    risk = calculate_school_risk(temp, wind, snow, precip, humidity, visibility)
    recommendation = school_recommendation(risk)
    fig = go.Figure(data=[
        go.Bar(
            x=["Temperature", "Wind Chill", "Snow", "Precipitation", "Humidity", "Visibility"],
            y=[temperature_weight(temp) * 100, wind, snow, precip * 10, humidity, 100 - visibility],
            marker_color=['blue', 'turquoise', 'teal', 'yellow', 'green', 'orange']
        )
    ])
    fig.update_layout(title="School Risk Factor Contribution", yaxis_title="Impact Score")
    return f"Risk Factor: {risk:.2f} (0 = No Risk, 100 = Extreme Risk)", fig, recommendation

@app.callback(
    [Output("business-risk", "children"),
     Output("business-risk-graph", "figure"),
     Output("business-recommendation", "children")],
    [Input("business-temp", "value"),
     Input("business-wind", "value"),
     Input("business-snow", "value"),
     Input("business-precip", "value"),
     Input("business-humidity", "value"),
     Input("business-visibility", "value")]
)
def update_business_dashboard(temp, wind, snow, precip, humidity, visibility):
    if None in [temp, wind, snow, precip, humidity, visibility]:
        return "Please fill in all inputs.", {}, ""
    risk = calculate_business_risk(temp, wind, snow, precip, humidity, visibility)
    recommendation = business_recommendation(risk)
    fig = go.Figure(data=[
        go.Pie(
            labels=["Temperature", "Wind Chill", "Snow", "Precipitation", "Humidity", "Visibility"],
            values=[temperature_weight(temp) * 100, wind, snow, precip * 10, humidity, 100 - visibility]
        )
    ])
    fig.update_layout(title="Business Risk Factor Contribution")
    return f"Risk Factor: {risk:.2f} (0 = No Risk, 100 = Extreme Risk)", fig, recommendation

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True, port=8054)
