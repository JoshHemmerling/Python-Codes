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
def calculate_school_risk(temp_c, wind_speed, snow_accum, precip_rate):
    temp_f = temp_c * 9 / 5 + 32  # Convert to Fahrenheit for wind chill
    wind_chill = calculate_wind_chill(temp_f, wind_speed)
    temp_weight = temperature_weight(temp_c)
    return (
        (100 - wind_chill) * temp_weight * 0.6 +  # Higher wind chill means less risk
        snow_accum * 0.4 +                  # Higher snow accumulation increases risk
        precip_rate * 10 * 0.9              # Precipitation rate in in/hr
    )

# Risk factor calculation for manufacturing businesses
def calculate_business_risk(temp_c, wind_speed, snow_accum, precip_rate):
    temp_f = temp_c * 9 / 5 + 32  # Convert to Fahrenheit for wind chill
    wind_chill = calculate_wind_chill(temp_f, wind_speed)
    temp_weight = temperature_weight(temp_c)
    return (
        (100 - wind_chill) * temp_weight * 0.25 +  # Higher wind chill decreases customer traffic
        snow_accum * 0.8 +                  # Higher snow accumulation increases costs
        precip_rate * 10 * 1.0              # Precipitation rate in in/hr
    )

# Recommendations for businesses
def business_recommendation(risk_score):
    if risk_score < 20:
        return "No disruptions expected."
    elif 20 <= risk_score < 50:
        return "Monitor weather conditions; consider backup plans for supply chain or worker safety."
    elif 50 <= risk_score < 80:
        return "Prepare for potential operational halts; ensure worker safety measures are in place."
    else:
        return "Extreme risk! Strongly recommend halting operations; ensure all safety protocols are implemented."

# Recommendations for schools
def school_recommendation(risk_score):
    if risk_score < 20:
        return "Normal operations expected."
    elif 20 <= risk_score < 50:
        return "Monitor weather conditions; consider contingency plans."
    elif 50 <= risk_score < 80:
        return "High risk of disruption; prepare for possible closures."
    else:
        return "Extreme risk! Likely school closure due to severe weather."

# App layout
app.layout = html.Div([
    html.H1("Weather Risk Dashboard"),
    
    # Schools Dashboard
    html.Div([
        html.H2("Schools Dashboard"),
        html.Label("Temperature (°C):"),
        dcc.Input(id="school-temp", type="number", step=0.1),
        html.Label("Wind Speed (mph):"),
        dcc.Input(id="school-wind", type="number", step=0.1),
        html.Label("Snow Accumulation (in):"),
        dcc.Input(id="school-snow", type="number", step=0.1),
        html.Label("Precipitation Rate (in/hr):"),
        dcc.Input(id="school-precip", type="number", step=0.01),
        html.H3("School Risk Factor:"),
        html.Div(id="school-risk", style={"font-size": "20px", "color": "red"}),
        dcc.Graph(id="school-risk-graph"),
        html.H3("Recommendation:"),
        html.Div(id="school-recommendation", style={"font-size": "18px", "color": "green"})
    ], style={"border": "1px solid black", "padding": "20px", "margin-bottom": "20px"}),

    # Businesses Dashboard
    html.Div([
        html.H2("Manufacturing Businesses Dashboard"),
        html.Label("Temperature (°C):"),
        dcc.Input(id="business-temp", type="number", step=0.1),
        html.Label("Wind Speed (mph):"),
        dcc.Input(id="business-wind", type="number", step=0.1),
        html.Label("Snow Accumulation (in):"),
        dcc.Input(id="business-snow", type="number", step=0.1),
        html.Label("Precipitation Rate (in/hr):"),
        dcc.Input(id="business-precip", type="number", step=0.01),
        html.H3("Business Risk Factor:"),
        html.Div(id="business-risk", style={"font-size": "20px", "color": "blue"}),
        dcc.Graph(id="business-risk-graph"),
        html.H3("Recommendation:"),
        html.Div(id="business-recommendation", style={"font-size": "18px", "color": "green"})
    ], style={"border": "1px solid black", "padding": "20px"})
])

# Callbacks for Schools Dashboard
@app.callback(
    [Output("school-risk", "children"),
     Output("school-risk-graph", "figure"),
     Output("school-recommendation", "children")],
    [Input("school-temp", "value"),
     Input("school-wind", "value"),
     Input("school-snow", "value"),
     Input("school-precip", "value")]
)
def update_school_dashboard(temp, wind, snow, precip):
    if temp is None or wind is None or snow is None or precip is None:
        return "Please fill in all inputs.", {}, ""
    risk = calculate_school_risk(temp, wind, snow, precip)
    recommendation = school_recommendation(risk)
    fig = go.Figure(data=[
        go.Bar(x=["Temperature", "Wind Speed", "Snow Accumulation", "Precipitation Rate"],
               y=[temperature_weight(temp) * 100, wind, snow, precip * 10],
               marker_color=['blue', 'turquoise', 'teal', 'yellow'])
    ])
    fig.update_layout(title="School Risk Factor Contribution", yaxis_title="Impact Score")
    return f"Risk Factor: {risk:.2f} (0 = No Risk, 100 = Extreme Risk)", fig, recommendation

# Callbacks for Businesses Dashboard
@app.callback(
    [Output("business-risk", "children"),
     Output("business-risk-graph", "figure"),
     Output("business-recommendation", "children")],
    [Input("business-temp", "value"),
     Input("business-wind", "value"),
     Input("business-snow", "value"),
     Input("business-precip", "value")]
)
def update_business_dashboard(temp, wind, snow, precip):
    if temp is None or wind is None or snow is None or precip is None:
        return "Please fill in all inputs.", {}, ""
    risk = calculate_business_risk(temp, wind, snow, precip)
    recommendation = business_recommendation(risk)
    fig = go.Figure(data=[
        go.Pie(labels=["Temperature", "Wind Chill", "Snow", "Precipitation"],
               values=[temperature_weight(temp) * 100, wind, snow, precip * 10])
    ])
    fig.update_layout(title="Business Risk Factor Contribution")
    return f"Risk Factor: {risk:.2f} (0 = No Risk, 100 = Extreme Risk)", fig, recommendation

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True, port=8053)
