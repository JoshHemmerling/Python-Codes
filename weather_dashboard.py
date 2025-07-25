import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
from dash import Dash

# Initialize the Dash app
app = Dash(__name__)

# Calculate wind chill function
def calculate_wind_chill(temp_f, wind_speed):
    return 35.74 + 0.6215 * temp_f - 35.75 * (wind_speed ** 0.16) + 0.4275 * temp_f * (wind_speed ** 0.16)

# Risk factor calculation for schools
def calculate_school_risk(temp_c, wind_speed, snow_accum, precip_rate):
    temp_f = temp_c * 9 / 5 + 32  # Convert to Fahrenheit for wind chill
    wind_chill = calculate_wind_chill(temp_f, wind_speed)
    # Risk factor (example weights for school safety)
    return (
        (100 - wind_chill) * 0.6 +  # Higher wind chill means less risk
        snow_accum * 0.4 +          # Higher snow accumulation increases risk
        precip_rate * 10 * 0.9     # Precipitation rate in in/hr
    )

# Risk factor calculation for businesses
def calculate_business_risk(temp_c, wind_speed, snow_accum, precip_rate):
    temp_f = temp_c * 9 / 5 + 32  # Convert to Fahrenheit for wind chill
    wind_chill = calculate_wind_chill(temp_f, wind_speed)
    # Risk factor (example weights for business optimization)
    return (
        (100 - wind_chill) * 0.3 +  # Higher wind chill decreases customer traffic
        snow_accum * 0.8 +          # Higher snow accumulation increases costs
        precip_rate * 10 * 1      # Precipitation rate in in/hr
    )

# App layout
app.layout = html.Div([
    html.H1("Weather Risk Dashboard"),
    
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
    ], style={"border": "1px solid black", "padding": "20px", "margin-bottom": "20px"}),

    html.Div([
        html.H2("Businesses Dashboard"),
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
    ], style={"border": "1px solid black", "padding": "20px"})
])

# Callbacks for Schools Dashboard
@app.callback(
    Output("school-risk", "children"),
    [Input("school-temp", "value"),
     Input("school-wind", "value"),
     Input("school-snow", "value"),
     Input("school-precip", "value")]
)
def update_school_risk(temp, wind, snow, precip):
    if temp is None or wind is None or snow is None or precip is None:
        return "Please fill in all inputs."
    risk = calculate_school_risk(temp, wind, snow, precip)
    return f"Risk Factor: {risk:.2f} (0 = No Risk, 100 = High Risk)"

# Callbacks for Businesses Dashboard
@app.callback(
    Output("business-risk", "children"),
    [Input("business-temp", "value"),
     Input("business-wind", "value"),
     Input("business-snow", "value"),
     Input("business-precip", "value")]
)
def update_business_risk(temp, wind, snow, precip):
    if temp is None or wind is None or snow is None or precip is None:
        return "Please fill in all inputs."
    risk = calculate_business_risk(temp, wind, snow, precip)
    return f"Risk Factor: {risk:.2f} (0 = No Risk, 100 = High Risk)"

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True, port=8051)