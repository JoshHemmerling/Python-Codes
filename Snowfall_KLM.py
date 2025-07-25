from simplekml import Kml, Style

# Snowfall ranges and corresponding colors (RGBA to AABBGGRR format for KML)
SNOWFALL_RANGES = [
    {"range": (0, 1), "color": "80FF0000"},  # Red for 0-1 inch
    {"range": (1, 3), "color": "80FF7F00"},  # Orange for 1-3 inches
    {"range": (4, 8), "color": "80FFFF00"},  # Yellow for 4-8 inches
    {"range": (8, 12), "color": "8000FF00"},  # Green for 8-12 inches
    {"range": (12, 18), "color": "8000FFFF"},  # Cyan for 12-18 inches
    {"range": (18, 24), "color": "800000FF"},  # Blue for 18-24 inches
    {"range": (24, 30), "color": "80FF00FF"},  # Magenta for 24-30 inches
]

# Expanded list of major towns in Western New York (Erie, Niagara, Chautauqua, Allegany, Cattaraugus Counties)
TOWNS = {
    # **Erie County**
    "Buffalo": (42.8864, -78.8784),
    "Orchard Park": (42.7675, -78.7431),
    "Hamburg": (42.7156, -78.8295),
    "East Aurora": (42.767, -78.6136),
    "Springville": (42.5084, -78.6642),
    "Williamsville": (42.9634, -78.735),
    "Cheektowaga": (42.9025, -78.7446),
    "Lancaster": (42.9006, -78.6703),
    "Amherst": (42.9784, -78.7998),
    "Tonawanda": (43.0203, -78.8803),
    "Kenmore": (42.9656, -78.8717),
    "Grand Island": (43.0205, -78.9612),
    "Clarence": (42.9776, -78.5778),

    # **Niagara County**
    "Niagara Falls": (43.0962, -79.0377),
    "North Tonawanda": (43.0387, -78.8642),
    "Lockport": (43.1706, -78.6903),
    "Lewiston": (43.1725, -79.0428),
    "Newfane": (43.2853, -78.7086),
    "Wilson": (43.3114, -78.8289),

    # **Chautauqua County**
    "Jamestown": (42.097, -79.2353),
    "Dunkirk": (42.4795, -79.3333),
    "Fredonia": (42.4401, -79.3314),
    "Westfield": (42.3223, -79.5784),
    "Lakewood": (42.0987, -79.3445),

    # **Allegany County**
    "Wellsville": (42.1223, -77.9472),
    "Alfred": (42.2534, -77.7894),
    "Belmont": (42.2223, -78.0347),
    "Fillmore": (42.4636, -78.1178),
    "Bolivar": (42.0678, -78.1689),

    # **Cattaraugus County**
    "Olean": (42.0836, -78.4292),
    "Salamanca": (42.1573, -78.7156),
    "Ellicottville": (42.2759, -78.6692),
    "Little Valley": (42.2492, -78.7992),
    "Gowanda": (42.4636, -78.9353),
    "Franklinville": (42.3384, -78.4567)
}

# **Input snowfall amounts for each town manually**
snowfall_amounts = {
    "Buffalo": 2,
    "Orchard Park":56,
    "Hamburg": 6,
    "East Aurora": 6,
    "Springville": 7,
    "Williamsville": 2,
    "Cheektowaga": 4,
    "Lancaster": 4,
    "Amherst": 4,
    "Tonawanda": 2,
    "Kenmore": 3,
    "Grand Island": 2,
    "Clarence": 4,
    "Niagara Falls": 1,
    "North Tonawanda": 2,
    "Lockport": 4,
    "Lewiston": 3,
    "Newfane": 2,
    "Wilson": 1,
    "Jamestown": 18,
    "Dunkirk": 13,
    "Fredonia": 14,
    "Westfield": 10,
    "Lakewood": 9,
    "Wellsville": 17,
    "Alfred": 22,
    "Belmont": 20,
    "Fillmore": 23,
    "Bolivar": 24,
    "Olean": 25,
    "Salamanca": 19,
    "Ellicottville": 28,
    "Little Valley": 21,
    "Gowanda": 27,
    "Franklinville": 26
}

# Function to determine color based on snowfall
def get_color_for_snowfall(snowfall):
    for range_info in SNOWFALL_RANGES:
        low, high = range_info["range"]
        if low <= snowfall <= high:
            return range_info["color"]
    return "80FFFFFF"  # Default white if no range matches

# Function to generate a bounding box around a point
def generate_bounding_box(lat, lon, offset=0.1):
    """Creates a bounding box (rectangle) around a lat/lon point."""
    return [
        (lon - offset, lat - offset),  # Bottom-left
        (lon + offset, lat - offset),  # Bottom-right
        (lon + offset, lat + offset),  # Top-right
        (lon - offset, lat + offset),  # Top-left
        (lon - offset, lat - offset),  # Closing the polygon
    ]

# Create KML
kml = Kml()

# Generate snowfall zones
for range_info in SNOWFALL_RANGES:
    low, high = range_info["range"]
    color = range_info["color"]

    # Create a new style for this range
    style = Style()
    style.polystyle.color = color
    style.linestyle.color = color
    style.linestyle.width = 2

    for town, coords in TOWNS.items():
        snowfall = snowfall_amounts.get(town, 0)
        if low <= snowfall <= high:
            # Generate bounding box for this town
            polygon_coords = generate_bounding_box(*coords)
            pol = kml.newpolygon(name=f"{town}: {low}-{high} inches",
                                 outerboundaryis=polygon_coords)
            pol.style = style

# Save the KML file
kml.save("Snowfall_Zones_WNY.kml")
print("KML file 'Snowfall_Zones_WNY.kml' created.")


