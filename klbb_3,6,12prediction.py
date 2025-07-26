import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier # Import LightGBM

# Load and preprocess data
df = pd.read_csv("klbb_wrf_features.csv", encoding='latin1')

# Convert 'XTIME' to datetime and set as index
df['XTIME'] = pd.to_datetime(df['XTIME'])
df = df.set_index('XTIME')

# Drop the 'Unnamed: 0' column
df = df.drop(columns=['Unnamed: 0'])

# Drop specified columns
columns_to_drop = ['afwa_tlyrbot', 'afwa_afwa_tlyrtop', 'afwa_turb', 'C1H', 'C2H', 'C1F', 'C2F', 'C3H', 'C4H', 'C3F', 'C4F', 'AFWA_TLYRBOT', 'AFWA_TLYRTOP', 'AFWA_TURB']
df = df.drop(columns=columns_to_drop, errors='ignore')

# Replace placeholder values with NaN
df = df.replace(-9.999000e+30, np.nan)

# Drop any remaining non-numeric columns
numeric_cols = df.select_dtypes(include=np.number).columns
df = df[numeric_cols]

# Fill remaining missing values with the mean
df = df.fillna(df.mean())

# Create time-based features
df['hour_of_day'] = df.index.hour
df['day_of_year'] = df.index.dayofyear

# Create lagged features for UP_HELI_MAX at 10-minute intervals
df['UP_HELI_MAX_lag_1'] = df['UP_HELI_MAX'].shift(1)
df['UP_HELI_MAX_lag_2'] = df['UP_HELI_MAX'].shift(2)
df['UP_HELI_MAX_lag_3'] = df['UP_HELI_MAX'].shift(3)

# Create rolling mean feature for UP_HELI_MAX with a window of 3 at 10-minute intervals
df['UP_HELI_MAX_rolling_mean_3'] = df['UP_HELI_MAX'].rolling(window=3).mean()

# Drop rows with NaN values introduced by lagging and rolling mean
df = df.dropna()

print("Data loading, preprocessing, and feature creation complete.")
print(f"Shape of the final DataFrame: {df.shape}")
print(df.head())

# Define the multi-class target categorization function (assuming it's the same as before)
# This function needs to be defined once before the loop
def categorize_storm_intensity_percentile_10min(up_heli_max):
    # Assuming percentile thresholds were calculated earlier in the notebook
    # If not, they need to be calculated here based on the non-zero UP_HELI_MAX of the *original* df
    # For consistency, let's recalculate them here based on the df after initial cleaning but before lagging
    df_original_cleaned = pd.read_csv("klbb_wrf_features.csv", encoding='latin1')
    df_original_cleaned['XTIME'] = pd.to_datetime(df_original_cleaned['XTIME'])
    df_original_cleaned = df_original_cleaned.set_index('XTIME')
    df_original_cleaned = df_original_cleaned.drop(columns=['Unnamed: 0'])
    columns_to_drop = ['afwa_tlyrbot', 'afwa_afwa_tlyrtop', 'afwa_turb', 'C1H', 'C2H', 'C1F', 'C2F', 'C3H', 'C4H', 'C3F', 'C4F', 'AFWA_TLYRBOT', 'AFWA_TLYRTOP', 'AFWA_TURB']
    df_original_cleaned = df_original_cleaned.drop(columns=columns_to_drop, errors='ignore')
    df_original_cleaned = df_original_cleaned.replace(-9.999000e+30, np.nan)
    numeric_cols_original = df_original_cleaned.select_dtypes(include=np.number).columns
    df_original_cleaned = df_original_cleaned[numeric_cols_original]
    df_original_cleaned = df_original_cleaned.fillna(df_original_cleaned.mean())
    y_events_original = df_original_cleaned[df_original_cleaned['UP_HELI_MAX'] > 0]['UP_HELI_MAX']

    threshold_light_percentile = y_events_original.quantile(0.50)
    threshold_moderate_percentile = y_events_original.quantile(0.75)
    threshold_heavy_percentile = y_events_original.quantile(0.90)


    if up_heli_max == 0:
        return 0 # No storm
    elif up_heli_max <= threshold_light_percentile:
        return 1 # Light storm
    elif up_heli_max <= threshold_moderate_percentile:
        return 2 # Moderate storm
    elif up_heli_max <= threshold_heavy_percentile:
        return 3 # Heavy storm
    else:
        return 4 # Very Heavy Storm


# Define prediction horizons
prediction_horizons_hours = [3, 6, 12]
interval_minutes = 10
prediction_horizons_intervals = [int(h * 60 / interval_minutes) for h in prediction_horizons_hours]

# Store evaluation metrics for each horizon
evaluation_results = {}

# Iterate through each defined prediction horizon (in intervals)
for shift_intervals in prediction_horizons_intervals:
    prediction_horizon_hours = int(shift_intervals * interval_minutes / 60)
    print(f"\n--- Processing Prediction Horizon: {prediction_horizon_hours} hours ({shift_intervals} intervals) ---")

    # Create a new target column by shifting UP_HELI_MAX
    df_shifted = df.copy()
    df_shifted['UP_HELI_MAX_future'] = df_shifted['UP_HELI_MAX'].shift(-shift_intervals)

    # Apply the categorize_storm_intensity_percentile_10min function to the future target
    y_intensity_percentile_future = df_shifted['UP_HELI_MAX_future'].apply(categorize_storm_intensity_percentile_10min)

    # Drop rows where the future target is NaN (due to shifting)
    df_future_target = df_shifted.copy()
    df_future_target['y_intensity_percentile_future'] = y_intensity_percentile_future
    df_future_target = df_future_target.dropna(subset=['y_intensity_percentile_future'])

    y_current_horizon = df_future_target['y_intensity_percentile_future']
    X_current_horizon = df_future_target.drop(columns=['UP_HELI_MAX', 'UP_HELI_MAX_future', 'y_intensity_percentile_future'])

    # Print the distribution of the newly created multi-class target variable
    print(f"\nDistribution of Storm Intensity Categories for {prediction_horizon_hours}-hour future:")
    print(y_current_horizon.value_counts().sort_index())


    # Split data into training and test sets for the current horizon
    X_train_current, X_test_current, y_train_current, y_test_current = train_test_split(
        X_current_horizon, y_current_horizon,
        test_size=0.2,
        random_state=42,
        stratify=y_current_horizon # Stratify by the multi-class target
    )

    print("\nShapes of the resulting splits:")
    print("X_train_current:", X_train_current.shape)
    print("X_test_current:", X_test_current.shape)
    print("y_train_current:", y_train_current.shape)
    print("y_test_current:", y_test_current.shape)

    # Scale features for the current horizon
    scaler_current = StandardScaler()
    X_train_current_scaled = scaler_current.fit_transform(X_train_current)
    X_test_current_scaled = scaler_current.transform(X_test_current)
    print("\nScaling completed.")

    # Train a Multi-class LightGBM Classifier for the current horizon
    lgbm_current_classifier = LGBMClassifier(objective='multiclass',
                                            num_class=len(np.unique(y_train_current)),
                                            metric='multi_logloss',
                                            random_state=42,
                                            n_jobs=-1)

    lgbm_current_classifier.fit(X_train_current_scaled, y_train_current)
    print(f"\nLightGBM classifier for {prediction_horizon_hours}-hour prediction trained successfully.")

    # Evaluate the classifier for the current horizon
    y_pred_current = lgbm_current_classifier.predict(X_test_current_scaled)

    # Store evaluation metrics
    horizon_key = f"{prediction_horizon_hours} hours"
    evaluation_results[horizon_key] = {
        'accuracy': accuracy_score(y_test_current, y_pred_current),
        'classification_report': classification_report(y_test_current, y_pred_current), # Store as string
        'confusion_matrix': confusion_matrix(y_test_current, y_pred_current).tolist() # Convert to list for easy printing
    }


print("\n--- Summary of Evaluation Results for All Prediction Horizons ---")

# Display the stored evaluation metrics for each prediction horizon
for horizon, metrics in evaluation_results.items():
    print(f"\n--- Results for {horizon} ---")
    print(f"Accuracy: {metrics['accuracy']:.2f}")
    print("Classification Report:")
    print(metrics['classification_report']) # Print the stored string classification report

    print("Confusion Matrix:")
    print(np.array(metrics['confusion_matrix'])) # Convert back to numpy array for better display

print("\n--- All prediction horizons processed ---")
