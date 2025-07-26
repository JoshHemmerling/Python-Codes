import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier # Import LightGBM
import warnings

# Suppress FutureWarning from sklearn
warnings.filterwarnings("ignore", category=FutureWarning)


# --- Data Loading and Preprocessing ---
# Load CSV
df = pd.read_csv("KLBB_wrf_2d_May_Aug1998.csv", encoding='latin1', low_memory=False)
print(f"Shape before handling NaNs: {df.shape}")

# Drop the 'Unnamed: 0' column if it exists
columns_to_drop_initial = ['Unnamed: 0']
df = df.drop(columns=columns_to_drop_initial, errors='ignore')

# Parse 'XTIME' as datetime and set as index if it exists
if 'XTIME' in df.columns:
    df['XTIME'] = pd.to_datetime(df['XTIME'])
    df = df.set_index('XTIME')
else:
    print("Warning: 'XTIME' column not found. Cannot set as index.")


# Drop specified columns
columns_to_drop = ['afwa_tlyrbot', 'afwa_afwa_tlyrtop', 'afwa_turb', 'C1H', 'C2H', 'C1F', 'C2F', 'C3H', 'C4H', 'C3F', 'C4F', 'AFWA_TLYRBOT', 'AFWA_TLYRTOP', 'AFWA_TURB']
df = df.drop(columns=columns_to_drop, errors='ignore')
print(f"Shape after dropping specified columns: {df.shape}")

# Replace placeholder values with NaN
df = df.replace(-9.999000e+30, np.nan)

# Drop any remaining non-numeric columns
numeric_cols = df.select_dtypes(include=np.number).columns
df = df[numeric_cols]
print(f"Shape after dropping non-numeric columns: {df.shape}")

# Fill remaining missing values with the mean
df = df.fillna(df.mean())
print(f"Shape after filling remaining NaNs: {df.shape}")

# Check for remaining NaNs
print("Checking for remaining NaNs:")
print(df.isnull().sum()[df.isnull().sum() > 0])


# --- Create time-based features (10-min intervals) ---
# Only create time-based features if 'XTIME' was successfully set as index
if isinstance(df.index, pd.DatetimeIndex):
    df['hour_of_day'] = df.index.hour
    df['day_of_year'] = df.index.dayofyear

    # Create lagged features for UP_HELI_MAX at 10-minute intervals
    if 'UP_HELI_MAX' in df.columns:
        df['UP_HELI_MAX_lag_1'] = df['UP_HELI_MAX'].shift(1)
        df['UP_HELI_MAX_lag_2'] = df['UP_HELI_MAX'].shift(2)
        df['UP_HELI_MAX_lag_3'] = df['UP_HELI_MAX'].shift(3)

        # Create rolling mean feature for UP_HELI_MAX with a window of 3 at 10-minute intervals
        df['UP_HELI_MAX_rolling_mean_3'] = df['UP_HELI_MAX'].rolling(window=3).mean()

        # Drop rows with NaN values introduced by lagging and rolling mean
        df = df.dropna()
        print("\nHead of DataFrame after creating time-based and lagged features:")
        print(df.head())
    else:
        print("Warning: 'UP_HELI_MAX' column not found. Cannot create lagged or rolling mean features.")
else:
    print("Warning: Time-based features and lagged features cannot be created as 'XTIME' was not set as index.")


# --- Create multi-class target (10-min intervals) ---
# Filter the UP_HELI_MAX column for non-zero values
if 'UP_HELI_MAX' in df.columns:
    y_events = df[df['UP_HELI_MAX'] > 0]['UP_HELI_MAX']

    # Calculate percentile thresholds
    threshold_light_percentile = y_events.quantile(0.50)
    threshold_moderate_percentile = y_events.quantile(0.75)
    threshold_heavy_percentile = y_events.quantile(0.90)

    print(f"\nPercentile thresholds for non-zero UP_HELI_MAX:")
    print(f"50th percentile (potential Light storm threshold): {threshold_light_percentile:.4f}")
    print(f"75th percentile (potential Moderate storm threshold): {threshold_moderate_percentile:.4f}")
    print(f"90th percentile (potential Heavy storm threshold): {threshold_heavy_percentile:.4f}")

    # Define the multi-class target categorization function
    def categorize_storm_intensity_percentile_10min(up_heli_max):
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

    # Apply the function to create the multi-class target variable
    y_intensity_percentile_reindexed = df['UP_HELI_MAX'].apply(categorize_storm_intensity_percentile_10min)

    # Print the distribution of the new multi-class target variable
    print("\nDistribution of Storm Intensity Categories (based on Percentiles for 10-min data):")
    print(y_intensity_percentile_reindexed.value_counts().sort_index())


    # --- Split data into training and test sets (10-min intervals) ---
    # Define features (X) and target (y)
    X = df.drop(columns=['UP_HELI_MAX'])
    y = y_intensity_percentile_reindexed

    # Split data into training and test sets
    X_train_intensity_perc_10min, X_test_intensity_perc_10min, y_train_intensity_perc_10min, y_test_intensity_perc_10min = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nShapes of the resulting splits:")
    print("X_train_intensity_perc_10min:", X_train_intensity_perc_10min.shape)
    print("X_test_intensity_perc_10min:", X_test_intensity_perc_10min.shape)
    print("y_train_intensity_perc_10min:", y_train_intensity_perc_10min.shape)
    print("y_test_intensity_perc_10min:", y_test_intensity_perc_10min.shape)


    # --- Scale features (10-min intervals) ---
    # Instantiate StandardScaler
    scaler_10min = StandardScaler()

    # Fit and transform the training data, transform the test data
    X_train_intensity_perc_10min = scaler_10min.fit_transform(X_train_intensity_perc_10min)
    X_test_intensity_perc_10min = scaler_10min.transform(X_test_intensity_perc_10min)

    print("\nScaling completed.")


    # --- Train multi-class LightGBM classifier (10-min intervals) ---
    lgbm_intensity_classifier_10min = LGBMClassifier(objective='multiclass',
                                                   num_class=len(np.unique(y_train_intensity_perc_10min)),
                                                   metric='multi_logloss',
                                                   random_state=42,
                                                   n_jobs=-1)

    lgbm_intensity_classifier_10min.fit(X_train_intensity_perc_10min, y_train_intensity_perc_10min)

    print("\nLightGBM multi-class classifier (10-min intervals) trained successfully.")


    # --- Evaluate multi-class LightGBM classifier (10-min intervals) ---
    # Predict on the test set
    y_pred_lgbm_intensity_10min = lgbm_intensity_classifier_10min.predict(X_test_intensity_perc_10min)

    # Evaluate the multi-class LightGBM classifier (10-min intervals)
    print("\n--- Multi-class Classification Evaluation (LightGBM based on Percentiles for 10-min data) ---")
    print(f"Accuracy: {accuracy_score(y_test_intensity_perc_10min, y_pred_lgbm_intensity_10min):.2f}")
    print("Classification Report:")
    print(classification_report(y_test_intensity_perc_10min, y_pred_lgbm_intensity_10min))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_intensity_perc_10min, y_pred_lgbm_intensity_10min))


    # --- Prepare future data for prediction (10-min intervals) ---
    # 1. Create a future DataFrame with 10-minute frequency
    last_timestamp = df.index.max()
    future_start_time = last_timestamp + pd.Timedelta(minutes=10)
    future_end_time = future_start_time + pd.Timedelta(hours=24) # Predict 24 hours into the future
    future_time_index = pd.date_range(start=future_start_time, end=future_end_time, freq='10min')

    # Create a future DataFrame with the same columns as the original df, filled with NaNs initially
    df_future_10min = pd.DataFrame(index=future_time_index, columns=df.columns)

    # 2. Reindex the original df to include future timestamps
    df_reindexed_future = df.reindex(df.index.union(df_future_10min.index))

    # Fill NaNs introduced by reindexing with the mean of the original columns
    original_means = df.mean()
    df_reindexed_future = df_reindexed_future.fillna(original_means)

    # 3. Create time-based features for the reindexed DataFrame
    if isinstance(df_reindexed_future.index, pd.DatetimeIndex):
        df_reindexed_future['hour_of_day'] = df_reindexed_future.index.hour
        df_reindexed_future['day_of_year'] = df_reindexed_future.index.dayofyear

        # 4. Create lagged features and rolling mean for the reindexed DataFrame
        if 'UP_HELI_MAX' in df_reindexed_future.columns:
            df_reindexed_future['UP_HELI_MAX_lag_1'] = df_reindexed_future['UP_HELI_MAX'].shift(1)
            df_reindexed_future['UP_HELI_MAX_lag_2'] = df_reindexed_future['UP_HELI_MAX'].shift(2)
            df_reindexed_future['UP_HELI_MAX_lag_3'] = df_reindexed_future['UP_HELI_MAX'].shift(3)
            df_reindexed_future['UP_HELI_MAX_rolling_mean_3'] = df_reindexed_future['UP_HELI_MAX'].rolling(window=3).mean()

            # 5. Fill any NaNs introduced by lagged and rolling mean features
            df_reindexed_future = df_reindexed_future.fillna(df_reindexed_future.mean()) # Fill NaNs after creating lagged features

            # 6. Extract the future data portion
            X_future_10min = df_reindexed_future.loc[future_time_index].drop(columns=['UP_HELI_MAX'])

            print("\nFuture dataset preparation completed.")
            print("Head of X_future_10min:")
            print(X_future_10min.head())
            print("Tail of X_future_10min:")
            print(X_future_10min.tail())

            # 7. Ensure columns match the training data
            # Convert X_train_intensity_perc_10min (numpy array) back to DataFrame to easily compare columns
            X_train_intensity_perc_10min_df = pd.DataFrame(X_train_intensity_perc_10min, columns=X.columns)

            # Align columns - add missing columns to X_future_10min with fill value 0, and drop extra columns
            missing_cols_future = set(X_train_intensity_perc_10min_df.columns) - set(X_future_10min.columns)
            for c in missing_cols_future:
                X_future_10min[c] = 0

            extra_cols_future = set(X_future_10min.columns) - set(X_train_intensity_perc_10min_df.columns)
            X_future_10min = X_future_10min.drop(columns=list(extra_cols_future))

            # Ensure the order of columns is the same
            X_future_10min = X_future_10min[X_train_intensity_perc_10min_df.columns]

            print("\nColumns in X_future_10min after alignment:")
            print(X_future_10min.columns)
            print("\nColumns in X_train_intensity_perc_10min_df:")
            print(X_train_intensity_perc_10min_df.columns)

            print("\nShape of X_future_10min after alignment:", X_future_10min.shape)


            # --- Scale future data (10-min intervals) ---
            X_future_10min = scaler_10min.transform(X_future_10min)
            print("\nFuture data scaled successfully.")


            # --- Predict future storm intensity (10-min intervals) ---
            future_intensity_predictions_10min = lgbm_intensity_classifier_10min.predict(X_future_10min)

            print("\nStorm intensity predictions for future data have been made.")


            # --- Map predicted intensity to a single metric (10-min intervals) ---
            # Calculate mean UP_HELI_MAX for each intensity category using the 10-min training data
            mean_up_heli_max_by_category_10min = df['UP_HELI_MAX'].groupby(y_intensity_percentile_reindexed).mean()
            print("\nMean UP_HELI_MAX by Storm Intensity Category (10-min data):")
            print(mean_up_heli_max_by_category_10min)

            # Create a Series to store estimated UP_HELI_MAX for future time steps using the original future index
            future_estimated_up_heli_max_10min = pd.Series(index=df_future_10min.index)

            # Map the predicted intensity categories to the mean UP_HELI_MAX values
            future_estimated_up_heli_max_10min = pd.Series(future_intensity_predictions_10min, index=df_future_10min.index).map(mean_up_heli_max_by_category_10min)

            # Print the head of the estimated UP_HELI_MAX for future steps
            print("\nEstimated UP_HELI_MAX for future time steps (10-min intervals):")
            print(future_estimated_up_heli_max_10min.head())


            # --- Identify predicted storm instances (10-min intervals) ---
            # 1. Create a boolean mask for predicted storm instances
            storm_mask_10min = future_intensity_predictions_10min > 0

            # 2. Filter the X_future_10min (NumPy array) using the mask
            X_predicted_storms_10min_array = X_future_10min[storm_mask_10min]

            # 3. Create a DataFrame from the filtered features, using the index from the original df_future_10min filtered by the same mask
            # Use the columns from the original X DataFrame
            X_predicted_storms_10min = pd.DataFrame(X_predicted_storms_10min_array, index=df_future_10min.index[storm_mask_10min], columns=X.columns)

            # 4. Filter the future_estimated_up_heli_max_storms_10min Series using the same boolean mask
            future_estimated_up_heli_max_storms_10min = future_estimated_up_heli_max_10min[storm_mask_10min]

            # 5. Print the number of predicted storm instances and display the head of the filtered data
            print(f"\nNumber of predicted storm instances (10-min intervals): {X_predicted_storms_10min.shape[0]}")
            print("\nHead of X_predicted_storms_10min:")
            print(X_predicted_storms_10min.head())
            print("\nHead of future_estimated_up_heli_max_storms_10min:")
            print(future_estimated_up_heli_max_storms_10min.head())


            # --- Visualize future storm predictions (10-min intervals) ---
            print("\n--- Visualizing Future Storm Predictions (10-min intervals) ---")

            if not future_estimated_up_heli_max_storms_10min.empty:
                plt.figure(figsize=(12, 6))
                plt.plot(future_estimated_up_heli_max_storms_10min.index, future_estimated_up_heli_max_storms_10min.values, marker='o', linestyle='-', color='red')
                plt.title('Predicted Storm Intensity (Estimated UP_HELI_MAX) for Future Time Steps (10-min intervals)')
                plt.xlabel('Time')
                plt.ylabel('Estimated UP_HELI_MAX')
                plt.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
            else:
                print("No storms predicted in the future time steps to visualize.")

            # --- Display Accuracy Readings (10-min intervals) ---
            print("\n--- Accuracy Readings (LightGBM based on Percentiles for 10-min data) ---")
            print(f"Accuracy: {accuracy_score(y_test_intensity_perc_10min, y_pred_lgbm_intensity_10min):.2f}")
            print("\nClassification Report:")
            print(classification_report(y_test_intensity_perc_10min, y_pred_lgbm_intensity_10min))
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test_intensity_perc_10min, y_pred_lgbm_intensity_10min))
        else:
            print("Warning: 'UP_HELI_MAX' column not found in the reindexed DataFrame. Cannot prepare future data for prediction.")
    else:
        print("Warning: Cannot prepare future data for prediction as the DataFrame index is not a DatetimeIndex.")
else:
    print("Warning: 'UP_HELI_MAX' column not found. Cannot create multi-class target or proceed with modeling.")
