import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
import warnings
import os
import geopandas as gpd
from shapely.geometry import Point, box
import folium
from folium.plugins import HeatMap, TimestampedGeoJson
import requests
import zipfile

# Suppress FutureWarning from sklearn
warnings.filterwarnings("ignore", category=FutureWarning)

# Binary Tornado Hyperparameter Tuning Complete.
# Best parameters for Binary Tornado Model: {'classifier__learning_rate': 0.05, 'classifier__max_depth': -1, 'classifier__min_child_samples': 50, 'classifier__n_estimators': 100, 'classifier__num_leaves': 31, 'classifier__scale_pos_weight': np.float64(9888.0)}
# Best cross-validation F1-score for Binary Tornado Model (Positive Class): 0.3560
# Best binary tornado model (pipeline) stored

# --- Data Loading and Preprocessing / Deep Learning (Imputation) ---
def load_and_preprocess_data(file_path):
    """Loads, preprocesses data, creates time-based and lagged features, and imputes missing values, including AFWA_TORNADO."""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding='latin1', low_memory=False)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            print(f"Unsupported file format for {file_path}.")
            return None

        columns_to_drop_initial = ['Unnamed: 0']
        df = df.drop(columns=columns_to_drop_initial, errors='ignore')

        if 'XTIME' in df.columns:
            df['XTIME'] = pd.to_datetime(df['XTIME'])
            df = df.set_index('XTIME')
        else:
            print(f"Warning: 'XTIME' column not found in {file_path}. Cannot set as index.")
            return None

        # Include AFWA_TORNADO in the columns to keep, and drop others
        columns_to_keep = [col for col in df.columns if col not in ['afwa_tlyrbot', 'afwa_afwa_tlyrtop', 'afwa_turb', 'C1H', 'C2H', 'C1F', 'C2F', 'C3H', 'C4H', 'C3F', 'C4F', 'AFWA_TLYRBOT', 'AFWA_TLYRTOP', 'AFWA_TURB']]
        df = df[columns_to_keep]


        df = df.replace(-9.999000e+30, np.nan)

        numeric_cols = df.select_dtypes(include=np.number).columns
        df = df[numeric_cols]

        # Impute missing values using the defined function
        df = impute_missing_deep_learning_avg(df)

        if isinstance(df.index, pd.DatetimeIndex):
            df['hour_of_day'] = df.index.hour
            df['day_of_year'] = df.index.dayofyear

            if 'UP_HELI_MAX' in df.columns:
                df['UP_HELI_MAX_lag_1'] = df['UP_HELI_MAX'].shift(1)
                df['UP_HELI_MAX_lag_2'] = df['UP_HELI_MAX'].shift(2)
                df['UP_HELI_MAX_lag_3'] = df['UP_HELI_MAX'].shift(3)
                df['UP_HELI_MAX_rolling_mean_3'] = df['UP_HELI_MAX'].rolling(window=3).mean()
                df = df.dropna() # Drop NaNs introduced by lagging/rolling
                return df
            else:
                print(f"Warning: 'UP_HELI_MAX' column not found in {file_path}. Cannot create lagged features.")
                return None
        else:
            print(f"Warning: Time-based features and lagged features cannot be created for {file_path} as 'XTIME' was not set as index.")
            return None
    except Exception as e:
        print(f"Error loading or preprocessing data from {file_path}: {e}")
        return None

# --- Deep learning imputation (Averaging surrounding points) ---
def impute_missing_deep_learning_avg(df):
    """
    Imputes missing values in a DataFrame by averaging the two surrounding points
    for each column with NaNs. Handles edge cases with forward/backward fill.

    Args:
        df: pandas DataFrame with potential missing values.

    Returns:
        pandas DataFrame with missing values imputed.
    """
    df_imputed = df.copy()
    for col in df_imputed.columns:
        if df_imputed[col].isnull().any():
            # Interpolate using linear method (which averages surrounding non-NaNs)
            df_imputed[col] = df_imputed[col].interpolate(method='linear', limit_direction='both')

            # Handle any remaining NaNs (e.g., at the start/end or consecutive NaNs)
            # For simplicity and robustness in this case, we can use ffill and bfill
            df_imputed[col] = df_imputed[col].ffill()
            df_imputed[col] = df_imputed[col].bfill()

            # If still NaNs (e.g., column was all NaNs), fill with mean
            if df_imputed[col].isnull().any():
                df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mean())

    return df_imputed


# --- Multi-class Classification Target Creation (based on Percentiles and AFWA_TORNADO) ---
def categorize_storm_intensity_percentile_and_tornado(row, original_df_cleaned):
    """Categorizes storm intensity based on percentiles of non-zero UP_HELI_MAX and AFWA_TORNADO."""
    up_heli_max = row.get('UP_HELI_MAX', 0) # Default to 0 if column missing
    afwa_tornado = row.get('AFWA_TORNADO', 0) # Default to 0 if column missing

    # Check for tornado first
    if afwa_tornado > 0:
        return 5  # New class for Tornado Producing Storms

    # If no tornado, categorize based on UP_HELI_MAX percentiles (using original cleaned data for thresholds)
    if 'UP_HELI_MAX' in original_df_cleaned.columns:
        y_events_original = original_df_cleaned[original_df_cleaned['UP_HELI_MAX'] > 0]['UP_HELI_MAX']

        if not y_events_original.empty:
            threshold_light_percentile = y_events_original.quantile(0.50)
            threshold_moderate_percentile = y_events_original.quantile(0.75)
            threshold_heavy_percentile = y_events_original.quantile(0.90)

            if up_heli_max == 0:
                return 0  # No storm
            elif up_heli_max <= threshold_light_percentile:
                return 1  # Light storm
            elif up_heli_max <= threshold_moderate_percentile:
                return 2  # Moderate storm
            elif up_heli_max <= threshold_heavy_percentile:
                return 3  # Heavy storm
            else:
                return 4  # Very Heavy Storm
        else:
             return 0 # No storms found in original data (shouldn't happen if UP_HELI_MAX exists)
    else:
        # Fallback if UP_HELI_MAX is missing but AFWA_TORNADO was 0 or missing
        return 0 # Assume No Storm if no tornado and no UP_HELI_MAX data


# --- Training ---
# Function to train and evaluate model for a given horizon (individual locations)
def train_and_evaluate_horizon(df, shift_intervals, original_df_cleaned):
    """Trains and evaluates LightGBM model for a specific prediction horizon."""
    prediction_horizon_hours = int(shift_intervals * 10 / 60)
    print(f"\n--- Processing Prediction Horizon: {prediction_horizon_hours} hours ({shift_intervals} intervals) ---")

    df_shifted = df.copy()
    # Need to shift both UP_HELI_MAX and AFWA_TORNADO for future prediction target
    df_shifted['UP_HELI_MAX_future'] = df_shifted['UP_HELI_MAX'].shift(-shift_intervals)
    if 'AFWA_TORNADO' in df_shifted.columns:
         df_shifted['AFWA_TORNADO_future'] = df_shifted['AFWA_TORNADO'].shift(-shift_intervals)
    else:
         df_shifted['AFWA_TORNADO_future'] = 0 # Assume no tornado if column missing

    # Apply the updated categorization function
    y_intensity_future_target = df_shifted.apply(
        lambda row: categorize_storm_intensity_percentile_and_tornado(
            {'UP_HELI_MAX': row['UP_HELI_MAX_future'], 'AFWA_TORNADO': row.get('AFWA_TORNADO_future', 0)},
            original_df_cleaned # Pass the original cleaned df for percentiles
        ),
        axis=1
    )


    df_future_target = df_shifted.copy()
    df_future_target['y_intensity_future_target'] = y_intensity_future_target
    # Drop NaNs introduced by shifting the target columns
    df_future_target = df_future_target.dropna(subset=['y_intensity_future_target'])


    y_current_horizon = df_future_target['y_intensity_future_target']
    # Drop original target columns and the future target column from features
    columns_to_drop_for_X = ['UP_HELI_MAX', 'AFWA_TORNADO', 'UP_HELI_MAX_future', 'AFWA_TORNADO_future', 'y_intensity_future_target']
    X_current_horizon = df_future_target.drop(columns=columns_to_drop_for_X, errors='ignore')


    print(f"\nDistribution of Storm Intensity Categories for {prediction_horizon_hours}-hour future:")
    print(y_current_horizon.value_counts().sort_index())

    if X_current_horizon.empty or y_current_horizon.empty or len(np.unique(y_current_horizon)) < 2:
        print(f"Insufficient data or target classes for {prediction_horizon_hours}-hour prediction. Skipping.")
        return None, None, None, None, None, None

    # Check for classes with only one member before stratifying
    stratify_y = y_current_horizon
    value_counts = y_current_horizon.value_counts()
    # Filter out classes with only one sample for stratification
    classes_to_stratify = value_counts[value_counts >= 2].index.tolist()
    if len(classes_to_stratify) < len(np.unique(y_current_horizon)):
        print(f"Warning: Some classes in the target variable have less than 2 members. Skipping stratification for {prediction_horizon_hours}-hour prediction.")
        # If stratification is not possible for all classes, proceed without it
        X_train_current, X_test_current, y_train_current, y_test_current = train_test_split(
            X_current_horizon, y_current_horizon,
            test_size=0.2,
            random_state=42
        )
    else:
        # Filter the data to include only classes with 2 or more samples for stratification
        stratify_mask = y_current_horizon.isin(classes_to_stratify)
        X_current_horizon_stratify = X_current_horizon.loc[stratify_mask]
        y_current_horizon_stratify = y_current_horizon.loc[stratify_mask]

        X_train_current, X_test_current, y_train_current, y_test_current = train_test_split(
            X_current_horizon_stratify, y_current_horizon_stratify,
            test_size=0.2,
            random_state=42,
            stratify=y_current_horizon_stratify
        )


    # Fit preprocessor on the training data subset for this horizon
    # Ensure the preprocessor is fitted on a DataFrame with column names
    categorical_features_current = X_train_current.select_dtypes(include=['category', 'object']).columns
    numerical_features_current = X_train_current.select_dtypes(include=np.number).columns
    preprocessor_current = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features_current),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_current)
            ],
            remainder='passthrough'
        )
    preprocessor_current.fit(X_train_current)


    X_train_current_scaled = preprocessor_current.transform(X_train_current)
    X_test_current_scaled = preprocessor_current.transform(X_test_current)

    lgbm_current_classifier = LGBMClassifier(objective='multiclass',
                                            num_class=len(np.unique(y_train_current)),
                                            metric='multi_logloss',
                                            random_state=42,
                                            n_jobs=-1)

    lgbm_current_classifier.fit(X_train_current_scaled, y_train_current)

    y_pred_current = lgbm_current_classifier.predict(X_test_current_scaled)

    evaluation_metrics = {
        'accuracy': accuracy_score(y_test_current, y_pred_current),
        'classification_report': classification_report(y_test_current, y_pred_current, zero_division=0), # Set zero_division to 0
        'confusion_matrix': confusion_matrix(y_test_current, y_pred_current).tolist()
    }

    # Return the fitted preprocessor for this horizon
    return evaluation_metrics, lgbm_current_classifier, preprocessor_current, X_current_horizon.columns, df_future_target.index.max(), df_future_target # Return last timestamp and df_future_target


# --- Combined model training and evaluation (Multi-class) ---
# Initialize an empty list to store the processed DataFrames from each location
processed_dfs = {} # Use dictionary to store processed dfs by location name

# Iterate through the file_paths list (assuming file_paths is defined in a previous cell)
file_paths = [
    "/content/KAMA_wrf_2d_May_Aug1998.parquet",
    "/content/KDFW_wrf_2d_May_Aug1998.parquet",
    "/content/KHOU_wrf_2d_May_Aug1998.parquet",
    "/content/KLBB_wrf_2d_May_Aug1998.parquet",
    "/content/KSAT_wrf_2d_May_Aug1998.parquet"
]
prediction_horizons_intervals = [18, 36, 72] # 3 hours, 6 hours, 12 hours

processed_dfs_list = [] # List for concatenation
original_dfs_cleaned = {} # Store original cleaned dfs for percentile calculation

for file_path in file_paths:
    location_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"\n--- Loading and preprocessing data for {location_name} from: {file_path} ---")

    # Load and preprocess data with imputation and feature engineering, including AFWA_TORNADO
    df_processed = load_and_preprocess_data(file_path)

    if df_processed is not None:
        print(f"Data loaded and preprocessed for {location_name}.")
        # Add a new column named 'location'
        df_processed['location'] = location_name
        # Store processed df in the dictionary
        processed_dfs[location_name] = df_processed
        # Append to list for combined data
        processed_dfs_list.append(df_processed)

        # Store original cleaned df for percentile calculations
        # Reload is not ideal, but ensures we have a version for percentiles if preprocessing changes
        original_dfs_cleaned[location_name] = load_and_preprocess_data(file_path)

    else:
        print(f"Failed to process data for {location_name}. Skipping.")

# Concatenate all DataFrames in the list into a single DataFrame
if processed_dfs_list:
    df_combined = pd.concat(processed_dfs_list)

    # Ensure the 'location' column in df_combined is of type 'category'.
    df_combined['location'] = df_combined['location'].astype('category')
    print("'location' column converted to category type.")

    # --- Manual One-Hot Encoding for 'location' before splitting ---
    if 'location' in df_combined.columns:
        df_combined = pd.get_dummies(df_combined, columns=['location'], prefix='location', dummy_na=False)
        print("Manually one-hot encoded 'location' column.")
        # Display head with new columns
        print("\n--- Combined DataFrame after One-Hot Encoding ---")
        display(df_combined.head())


    # Display the head of the df_combined DataFrame and its shape
    print(f"\nShape of combined DataFrame: {df_combined.shape}")

    # Create the combined target variable using the updated categorization function
    # Need to apply row-wise as categorize_storm_intensity_percentile_and_tornado expects a row
    if 'UP_HELI_MAX' in df_combined.columns and 'AFWA_TORNADO' in df_combined.columns:
         # Pass the entire df_combined for percentile calculations as it's already processed
        y_intensity_combined = df_combined.apply(
            lambda row: categorize_storm_intensity_percentile_and_tornado(row, df_combined),
            axis=1
        )
        print("Combined target variable 'y_intensity_combined' created with tornado class.")
    else:
        print("Warning: 'UP_HELI_MAX' or 'AFWA_TORNADO' column not found in df_combined. Cannot define combined target with tornado class.")
        y_intensity_combined = None
        X_combined = None
        # Skip further steps if target cannot be defined

    if y_intensity_combined is not None and not y_intensity_combined.empty:
        # Define the feature set X_combined by dropping target and related columns.
        # Include original UP_HELI_MAX and AFWA_TORNADO in columns to drop for features
        # Also drop future target columns if they exist from previous operations in the same cell run
        columns_to_drop_for_X = ['UP_HELI_MAX', 'AFWA_TORNADO', 'UP_HELI_MAX_future', 'AFWA_TORNADO_future', 'y_intensity_future_target', 'y_intensity_combined']
        X_combined = df_combined.drop(columns=columns_to_drop_for_X, errors='ignore')
        print("Feature set 'X_combined' created.")

        # Identify numerical features - 'location' encoded columns are now numerical
        numerical_features = X_combined.select_dtypes(include=np.number).columns
        # No separate categorical features needed for the ColumnTransformer after manual encoding

        # Split X_combined and y_intensity_combined into training and testing sets.
        if not X_combined.empty and not y_intensity_combined.empty and len(np.unique(y_intensity_combined)) >= 2:
            # Check for classes with only one member before stratifying
            stratify_y = y_intensity_combined
            value_counts = y_intensity_combined.value_counts()
            # Filter out classes with only one sample for stratification
            classes_to_stratify = value_counts[value_counts >= 2].index.tolist()

            if len(classes_to_stratify) < len(np.unique(y_intensity_combined)):
                 print(f"Warning: Some classes in the combined target variable have less than 2 members. Skipping stratification.")
                 # If stratification is not possible for all classes, proceed without it
                 X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(
                    X_combined, y_intensity_combined,
                    test_size=0.2,
                    random_state=46 # Changed random state to potentially get a split with all categories
                 )
            else:
                # Filter the data to include only classes with 2 or more samples for stratification
                stratify_mask = y_intensity_combined.isin(classes_to_stratify)
                X_combined_stratify = X_combined.loc[stratify_mask]
                y_intensity_combined_stratify = y_intensity_combined.loc[stratify_mask]

                X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(
                    X_combined_stratify, y_intensity_combined_stratify,
                    test_size=0.2,
                    random_state=46,
                    stratify=y_intensity_combined_stratify
                )

            print("Data split into training and testing sets for multi-class model.")
            print(f"Training set shape: {X_train_combined.shape}, {y_train_combined.shape}")
            print(f"Testing set shape: {X_test_combined.shape}, {y_test_combined.shape}")
            print("\nDistribution of Storm Intensity Categories in combined training set:")
            print(y_train_combined.value_counts().sort_index())
            print("\nDistribution of Storm Intensity Categories in combined testing set:")
            print(y_test_combined.value_counts().sort_index())

            # Create and Fit the preprocessor *after* the split, on training data
            # Now only scaling numerical features
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_features)
                ],
                remainder='passthrough' # Keep the one-hot encoded location columns
            )
            preprocessor.fit(X_train_combined) # Fit only on training data
            print("Preprocessor fitted on the combined multi-class training dataset (scaling numerical features).")


            # Transform the training and testing sets using the fitted preprocessor.
            X_train_combined_processed = preprocessor.transform(X_train_combined)
            X_test_combined_processed = preprocessor.transform(X_test_combined)
            print("Numerical features scaled for multi-class training and testing sets.")
            print(f"Processed multi-class training set shape: {X_train_combined_processed.shape}")
            print(f"Processed multi-class testing set shape: {X_test_combined_processed.shape}")


            # Initialize a LGBMClassifier.
            # Train the LGBMClassifier.
            lgbm_combined_classifier = LGBMClassifier(objective='multiclass',
                                                    num_class=len(np.unique(y_train_combined)), # Use unique classes in training data
                                                    metric='multi_logloss',
                                                    random_state=42,
                                                    n_jobs=-1)

            print("\nTraining LightGBM multi-class model (with Tornado Class)...")
            lgbm_combined_classifier.fit(X_train_combined_processed, y_train_combined)
            print("LightGBM multi-class model trained.")

            # Predict storm intensity on the scaled testing data.
            y_pred_combined = lgbm_combined_classifier.predict(X_test_combined_processed)
            print("Predictions made on multi-class testing data.")

            # Evaluate the model's performance.
            print("\n--- Evaluation Metrics for Combined Multi-class Model (with Tornado Class) ---")
            accuracy_combined = accuracy_score(y_test_combined, y_pred_combined)
            classification_report_combined = classification_report(y_test_combined, y_pred_combined, zero_division=0)

            print(f"Accuracy: {accuracy_combined:.2f}")
            print("Classification Report:")
            print(classification_report_combined)
            print("Confusion Matrix:")
            display(confusion_matrix(y_test_combined, y_pred_combined))


            # Store the trained model, preprocessor, and original training columns.
            # Store the columns *after* manual one-hot encoding for prediction
            combined_model_artifacts = {
                'model': lgbm_combined_classifier,
                'preprocessor': preprocessor, # This preprocessor only scales numerical features
                'columns': X_combined.columns.tolist() # Store columns after manual encoding
            }
            print("\nTrained multi-class model, preprocessor, and feature columns (after manual encoding) stored.")
        else:
             print("Insufficient data or target classes for combined multi-class model training. Skipping training and evaluation.")
             combined_model_artifacts = None


    else:
        print("Combined data or target variable not available. Cannot train combined multi-class model.")
        combined_model_artifacts = None


    # --- Prepare Data for Binary Tornado Prediction ---
    print("\n--- Preparing Data for Binary Tornado Prediction ---")

    # Create the binary target variable for tornado prediction
    if 'AFWA_TORNADO' in df_combined.columns:
        y_tornado = (df_combined['AFWA_TORNADO'] > 0).astype(int)
        print("Binary target variable 'y_tornado' created.")

        # Define the feature set X_tornado by dropping target and related columns.
        # We'll use the same features as the multi-class model (X_combined) but ensure AFWA_TORNADO is not included
        columns_to_drop_for_X_tornado = ['AFWA_TORNADO'] # Only need to explicitly drop AFWA_TORNADO if it wasn't dropped for X_combined
        X_tornado = X_combined.drop(columns=columns_to_drop_for_X_tornado, errors='ignore') # Use X_combined as base

        print("Feature set 'X_tornado' created.")


        # Check the distribution of the binary target variable
        print("\nDistribution of Tornado Occurrence (Binary Target):")
        print(y_tornado.value_counts())

        # Split the data into training and testing sets for the binary model
        if not X_tornado.empty and not y_tornado.empty and len(np.unique(y_tornado)) >= 2:
            # Stratify the split to maintain the rare tornado class distribution
            X_train_tornado, X_test_tornado, y_train_tornado, y_test_tornado = train_test_split(
                X_tornado, y_tornado,
                test_size=0.2,
                random_state=42,
                stratify=y_tornado # Stratify based on the binary tornado target
            )
            print("\nData split into training and testing sets for binary tornado model.")
            print(f"Binary Training set shape: {X_train_tornado.shape}, {y_train_tornado.shape}")
            print(f"Binary Testing set shape: {X_test_tornado.shape}, {y_test_tornado.shape}")
            print("\nDistribution of Tornado Occurrence in binary training set:")
            print(y_train_tornado.value_counts())
            print("\nDistribution of Tornado Occurrence in binary testing set:")
            print(y_test_tornado.value_counts())

            # --- Train and Evaluate Binary Tornado Prediction Model ---
            print("\n--- Training and Evaluating Binary Tornado Prediction Model ---")

            # Identify numerical features for the binary model - 'location' encoded columns are numerical
            numerical_features_tornado = X_train_tornado.select_dtypes(include=np.number).columns
            # No separate categorical features needed for the ColumnTransformer after manual encoding


            # Create and Fit a column transformer for scaling numerical features for the binary model *after* the split
            preprocessor_tornado = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_features_tornado)
                ],
                remainder='passthrough' # Keep the one-hot encoded location columns
            )

            preprocessor_tornado.fit(X_train_tornado) # Fit only on training data
            print("Preprocessor fitted on the binary tornado training data (scaling numerical features).")


            # Transform the training and testing sets
            X_train_tornado_processed = preprocessor_tornado.transform(X_train_tornado)
            X_test_tornado_processed = preprocessor_tornado.transform(X_test_tornado)
            print("Numerical features scaled for binary tornado model.")
            print(f"Processed binary training set shape: {X_train_tornado_processed.shape}")
            print(f"Processed binary testing set shape: {X_test_tornado_processed.shape}")


            # Initialize a LGBMClassifier for binary classification
            # Use objective='binary' and metric='binary_logloss'
            # Consider scale_pos_weight to handle class imbalance (optional, but can help)
            # Calculate scale_pos_weight: count(negative class) / count(positive class)
            scale_pos_weight_value = (y_train_tornado == 0).sum() / (y_train_tornado == 1).sum() if (y_train_tornado == 1).sum() > 0 else 1
            print(f"\nUsing scale_pos_weight: {scale_pos_weight_value}")


            lgbm_tornado_classifier = LGBMClassifier(objective='binary',
                                                    metric='binary_logloss',
                                                    random_state=42,
                                                    n_jobs=-1,
                                                    scale_pos_weight=scale_pos_weight_value # Add scale_pos_weight
                                                    )

            print("\nTraining Binary LightGBM model for tornado prediction...")
            lgbm_tornado_classifier.fit(X_train_tornado_processed, y_train_tornado)
            print("Binary LightGBM model trained.")

            # Predict tornado occurrence on the scaled testing data
            y_pred_tornado = lgbm_tornado_classifier.predict(X_test_tornado_processed)
            print("Predictions made on binary testing data.")

            # Evaluate the model's performance
            print("\n--- Evaluation Metrics for Binary Tornado Prediction Model ---")
            accuracy_tornado = accuracy_score(y_test_tornado, y_pred_tornado)
            classification_report_tornado = classification_report(y_test_tornado, y_pred_tornado, zero_division=0) # Set zero_division to 0
            confusion_matrix_tornado = confusion_matrix(y_test_tornado, y_pred_tornado)
            f1_score_tornado = f1_score(y_test_tornado, y_pred_tornado, zero_division=0) # F1-score is good for imbalanced data

            print(f"Accuracy: {accuracy_tornado:.2f}")
            print(f"F1-Score: {f1_score_tornado:.2f}")
            print("Classification Report:")
            print(classification_report_tornado)
            print("Confusion Matrix:")
            display(confusion_matrix_tornado)

            # Store the trained binary model and preprocessor
            binary_tornado_model_artifacts = {
                'model': lgbm_tornado_classifier,
                'preprocessor': preprocessor_tornado, # This preprocessor only scales numerical features
                'columns': X_tornado.columns.tolist() # Store columns after manual encoding
            }
            print("\nTrained binary tornado model and preprocessor stored.")


        else:
             print("Insufficient data or target classes for binary tornado model training. Skipping training and evaluation.")
             binary_tornado_model_artifacts = None


    else:
        print("Warning: 'AFWA_TORNADO' column not found in df_combined. Cannot prepare data for binary tornado prediction.")
        X_train_tornado, X_test_tornado, y_train_tornado, y_test_tornado = None, None, None, None
        binary_tornado_model_artifacts = None


else:
    print("Combined DataFrame (df_combined) not available. Please run the combined data loading and preprocessing cell first.")
    X_train_tornado, X_test_tornado, y_train_tornado, y_test_tornado = None, None, None, None
    binary_tornado_model_artifacts = None


# --- Feature importance analysis (Overall with Class Context) ---
# This section analyzes feature importance for the MULTI-CLASS model.
if 'combined_model_artifacts' in locals() and combined_model_artifacts is not None:
    print("\n--- Analyzing Feature Importance from Combined Multi-class Model (Overall with Class Context) ---")

    lgbm_combined_classifier = combined_model_artifacts['model']
    # Need to get the feature names from the multi-class preprocessor's fitted transformer
    combined_preprocessor_fitted = combined_model_artifacts['preprocessor']


    # Get feature names after preprocessing using the ColumnTransformer's method
    try:
        # Use the get_feature_names_out from the fitted preprocessor
        feature_names_after_preprocessing = combined_preprocessor_fitted.get_feature_names_out()
    except Exception as e:
        print(f"Error getting feature names from multi-class preprocessor: {e}")
        # Fallback to generic names if getting processed names fails
        if hasattr(lgbm_combined_classifier, 'n_features_'):
             feature_names_after_preprocessing = [f'feature_{i}' for i in range(lgbm_combined_classifier.n_features_)]
        else:
             feature_names_after_preprocessing = [f'feature_{i}' for i in range(len(combined_model_artifacts['columns']))] # Use original columns count as fallback
        print("Using generic feature names for feature importance due to error.")


    # Get overall feature importances from the trained model
    feature_importances = lgbm_combined_classifier.feature_importances_

    # Check if the number of feature importances matches the number of feature names
    if len(feature_importances) != len(feature_names_after_preprocessing):
        print(f"Warning: Mismatch between number of feature importances ({len(feature_importances)}) and generated feature names ({len(feature_names_after_preprocessing)}) for multi-class model.")
        # If there's a mismatch, it's hard to correctly label the importances.
        # We can still show the importances with generic names as a last resort.
        feature_names_after_preprocessing = [f'feature_{i}' for i in range(len(feature_importances))]
        print("Attempting to display with generic feature names due to mismatch.")


    # Create a DataFrame to store feature importance
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names_after_preprocessing,
        'Importance': feature_importances
    })

    # Sort features by importance in descending order
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Display the top N features
    top_n = 20 # Display top 20 features
    print(f"\nTop {top_n} Overall Feature Importances for Multi-class Model:")
    display(feature_importance_df.head(top_n))

    # Visualize feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(top_n))
    plt.title(f'Top {top_n} Overall Feature Importances from Combined Multi-class Model')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

    # Display the classification report again for performance context
    if 'classification_report_combined' in locals():
         print("\nClassification Report (for multi-class model performance context):")
         print(classification_report_combined)
    else:
         print("\nMulti-class classification report not available. Please ensure multi-class model training was successful.")


else:
    print("Combined multi-class model artifacts are not available. Cannot perform feature importance analysis for multi-class model.")


# --- Feature importance analysis for Binary Tornado Model ---
if 'binary_tornado_model_artifacts' in locals() and binary_tornado_model_artifacts is not None:
    print("\n--- Analyzing Feature Importance from Binary Tornado Model ---")

    lgbm_tornado_classifier_binary = binary_tornado_model_artifacts['model']
    tornado_preprocessor_binary = binary_tornado_model_artifacts['preprocessor']

    # Get feature names after preprocessing using the ColumnTransformer's method
    try:
        feature_names_after_preprocessing_tornado = tornado_preprocessor_binary.get_feature_names_out()
    except Exception as e:
        print(f"Error getting feature names from binary preprocessor: {e}")
        if hasattr(lgbm_tornado_classifier_binary, 'n_features_'):
             feature_names_after_preprocessing_tornado = [f'feature_{i}' for i in range(lgbm_tornado_classifier_binary.n_features_)]
        else:
             feature_names_after_preprocessing_tornado = [f'feature_{i}' for i in range(len(binary_tornado_model_artifacts['columns']))] # Use original columns count as fallback
        print("Using generic feature names for feature importance due to error.")


    # Get overall feature importances from the trained binary model
    feature_importances_tornado = lgbm_tornado_classifier_binary.feature_importances_

    # Check if the number of feature importances matches the number of feature names
    if len(feature_importances_tornado) != len(feature_names_after_preprocessing_tornado):
        print(f"Warning: Mismatch between number of feature importances ({len(feature_importances_tornado)}) and generated feature names ({len(feature_names_after_preprocessing_tornado)}) for binary model.")
        # If there's a mismatch, it's hard to correctly label the importances.
        # We can still show the importances with generic names as a last resort.
        feature_names_after_preprocessing_tornado = [f'feature_{i}' for i in range(len(feature_importances_tornado))]
        print("Attempting to display with generic feature names due to mismatch.")


    # Create a DataFrame to store feature importance for the binary model
    feature_importance_df_tornado = pd.DataFrame({
        'Feature': feature_names_after_preprocessing_tornado,
        'Importance': feature_importances_tornado
    })

    # Sort features by importance in descending order
    feature_importance_df_tornado = feature_importance_df_tornado.sort_values(by='Importance', ascending=False)

    # Display the top N features for the binary model
    top_n_tornado = 20 # Display top 20 features
    print(f"\nTop {top_n_tornado} Feature Importances for Binary Tornado Model:")
    display(feature_importance_df_tornado.head(top_n_tornado))

    # Visualize feature importance for the binary model
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df_tornado.head(top_n_tornado))
    plt.title(f'Top {top_n_tornado} Feature Importances from Binary Tornado Model')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()

    # Display the classification report again for performance context
    if 'classification_report_tornado' in locals():
         print("\nClassification Report (for binary tornado model performance context):")
         print(classification_report_tornado)
    else:
         print("\nBinary tornado classification report not available. Please ensure binary model training was successful.")


else:
    print("Binary tornado model artifacts are not available. Cannot perform feature importance analysis for binary model.")


# --- Pre-storm detection (Future Prediction and Visualization) ---
# This section will now use BOTH models for prediction and combine their outputs for visualization.

# Function to prepare future data for prediction (modified to return X_future for BOTH models)
def prepare_future_data_for_both_models(df, combined_preprocessor_fitted, binary_tornado_preprocessor_fitted, combined_original_columns, binary_tornado_original_columns, last_timestamp):
    """Prepares future data for prediction using both multi-class and binary tornado preprocessors."""
    future_start_time = last_timestamp + pd.Timedelta(minutes=10)
    future_end_time = future_start_time + pd.Timedelta(hours=24)
    future_time_index = pd.date_range(start=future_start_time, end=future_end_time, freq='10min')

    # Ensure unique timestamps in the future index
    future_time_index = future_time_index[~future_time_index.duplicated()]

    # Create a future DataFrame with the desired index and columns from the original df, initially filled with NaNs
    # Need original columns plus the manually encoded location columns
    original_cols_plus_encoded_location = [col for col in df.columns if not col.startswith('location_')]
    location_encoded_cols = [col for col in df.columns if col.startswith('location_')]
    cols_for_future_df = original_cols_plus_encoded_location + location_encoded_cols # Combine original and encoded location cols


    df_future = pd.DataFrame(index=future_time_index, columns=cols_for_future_df)


    # Concatenate the original DataFrame with the future DataFrame
    df_combined_future = pd.concat([df, df_future])

    # Sort by index to ensure correct time series order
    df_combined_future = df_combined_future.sort_index()

    if not isinstance(df_combined_future.index, pd.DatetimeIndex):
        print("Warning: Combined DataFrame index is not a DatetimeIndex. Cannot prepare future data.")
        return None, None, None, None

    # Impute missing values in the combined DataFrame.
    df_combined_future = impute_missing_deep_learning_avg(df_combined_future)

    # --- Debugging: Check for NaNs/Infs after imputation in df_combined_future ---
    print("\n--- Debugging: Checking for NaNs/Infs in df_combined_future after imputation ---")
    nan_check = df_combined_future.isnull().sum()
    inf_check = np.isinf(df_combined_future.select_dtypes(include=np.number)).sum()

    if nan_check.sum() > 0:
        print("NaNs found in the following columns after imputation:")
        print(nan_check[nan_check > 0])
    else:
        print("No NaNs found in df_combined_future after imputation.")

    if inf_check.sum() > 0:
        print("Infs found in the following numerical columns after imputation:")
        print(inf_check[inf_check > 0])
    else:
        print("No Infs found in numerical columns of df_combined_future after imputation.")
    print("--- End Debugging Check ---")


    # Recalculate time-based and lagged features on the combined data
    if isinstance(df_combined_future.index, pd.DatetimeIndex):
        df_combined_future['hour_of_day'] = df_combined_future.index.hour
        df_combined_future['day_of_year'] = df_combined_future.index.dayofyear

        # Lagged features for UP_HELI_MAX
        if 'UP_HELI_MAX' in df_combined_future.columns:
             df_combined_future['UP_HELI_MAX_lag_1'] = df_combined_future['UP_HELI_MAX'].shift(1)
             df_combined_future['UP_HELI_MAX_lag_2'] = df_combined_future['UP_HELI_MAX'].shift(2)
             # Corrected typo here
             df_combined_future['UP_HELI_MAX_lag_3'] = df_combined_future['UP_HELI_MAX'].shift(3)
             df_combined_future['UP_HELI_MAX_rolling_mean_3'] = df_combined_future['UP_HELI_MAX'].rolling(window=3).mean()
        else:
             # Add placeholder columns filled with NaNs if UP_HELI_MAX is not available for lagging
             df_combined_future['UP_HELI_MAX_lag_1'] = np.nan
             df_combined_future['UP_HELI_MAX_lag_2'] = np.nan
             df_combined_future['UP_HELI_MAX_lag_3'] = np.nan
             df_combined_future['UP_HELI_MAX_rolling_mean_3'] = np.nan


        # Impute NaNs introduced by lagging/rolling in the combined data
        df_combined_future = impute_missing_deep_learning_avg(df_combined_future)

        # --- Debugging: Check for NaNs/Infs after lagging/rolling imputation ---
        print("\n--- Debugging: Checking for NaNs/Infs in df_combined_future after lagging/rolling imputation ---")
        nan_check_lag = df_combined_future.isnull().sum()
        inf_check_lag = np.isinf(df_combined_future.select_dtypes(include=np.number)).sum()

        if nan_check_lag.sum() > 0:
            print("NaNs found in the following columns after lagging/rolling imputation:")
            print(nan_check_lag[nan_check_lag > 0])
        else:
            print("No NaNs found in df_combined_future after lagging/rolling imputation.")

        if inf_check_lag.sum() > 0:
            print("Infs found in the following numerical columns after lagging/rolling imputation:")
            print(inf_check_lag[inf_check_lag > 0])
        else:
            print("No Infs found in numerical columns of df_combined_future after lagging/rolling imputation.")
        print("--- End Debugging Check ---")


        # Select only the future time index rows from the combined DataFrame
        df_future_subset = df_combined_future.loc[future_time_index].copy()

        # Prepare X for the Multi-class model - drop original target columns
        cols_to_drop_for_multi_X = ['UP_HELI_MAX', 'AFWA_TORNADO'] # Drop original targets from features
        X_future_multi = df_future_subset.drop(columns=cols_to_drop_for_multi_X, errors='ignore')
        # Align columns with the multi-class model's training columns (which include encoded location)
        X_future_multi = X_future_multi.reindex(columns=combined_original_columns, fill_value=0)


        # Prepare X for the Binary Tornado model - drop original target columns
        cols_to_drop_for_binary_X = ['UP_HELI_MAX', 'AFWA_TORNADO'] # Drop original targets from features
        X_future_binary = df_future_subset.drop(columns=cols_to_drop_for_binary_X, errors='ignore')
         # Align columns with the binary tornado model's training columns (which include encoded location)
        X_future_binary = X_future_binary.reindex(columns=binary_tornado_original_columns, fill_value=0)


        # Transform using the respective fitted preprocessors
        X_future_multi_scaled = None
        if combined_preprocessor_fitted is not None:
            try:
                # Ensure the input to transform is a DataFrame with the expected column names
                X_future_multi_scaled = combined_preprocessor_fitted.transform(X_future_multi)
                # It returns a sparse matrix or numpy array, convert back to DataFrame for consistency if needed,
                # but transform output is usually ready for model.predict
                # For now, keep it as is if it's numpy/sparse as model.predict handles it.

            except ValueError as e:
                 print(f"Error transforming future data for multi-class model: {e}")

        X_future_binary_scaled = None
        if binary_tornado_preprocessor_fitted is not None:
            try:
                # Ensure the input to transform is a DataFrame with the expected column names
                X_future_binary_scaled = binary_tornado_preprocessor_fitted.transform(X_future_binary)
                 # It returns a sparse matrix or numpy array, convert back to DataFrame for consistency if needed
            except ValueError as e:
                 print(f"Error transforming future data for binary tornado model: {e}")


        return X_future_multi_scaled, X_future_binary_scaled, future_time_index, df_combined_future

    else:
        print("Warning: Time-based features and lagged features cannot be created for future data as index is not DatetimeIndex.")
        return None, None, None, None


# Function to predict future storm intensity using BOTH models
def predict_future_storms_with_both_models(lgbm_multi_classifier, combined_preprocessor_fitted, lgbm_tornado_classifier, binary_tornado_preprocessor_fitted, df, location_name, prediction_horizons_intervals, combined_model_artifacts, binary_tornado_model_artifacts):
    """Predicts future storm intensity and tornado occurrence using both models for a given location and horizons."""
    if (lgbm_multi_classifier is None or combined_preprocessor_fitted is None or
        lgbm_tornado_classifier is None or binary_tornado_preprocessor_fitted is None or
        df is None or combined_model_artifacts is None or binary_tornado_model_artifacts is None):
        print(f"Skipping future prediction for {location_name} due to missing inputs.")
        return None

    predicted_storm_data_location = {}

    for shift_intervals in prediction_horizons_intervals:
        prediction_horizon_hours = int(shift_intervals * 10 / 60)
        horizon_key = f"{prediction_horizon_hours} hours"
        print(f"\nPredicting future storms and tornado occurrence for {location_name} at {horizon_key} horizon...")

        # Prepare future data using both preprocessors
        # Pass the fitted preprocessors
        X_future_multi_scaled, X_future_binary_scaled, future_time_index, df_reindexed_future = prepare_future_data_for_both_models(
            df.copy(), # Pass a copy
            combined_preprocessor_fitted, # Pass fitted preprocessor
            binary_tornado_preprocessor_fitted, # Pass fitted preprocessor
            combined_model_artifacts['columns'], # Multi-class original columns (after manual encoding)
            binary_tornado_model_artifacts['columns'], # Binary original columns (after manual encoding)
            df.index.max() # Use the last timestamp of the current location's processed data
        )

        if (X_future_multi_scaled is not None and X_future_binary_scaled is not None and
            future_time_index is not None and df_reindexed_future is not None):

            # Predict using the multi-class model
            future_intensity_predictions = lgbm_multi_classifier.predict(X_future_multi_scaled)
            predicted_intensity_series = pd.Series(future_intensity_predictions, index=future_time_index)

            # Predict using the binary tornado model
            future_tornado_predictions = lgbm_tornado_classifier.predict(X_future_binary_scaled)
            predicted_tornado_series = pd.Series(future_tornado_predictions, index=future_time_index)

            # Combine the predictions: If binary model predicts tornado (1), override multi-class prediction to class 5
            final_predicted_classes = predicted_intensity_series.copy()
            final_predicted_classes[predicted_tornado_series == 1] = 5 # Assign class 5 if tornado predicted

            # Filter for predicted storm instances (classes > 0)
            predicted_storm_instances = final_predicted_classes[final_predicted_classes > 0]


            print(f"Number of predicted storm instances (all types > 0) in future for {location_name} at {horizon_key}: {predicted_storm_instances.shape[0]}")
            if 5 in predicted_storm_instances.values:
                 print(f"Number of predicted Tornado Producing Storm instances (class 5) for {location_name} at {horizon_key}: {(predicted_storm_instances == 5).sum()}")

            # Store the final combined predicted classes
            predicted_storm_data_location[horizon_key] = final_predicted_classes

        else:
            print(f"Failed to prepare future data for {location_name} at {horizon_key}. Skipping prediction.")

    return predicted_storm_data_location


# --- Main Execution for Future Prediction and Visualization ---

all_predicted_storm_data = {} # To store predicted storm data for visualization

# Check if both models were successfully trained
if ('combined_model_artifacts' in locals() and combined_model_artifacts is not None and
    'binary_tornado_model_artifacts' in locals() and binary_tornado_model_artifacts is not None and
    'processed_dfs' in locals() and processed_dfs):

    lgbm_combined_classifier = combined_model_artifacts['model']
    combined_preprocessor_fitted = combined_model_artifacts['preprocessor'] # Get the fitted preprocessor
    lgbm_tornado_classifier = binary_tornado_model_artifacts['model']
    binary_tornado_preprocessor_fitted = binary_tornado_model_artifacts['preprocessor'] # Get the fitted preprocessor


    for location_name, df in processed_dfs.items():
        # Predict future storm intensity and tornado occurrence for this location and horizons using both models
        predicted_storm_data_location = predict_future_storms_with_both_models(
            lgbm_combined_classifier,
            combined_preprocessor_fitted, # Pass fitted preprocessor
            lgbm_tornado_classifier,
            binary_tornado_preprocessor_fitted, # Pass fitted preprocessor
            df,
            location_name,
            prediction_horizons_intervals,
            combined_model_artifacts, # Pass combined_model_artifacts
            binary_tornado_model_artifacts # Pass binary_tornado_model_artifacts
        )
        if predicted_storm_data_location:
             all_predicted_storm_data[location_name] = predicted_storm_data_location

    # --- Interactive Geospatial Visualization with Folium ---

    # Define the geographical coordinates for each of the five Texas locations
    location_coordinates = {
        'KAMA_wrf_2d_May_Aug1998': {'latitude': 35.2197, 'longitude': -100.9993}, # Amarillo, TX
        'KDFW_wrf_2d_May_Aug1998': {'latitude': 32.8998, 'longitude': -97.0403},  # Dallas/Fort Worth, TX
        'KHOU_wrf_2d_May_Aug1998': {'latitude': 29.7604, 'longitude': -95.3698},  # Houston, TX
        'KLBB_wrf_2d_May_Aug1998': {'latitude': 33.5779, 'longitude': -101.8552}, # Lubbock, TX
        'KSAT_wrf_2d_May_Aug1998': {'latitude': 29.4241, 'longitude': -98.4935}   # San Antonio, TX
    }

    # Approximate center of Texas for map initialization
    texas_center = [31.9686, -99.9018]

    # Create a base map of Texas
    m = folium.Map(location=texas_center, zoom_start=6)

    # Define color map or discrete colors for intensity levels, including the new tornado class
    colors = {0: 'blue', 1: 'green', 2: 'yellow', 3: 'orange', 4: 'red', 5: 'purple'} # Added purple for tornado
    intensity_labels = {0: 'No Storm', 1: 'Light', 2: 'Moderate', 3: 'Heavy', 4: 'Very Heavy', 5: 'Tornado Producing'} # Added label for tornado


    # Create a FeatureGroup for each prediction horizon to allow toggling
    horizon_groups = {}
    for shift_intervals in prediction_horizons_intervals:
        prediction_horizon_hours = int(shift_intervals * 10 / 60)
        horizon_key = f"{prediction_horizon_hours} hours"
        horizon_groups[horizon_key] = folium.FeatureGroup(name=horizon_key)
        m.add_child(horizon_groups[horizon_key])


    # Add markers for each location with predicted storm intensity for each horizon
    for location_name, coords in location_coordinates.items():
        for horizon_key, predicted_classes_series in all_predicted_storm_data.get(location_name, {}).items():
            if not predicted_classes_series.empty:
                # For visualization on a static point, use the maximum predicted intensity (including tornado class)
                max_intensity = predicted_classes_series.max()

                folium.CircleMarker(
                    location=[coords['latitude'], coords['longitude']],
                    radius=max_intensity * 5 + 5, # Adjust radius based on intensity (larger for tornado)
                    color=colors.get(max_intensity, 'gray'),
                    fill=True,
                    fill_color=colors.get(max_intensity, 'gray'),
                    fill_opacity=0.7,
                    tooltip=f"{location_name.replace('_wrf_2d_May_Aug1998', '')}<br>Predicted Intensity ({horizon_key}): {intensity_labels.get(max_intensity, 'Unknown')}"
                ).add_to(horizon_groups[horizon_key])
            else:
                 # Add a marker for locations with no predicted storms for this horizon
                folium.CircleMarker(
                    location=[coords['latitude'], coords['longitude']],
                    radius=5,
                    color=colors.get(0, 'blue'),
                    fill=True,
                    fill_color=colors.get(0, 'blue'),
                    fill_opacity=0.7,
                    tooltip=f"{location_name.replace('_wrf_2d_May_Aug1998', '')}<br>Predicted Intensity ({horizon_key}): {intensity_labels.get(0, 'Unknown')}"
                ).add_to(horizon_groups[horizon_key])


    # Add a layer control to toggle between horizons
    folium.LayerControl().add_to(m)

    # Display the map
    display(m)

else:
    print("Combined or binary tornado model artifacts or processed data not available. Skipping future prediction and visualization.")


# --- Prepare data for time-sensitive visualization (Historical) ---
# 1. Create an empty list to store the data points for visualization.
historical_visualization_data = []

# Define the geographical coordinates for each of the five Texas locations (assuming this is already defined)
location_coordinates = {
    'KAMA_wrf_2d_May_Aug1998': {'latitude': 35.2197, 'longitude': -100.9993}, # Amarillo, TX
    'KDFW_wrf_2d_May_Aug1998': {'latitude': 32.8998, 'longitude': -97.0403},  # Dallas/Fort Worth, TX
    'KHOU_wrf_2d_May_Aug1998': {'latitude': 29.7604, 'longitude': -95.3698},  # Houston, TX
    'KLBB_wrf_2d_May_Aug1998': {'latitude': 33.5779, 'longitude': -101.8552}, # Lubbock, TX
    'KSAT_wrf_2d_May_Aug1998': {'latitude': 29.4241, 'longitude': -98.4935}   # San Antonio, TX
}

# Assuming processed_dfs contains the preprocessed DataFrames for each location
# and original_dfs_cleaned contains the original cleaned dfs
if processed_dfs and original_dfs_cleaned:
    # Iterate through the processed_dfs dictionary.
    for location_name, df in processed_dfs.items():
        print(f"\nProcessing historical data for visualization for {location_name}...")

        # Use the original cleaned df for percentile calculations within the categorization function
        original_df_cleaned_location = original_dfs_cleaned.get(location_name)

        if original_df_cleaned_location is not None and 'UP_HELI_MAX' in df.columns and 'AFWA_TORNADO' in df.columns:
            # Categorize historical storm intensity using the updated categorization function.
            df['historical_intensity'] = df.apply(
                lambda row: categorize_storm_intensity_percentile_and_tornado(row, original_df_cleaned_location),
                axis=1
            )

            # Filter for timestamps where there is some storm activity (intensity > 0), including tornado class
            df_storms = df[df['historical_intensity'] > 0].copy()

            # Add location coordinates to the dataframe for easy access
            if location_name in location_coordinates:
                df_storms['latitude'] = location_coordinates[location_name]['latitude']
                df_storms['longitude'] = location_coordinates[location_name]['longitude']
            else:
                print(f"Warning: Coordinates not found for {location_name}. Skipping historical visualization data preparation for this location.")
                continue # Skip to the next location if coordinates is missing

            # Structure the data into a list of dictionaries suitable for visualization
            for index, row in df_storms.iterrows():
                historical_visualization_data.append({
                    'timestamp': index, # XTIME is already the index
                    'location_name': location_name.replace('_wrf_2d_May_Aug1998', ''), # Clean up location name
                    'latitude': row['latitude'],
                    'longitude': row['longitude'],
                    'historical_intensity': int(row['historical_intensity']) # Ensure intensity is integer
                })
            print(f"Prepared {len(df_storms)} historical storm data points for {location_name}.")

        else:
            print(f"Warning: Required columns ('UP_HELI_MAX', 'AFWA_TORNADO') or original cleaned data not found in processed data for {location_name}. Cannot prepare historical visualization data.")

    # 3. Convert the list of dictionaries into a pandas DataFrame.
    df_historical_visualization = pd.DataFrame(historical_visualization_data)

    # Ensure timestamp column is datetime and sort by timestamp
    df_historical_visualization['timestamp'] = pd.to_datetime(df_historical_visualization['timestamp'])
    df_historical_visualization = df_historical_visualization.sort_values(by='timestamp')


    # 4. Display the head of the resulting DataFrame and its information.
    print("\n--- Historical Visualization Data DataFrame ---")
    display(df_historical_visualization.head())
    print("\nDataFrame Info:")
    display(df_historical_visualization.info())

else:
    print("Processed data (processed_dfs) or original cleaned data (original_dfs_cleaned) is not available. Cannot prepare data for time-sensitive visualization.")
    df_historical_visualization = None # Ensure df_historical_visualization is None if data is not available


# --- Time-sensitive geospatial visualization (Historical) ---
# 2. Create a base Folium map centered on Texas with an appropriate zoom level.
texas_center = [31.9686, -99.9018] # Approximate center of Texas
m_historical = folium.Map(location=texas_center, zoom_start=6)

# Define color map and intensity labels (matching the previous plot)
colors = {0: 'blue', 1: 'green', 2: 'yellow', 3: 'orange', 4: 'red', 5: 'purple'} # Added purple for tornado
intensity_labels = {0: 'No Storm', 1: 'Light', 2: 'Moderate', 3: 'Heavy', 4: 'Very Heavy', 5: 'Tornado Producing'} # Added label for tornado


# 3. Prepare the historical storm data in a GeoJSON format suitable for TimestampedGeoJson.
geojson_features_historical = []

# 4. Define a function to determine marker properties (color and size) based on the historical storm intensity level.
def get_marker_properties_historical(intensity):
    """Determines marker color and size based on historical storm intensity."""
    color = colors.get(intensity, 'gray') # Default to gray if intensity level is unexpected
    radius = intensity * 5 + 5 # Adjust marker size based on intensity (e.g., 5 to 30 for class 5)
    return color, radius

if 'df_historical_visualization' in locals() and df_historical_visualization is not None:
    # 5. Iterate through the df_historical_visualization DataFrame and create a GeoJSON feature for each row.
    for index, row in df_historical_visualization.iterrows():
        timestamp = row['timestamp'].isoformat() + 'Z' # Format timestamp for GeoJSON
        location_name = row['location_name']
        latitude = row['latitude']
        longitude = row['longitude']
        historical_intensity = int(row['historical_intensity']) # Ensure intensity is integer

        color, radius = get_marker_properties_historical(historical_intensity)

        feature = {
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [longitude, latitude], # GeoJSON is [longitude, latitude]
            },
            'properties': {
                'time': timestamp,
                'popup': f"{location_name}<br>Intensity: {intensity_labels.get(historical_intensity, 'Unknown')}",
                'icon': 'circle', # Use circle icon for markers
                'iconstyle': {
                    'fillColor': color,
                    'fillOpacity': 0.7,
                    'stroke': 'true',
                    'color': 'black',
                    'weight': 1,
                    'radius': radius
                },
                'style': {'color': color}, # Style for line features (not applicable for points, but required)
            }
        }
        geojson_features_historical.append(feature)

    # 6. Create a TimestampedGeoJson layer using the list of GeoJSON features.
    # Sort features by time for TimestampedGeoJson to work correctly
    geojson_features_historical.sort(key=lambda x: x['properties']['time'])

    # Ensure the feature collection has a CRS specified
    geojson_data_historical = {
        'type': 'FeatureCollection',
        'features': geojson_features_historical,
        'crs': {
            'type': 'name',
            'properties': {
                'name': 'urn:ogc:def:crs:OGC:1.3:CRS84' # Standard CRS for geographic coordinates
            }
        }
    }

    # Add the TimestampedGeoJson layer to the Folium map.
    # Use period="PT10M" for 10 minute intervals
    # auto_play=False allows manual control with the slider
    # loop=False prevents the animation from looping automatically
    # transition_time controls the speed of the animation
    folium.plugins.TimestampedGeoJson(
        geojson_data_historical,
        period='PT10M', # Corresponds to 10 minute intervals
        add_last_point=True,
        auto_play=False,
        loop=False,
        transition_time=200, # Time in milliseconds between steps
        duration='PT1M', # Duration of each step (e.g., 1 minute display) - might need tuning
    ).add_to(m_historical)

    # Add a layer control to the map (optional, but good practice)
    folium.LayerControl().add_to(m_historical)

    # 7. Display the interactive Folium map.
    print("\n--- Interactive Historical Storm Visualization ---")
    display(m_historical)

else:
    print("Historical visualization data (df_historical_visualization) is not available. Cannot create time-sensitive geospatial visualization.")
