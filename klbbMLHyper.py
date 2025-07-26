import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, make_scorer
import numpy as np
from lightgbm import LGBMClassifier
from imblearn.over_sampling import ADASYN

# --- Data Loading and Preprocessing ---
# Load CSV
df = pd.read_csv("klbb_wrf_features.csv", encoding='latin1')
print(f"Shape before handling NaNs: {df.shape}")

# Drop the 'Unnamed: 0' and 'XTIME' columns
df = df.drop(columns=['Unnamed: 0', 'XTIME'])

# Drop specified columns and the columns that were found to be all NaNs
columns_to_drop = ['afwa_tlyrbot', 'afwa_afwa_tlyrtop', 'afwa_turb', 'C1H', 'C2H', 'C1F', 'C2F', 'C3H', 'C4H', 'C3F', 'C4F', 'AFWA_TLYRBOT', 'AFWA_TLYRTOP', 'AFWA_TURB']
df = df.drop(columns=columns_to_drop, errors='ignore')
print(f"Shape after dropping specified columns and all-NaN columns: {df.shape}")

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

# Set target
target = 'UP_HELI_MAX'
features = [col for col in df.columns if col not in [target]]

# Prepare data
X = df[features]
y = df[target]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# --- Multi-class Classification Target Creation (based on Percentiles) ---
# Examine the distribution of non-zero UP_HELI_MAX values
y_events = y[y > 0]

print("\nDistribution of non-zero UP_HELI_MAX values:")
print(y_events.describe())

# Calculate percentile thresholds for non-zero UP_HELI_MAX values
threshold_light_percentile = y_events.quantile(0.50)
threshold_moderate_percentile = y_events.quantile(0.75)
threshold_heavy_percentile = y_events.quantile(0.90)

print(f"\nPercentile thresholds for non-zero UP_HELI_MAX:")
print(f"50th percentile (potential Light storm threshold): {threshold_light_percentile:.2f}")
print(f"75th percentile (potential Moderate storm threshold): {threshold_moderate_percentile:.2f}")
print(f"90th percentile (potential Heavy storm threshold): {threshold_heavy_percentile:.2f}")

# Create the multi-class target variable based on UP_HELI_MAX thresholds derived from percentiles
def categorize_storm_intensity_percentile(up_heli_max):
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

y_intensity_percentile = y.apply(categorize_storm_intensity_percentile)

# Check the distribution of the new multi-class target variable
print("\nDistribution of Storm Intensity Categories (based on Percentiles):")
print(y_intensity_percentile.value_counts().sort_index())


# --- Train-Test Split ---
X_train_intensity_perc, X_test_intensity_perc, y_train_intensity_perc, y_test_intensity_perc = train_test_split(
    X_scaled, y_intensity_percentile, test_size=0.2, random_state=42, stratify=y_intensity_percentile
)

# --- Apply ADASYN Resampling ---
print("\nApplying ADASYN resampling to training data...")
adasyn = ADASYN(random_state=42)
X_train_resampled_adasyn, y_train_resampled_adasyn = adasyn.fit_resample(X_train_intensity_perc, y_train_intensity_perc)
print(f"Shape of training data after ADASYN: {X_train_resampled_adasyn.shape}")


# --- Hyperparameter Tuning (Grid Search) ---
# Define the parameter grid for LightGBM
# This is an example, you might want to adjust the ranges and parameters
param_grid = {
    'n_estimators': [100, 200], # Reduced for faster execution
    'learning_rate': [0.05, 0.1], # Reduced for faster execution
    'num_leaves': [31, 50], # Reduced for faster execution
    'max_depth': [-1, 10], # Reduced for faster execution
    'min_child_samples': [20, 50], # Reduced for faster execution
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0, 0.1], # Reduced for faster execution
    'reg_lambda': [0, 0.1], # Reduced for faster execution
}

# Create a custom scorer for weighted F1-score
weighted_f1_scorer = make_scorer(f1_score, average='weighted')

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=LGBMClassifier(objective='multiclass', num_class=len(np.unique(y_train_intensity_perc)),
                             metric='multi_logloss', random_state=42, n_jobs=-1),
    param_grid=param_grid,
    scoring=weighted_f1_scorer,
    cv=3,  # Number of cross-validation folds
    verbose=2,
    n_jobs=-1 # Use all available cores
)

# Fit the grid search to the ADASYN resampled training data
print("\nStarting Grid Search for LightGBM...")
grid_search.fit(X_train_resampled_adasyn, y_train_resampled_adasyn)

# Print the best parameters and the best score
print("\nBest parameters found by Grid Search:")
print(grid_search.best_params_)
print("\nBest weighted F1-score found by Grid Search:")
print(grid_search.best_score_)

# Get the best model
best_lgbm_model = grid_search.best_estimator_

# --- Evaluate Best Model ---
print("\n--- Evaluation of Best LightGBM Model on Test Set ---")
y_pred_best_lgbm = best_lgbm_model.predict(X_test_intensity_perc)

print("Classification Report:")
print(classification_report(y_test_intensity_perc, y_pred_best_lgbm))
print("Confusion Matrix:")
print(confusion_matrix(y_test_intensity_perc, y_pred_best_lgbm))
print(f"Weighted Avg F1 Score: {f1_score(y_test_intensity_perc, y_pred_best_lgbm, average='weighted'):.2f}")
