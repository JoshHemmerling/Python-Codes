import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import numpy as np
from sklearn.tree import export_graphviz
import graphviz
import matplotlib.pyplot as plt


# --- Data Loading and Preprocessing ---
# Load CSV
df = pd.read_csv("klbb_wrf_features.csv", encoding='latin1')
print(f"Shape before handling NaNs: {df.shape}")

# Drop the 'Unnamed: 0' and 'XTIME' columns
df = df.drop(columns=['Unnamed: 0', 'XTIME'])

# Drop specified columns and the columns that were found to be all NaNs
columns_to_drop = ['afwa_tlyrbot', 'afwa_afwa_tlyrtop', 'afwa_turb', 'C1H', 'C2H', 'C1F', 'C2F', 'C3H', 'C4H', 'C3F', 'C4F', 'AFWA_TLYRBOT', 'AFWA_TLYRTOP', 'AFWA_TURB']
df = df.drop(columns=columns_to_drop, errors='ignore') # Use errors='ignore' in case some columns are not present
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

# Define the classification target (event or no event)
# Assuming an event occurs if UP_HELI_MAX is greater than 0
threshold = 0
y_class = (y > threshold).astype(int)

# Train-test split for classification
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_scaled, y_class, test_size=0.2, random_state=42, stratify=y_class) # Use stratify to maintain the proportion of classes


# --- Hyperparameter Tuning for Random Forest Classifier ---
print("\n--- Hyperparameter Tuning (Grid Search) ---")
# Define the parameter grid (using the best parameters found previously as a tighter grid or confirming them)
# Using the best hyperparameters found previously: {'max_depth': 30, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
param_grid = {
    'n_estimators': [200], # Use the best n_estimators
    'max_depth': [30],       # Use the best max_depth
    'min_samples_split': [2], # Use the best min_samples_split
    'min_samples_leaf': [1]   # Use the best min_samples_leaf
}


# Create a GridSearchCV object (using the best parameters found previously)
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           scoring='f1', # Optimize for F1-score of the positive class
                           cv=3, # Use 3-fold cross-validation
                           n_jobs=-1) # Use all available cores

# Perform the grid search (this will be very fast since the grid is just the best parameters)
grid_search.fit(X_train_class, y_train_class)

# Print the best hyperparameters and the corresponding score
print("Best hyperparameters:", grid_search.best_params_)
print("Best F1-score (from Grid Search):", grid_search.best_score_)

# Get the best model
best_rf_classifier = grid_search.best_estimator_


# --- Evaluation of Tuned Random Forest Classifier ---
print("\n--- Classification Evaluation (Tuned Random Forest) ---")
y_pred_best_class = best_rf_classifier.predict(X_test_class)

print(f"Accuracy: {accuracy_score(y_test_class, y_pred_best_class):.2f}")
print("Classification Report:")
print(classification_report(y_test_class, y_pred_best_class))
print("Confusion Matrix:")
print(confusion_matrix(y_test_class, y_pred_best_class))


# --- Decision Tree Visualization (Example) ---
# Select one tree from the forest
try:
    estimator = best_rf_classifier.estimators_[0]
    dot_data = export_graphviz(estimator,
                    feature_names = features,
                    class_names=['No Event', 'Event'],
                    rounded = True, proportion = False,
                    precision = 2, filled = True)
    graph = graphviz.Source(dot_data)
    # Display the tree (will render as an image in Colab)
    display(graph)
    print("\nDisplayed graphviz source for one decision tree.")

    # Save the tree as a JPEG file
    graph.render('decision_tree', format='jpeg', cleanup=True)
    print("Decision tree saved as decision_tree.jpeg")


except Exception as e:
    print(f"\nCould not generate or display decision tree: {e}")
    print("Ensure graphviz is installed and configured if you need to visualize trees.")
