import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Prepare additional synthetic data
additional_data = {
    'Mach_number': np.linspace(0.1, 0.8, 100),
    'Altitude_km': np.linspace(1, 14, 100),
    'Ambient_temp_C': np.random.randint(40, 50, 100),
    'Solar_radiation_W_m2': np.random.randint(1000, 1500, 100),
    'Mass_flow_rate_kg_sec': np.random.uniform(0.08, 0.1, 100),
    'Cockpit_pressure_diff_kPa': np.random.uniform(30, 40, 100),
    'Cockpit_temp_C': np.random.randint(25, 50, 100)  # Target variable
}

# Create DataFrame for the additional data
df_additional = pd.DataFrame(additional_data)

# Assuming you have an existing DataFrame `df` to concatenate with
df = pd.concat([df, df_additional], ignore_index=True)

# Step 2: Split the data into features (X) and target (y)
X = df.drop(columns=['Cockpit_temp_C'])
y = df['Cockpit_temp_C']

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Create a pipeline with Polynomial Features and Gradient Boosting
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize features
    ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),  # Add polynomial terms
    ('model', GradientBoostingRegressor(random_state=42))  # Gradient Boosting Model
])

# Step 5: Define hyperparameter grid for tuning
param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.05, 0.1, 0.2]
}

# Step 6: Grid Search for hyperparameter tuning
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)

# Best model from GridSearchCV
best_model = grid_search.best_estimator_

# Step 7: Make predictions on the test set
y_pred = best_model.predict(X_test)

# Step 8: Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output results
print("Mean Squared Error:", mse)
print("R-squared:", r2)
print("Best Model Parameters:", grid_search.best_params_)

# Recommendations for cockpit settings based on predicted optimal conditions
def recommend_adjustments(predicted_temp, actual_temp):
    """Suggest adjustments based on the difference in predicted and actual temperatures."""
    if predicted_temp < actual_temp - 2:
        return "Increase cockpit temperature slightly."
    elif predicted_temp > actual_temp + 2:
        return "Reduce cockpit temperature slightly."
    else:
        return "Temperature is optimal."

# Apply recommendations
recommendations = [recommend_adjustments(pred, actual) for pred, actual in zip(y_pred, y_test)]
for i, rec in enumerate(recommendations[:5]):  # Display a few recommendations
    print(f"Test sample {i + 1}: {rec}")
