import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Assume df exists, otherwise generate some example data
# Example DataFrame df creation
df = pd.DataFrame({
    'Speed': np.random.uniform(200, 1200, 500),
    'Altitude': np.random.uniform(500, 10000, 500),
    'RCS': np.random.uniform(0.1, 10, 500),
    'Distance': np.random.uniform(1, 50, 500),
    'Cockpit_temp_C': np.random.randint(25, 50, 500)
})

# Additional data
additional_data = {
    'Mach_number': np.linspace(0.1, 0.8, 100),  # Additional Mach numbers
    'Altitude_km': np.linspace(1, 14, 100),  # Additional Altitudes
    'Ambient_temp_C': np.random.randint(40, 50, 100),  # Random ambient temperatures
    'Solar_radiation_W_m2': np.random.randint(1000, 1500, 100),  # Random solar radiation
    'Mass_flow_rate_kg_sec': np.random.uniform(0.08, 0.1, 100),  # Random mass flow rate
    'Cockpit_pressure_diff_kPa': np.random.uniform(30, 40, 100),  # Random cockpit pressure diff
    'Cockpit_temp_C': np.random.randint(25, 50, 100)  # Random cockpit temperatures
}

# Concatenate the additional data with the existing dataframe
df_additional = pd.DataFrame(additional_data)
df = pd.concat([df, df_additional], ignore_index=True)

# Split the data into features and target
X = df.drop(columns=['Cockpit_temp_C'])
y = df['Cockpit_temp_C']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
