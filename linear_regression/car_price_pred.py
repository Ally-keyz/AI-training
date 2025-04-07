import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the data
data = pd.read_csv("train.csv")

# Remove unwanted columns
data = data.drop(columns=["ID"])

# Check for null values
print(data.isnull().sum())

# Clean the 'Mileage' column (e.g., "123 000 km" -> 123000)
data["Mileage"] = data["Mileage"].str.replace(' km', '', regex=False)
data["Mileage"] = data["Mileage"].str.replace(' ', '')
data["Mileage"] = data["Mileage"].astype(int)

# Clean the 'Levy' column: replace '-' with NaN, then fill with the median
data["Levy"] = data["Levy"].replace('-', np.nan)
data["Levy"] = data["Levy"].astype(float)
data["Levy"] = data["Levy"].fillna(data["Levy"].median())

# Clean the 'Engine volume' column (remove non-numeric parts, e.g., "Turbo")
data["Engine volume"] = data["Engine volume"].str.replace(r'[^\d.]+', '', regex=True)
data["Engine volume"] = data["Engine volume"].astype(float)

# Define FEATURES and TARGET; use .copy() to avoid SettingWithCopyWarning
FEATURES = data[
    ["Levy", "Manufacturer", "Model", "Prod. year", "Category", "Leather interior",
     "Fuel type", "Engine volume", "Mileage", "Cylinders", "Gear box type", "Drive wheels", 
     "Doors", "Wheel", "Color", "Airbags"]
].copy()
TARGET = data["Price"]

# List of categorical columns to encode
categorical_cols = ["Manufacturer", "Model", "Category", "Leather interior",
                    "Fuel type", "Gear box type", "Drive wheels", "Doors", "Wheel", "Color"]

# Apply LabelEncoder to each categorical column
for col in categorical_cols:
    le = LabelEncoder()
    FEATURES.loc[:, col] = le.fit_transform(FEATURES.loc[:, col])

# Split the data into training and testing datasets
X_train, X_test, Y_train, Y_test = train_test_split(FEATURES, TARGET, test_size=0.3, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define and train the model
model = LinearRegression()
model.fit(X_train_scaled, Y_train)

# Make predictions
Y_pred = model.predict(X_test_scaled)

# Print the model accuracy (R² score)
print(f"Model accuracy: {r2_score(Y_test, Y_pred)}")

# Plot the results for visualization
plt.title("Car Prices Predictions")
plt.scatter(Y_test, Y_pred, color="red")
plt.ylabel("Predictions")
plt.xlabel("Tested Values")
plt.show()
