import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
data_set = pd.read_csv("train.csv")

# Drop unwanted column
data_set.drop(columns="ID", inplace=True)

# Clean data
data_set["Levy"] = data_set["Levy"].replace("-", np.nan).astype(float)
data_set["Levy"].fillna(data_set["Levy"].median(), inplace=True)

data_set["Mileage"] = data_set["Mileage"].str.replace(" km", "", regex=False).str.replace(" ", "").astype(int)

# Clean 'Engine volume' column
data_set["Engine volume"] = data_set["Engine volume"].str.replace("Turbo", "", regex=False)
data_set["Engine volume"] = data_set["Engine volume"].str.strip()
data_set["Engine volume"] = data_set["Engine volume"].astype(float)


# Define features and target
FEATURES = data_set[
    [
        "Levy", "Manufacturer", "Model", "Prod. year", "Category",
        "Leather interior", "Fuel type", "Engine volume", "Mileage", "Cylinders",
        "Gear box type", "Drive wheels", "Doors", "Wheel", "Color", "Airbags"
    ]
]
TARGET = data_set["Price"]

# Categorical columns
CATEGORICAL_COLUMNS = ["Manufacturer", "Model", "Category", "Color", "Fuel type", "Gear box type"]
COLUMNS_LABLE = ["Leather interior", "Drive wheels", "Doors", "Wheel"]

# Initialize column transformer
column_transformer = ColumnTransformer(
    transformers=[
        ("col", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ("cols", OrdinalEncoder(), COLUMNS_LABLE),
        ("num", StandardScaler(), [col for col in FEATURES.columns if col not in CATEGORICAL_COLUMNS and col not in COLUMNS_LABLE])
    ],
    remainder="passthrough"
)

# Initialize pipeline
pipeline = Pipeline([
    ("preprocessing", column_transformer),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(FEATURES, TARGET, test_size=0.3, random_state=42)

# Train model
pipeline.fit(X_train, Y_train)

# Predict
Y_pred = pipeline.predict(X_test)

# Check accuracy
print(f"Model R² Score: {r2_score(Y_test, Y_pred):.3f}")

# Visualize predictions
plt.figure(figsize=(8, 6))
plt.scatter(Y_test, Y_pred, color='skyblue', alpha=0.6, label="Predictions")
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], color='red', linestyle='--', label="Perfect Prediction")
plt.xlabel("Actual Car Price")
plt.ylabel("Predicted Car Price")
plt.title("Actual vs Predicted Car Prices")
plt.legend()
plt.grid(True)
plt.show()
