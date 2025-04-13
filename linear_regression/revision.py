import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
data_set = pd.read_csv("Life Expectancy Data.csv")

# Clean column names (remove leading/trailing spaces)
data_set.columns = data_set.columns.str.strip()

# Check for missing values
null_values_check = data_set.isnull().sum()
print(null_values_check)

# Collect columns with missing values
null_values = [col for col, val in null_values_check.items() if val > 0]
print("Columns with null values:", null_values)

# Fill missing values with the column mean
for column in null_values:
    data_set[column] = data_set[column].fillna(data_set[column].mean())

# Confirm all null values are handled
print(data_set.isnull().sum())

# Define features and target
FEATURES = data_set[
    [
        "Country", "Year", "Status", "Adult Mortality", "infant deaths", "Alcohol",
        "percentage expenditure", "Hepatitis B", "Measles", "BMI",
        "under-five deaths", "Polio", "Total expenditure", "Diphtheria", "HIV/AIDS",
        "GDP", "Population", "thinness  1-19 years", "thinness 5-9 years",
        "Income composition of resources", "Schooling"
    ]
]

TARGET = data_set["Life expectancy"]

# List of categorical column names only
CATEGORICAL_COLUMNS = ["Country", "Status"]

# Column transformer with OneHotEncoder
column_transformer = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ('num', StandardScaler(), [col for col in FEATURES.columns if col not in CATEGORICAL_COLUMNS])
    ]
)

# Create pipeline
pipeline = Pipeline([
    ("preprocessing", column_transformer),
    ("regressor", LinearRegression())
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
plt.xlabel("Actual Life Expectancy")
plt.ylabel("Predicted Life Expectancy")
plt.title("Actual vs Predicted Life Expectancy")
plt.legend()
plt.grid(True)
plt.show()


