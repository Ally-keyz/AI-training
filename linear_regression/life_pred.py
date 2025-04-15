import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

#get the data set
life_data = pd.read_csv("Life Expectancy Data.csv")

#print the first five rows of data
head = life_data.head()
print(head)

# look for missing values in the data

null = life_data.isnull().sum()
null_cols = [col for col , value in null.items() if value > 0]
print(null_cols)

# we also have to clean the missing values at the same time
for col in null_cols:
    life_data[col] = life_data[col].fillna(life_data[col].mean())

print(life_data.isnull().sum())

# we have to then select the data into categories for easy handling
# Define features and target
FEATURES = life_data[
    [
        "Country", "Year", "Status", "Adult Mortality", "infant deaths",
        "Alcohol", "percentage expenditure", "Hepatitis B", "Measles ",
        " BMI ", "under-five deaths ", "Polio", "Total expenditure", "Diphtheria ",
        " HIV/AIDS", "GDP", "Population", " thinness  1-19 years",
        " thinness 5-9 years", "Income composition of resources","Schooling"
    ]
]

TARGET = life_data["Life expectancy "]

# List of categorical column names only
CATEGORICAL_COLUMNS = ["Country", "Status"]

print(FEATURES)

#create a column transformer

column_transformer = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ('num', StandardScaler(), [col for col in FEATURES.columns if col not in CATEGORICAL_COLUMNS])
    ],
    remainder="passthrough"
)

pipeline = Pipeline([
    ("preprocessing", column_transformer),
    ("regressor", LinearRegression())
])

X_train , X_test , Y_train , Y_test = train_test_split(FEATURES , TARGET , test_size=0.2,random_state=42)

#train the model
pipeline.fit(X_train,Y_train)

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