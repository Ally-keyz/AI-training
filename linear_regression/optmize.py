#life predictor enhancement and revision
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score , mean_squared_error
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor 
#load the data
data_set = pd.read_csv("Life Expectancy Data.csv")

#check for missing values
null_values_check = data_set.isnull().sum()
print(null_values_check)

# we have to then clean them at a time
null_values = [ col for col , value in null_values_check.items() if value > 0]

print(null_values)

# lets to the cleaning of the data

for col in null_values:
    if data_set[col].dtype == 'object':
        data_set[col] = data_set[col].fillna(data_set[col].mode()[0])
    else:   
      data_set[col] = data_set[col].fillna(data_set[col].mean())

print(data_set.isnull().sum())

# so we also have to separate the data

FEATURES = data_set[
    [
        "Country", "Year", "Status", "Adult Mortality", "infant deaths",
        "Alcohol", "percentage expenditure", "Hepatitis B", "Measles ",
        " BMI ", "under-five deaths ", "Polio", "Total expenditure", "Diphtheria ",
        " HIV/AIDS", "GDP", "Population", " thinness  1-19 years",
        " thinness 5-9 years", "Income composition of resources","Schooling"
    ]
]

TARGET = data_set["Life expectancy "]

# List of categorical column names only
CATEGORICAL_COLUMNS = ["Country", "Status"]

# we have to the initalise the column transformer

column_transformer = ColumnTransformer(
    transformers= [
       ('cat', OneHotEncoder(handle_unknown="ignore") , CATEGORICAL_COLUMNS),
       ('num' , StandardScaler() , [ col for col in FEATURES.columns if col not in CATEGORICAL_COLUMNS])
    ]
)

# we have to then initialise the pipeline

pipeline = Pipeline([
    ("preprocessing", column_transformer),
    ("regressor", RandomForestRegressor(n_estimators=100 , random_state=42))
])


# we then have to split the data into trainig and testing dataset

X_train , X_test , Y_train , Y_test = train_test_split(FEATURES ,TARGET ,test_size=0.2,random_state=42)

# train our model

pipeline.fit(X_train,Y_train)

# make predictions
Y_pred = pipeline.predict(X_test)

# Check accuracy
print(f"Model R² Score: {r2_score(Y_test, Y_pred):.3f}")
print(f"Mean squared error : {mean_squared_error(Y_test ,Y_pred):.3f}")

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