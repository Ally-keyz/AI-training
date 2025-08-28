import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler , OneHotEncoder , OrdinalEncoder
from sklearn.metrics import mean_squared_error , r2_score , mean_absolute_error

# load the data set
df = pd.read_csv("Life Expectancy Data.csv" )

#read the first 5 columns
print(df.head())

# check for the missing values

null = df.isnull().sum()
print(null)
null_cols = [col for col , value in null.items() if value > 0]
print (null_cols)

for col in null_cols :
    df[col] = df[col].fillna(df[col].mean())

# spliting the data into categorical and numerical data and the target

CATEGORICA_COLS = ["Country"]

FEATURES = df[ [
        "Country", "Year", "Status", "Adult Mortality", "infant deaths",
        "Alcohol", "percentage expenditure", "Hepatitis B", "Measles ",
        " BMI ", "under-five deaths ", "Polio", "Total expenditure", "Diphtheria ",
        " HIV/AIDS", "GDP", "Population", " thinness  1-19 years",
        " thinness 5-9 years", "Income composition of resources","Schooling"
    ]]

TARGET = df["Life expectancy "]

column_transformer = ColumnTransformer(
    transformers=[
        ('cat',OneHotEncoder(handle_unknown="ignore"), CATEGORICA_COLS),
        ('unique',OrdinalEncoder(handle_unknown="use_encoded_value" , unknown_value=-1), ["Status"]),
        ('num', StandardScaler() , [col for col in FEATURES.columns  if col not in CATEGORICA_COLS + ["Status"]])
    ]
)

#we will have to prepare our pipeline

pipeline = Pipeline([
    ('preprocessing',column_transformer),
    ('regressor',LinearRegression())
])

# split the data into training and testing data sets

X_train , X_test , Y_train , Y_test = train_test_split(FEATURES , TARGET , test_size=0.2 , random_state=42)

# train our model on the data set
pipeline.fit(X_train,Y_train)

# make predictions
Y_pred = pipeline.predict(X_test)

# evaluate our model

MRE = mean_squared_error(Y_test , Y_pred)
MAE = mean_absolute_error(Y_test , Y_pred)
r2score = r2_score(Y_test , Y_pred)

print(f"MRE : {MRE} , MAE: {MAE} , R2SCORE :{r2score}")