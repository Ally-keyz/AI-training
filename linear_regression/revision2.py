import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# load the data
life_data = pd.read_csv("Life Expectancy Data.csv")

# read the first 5 row of the data
print(life_data.head())

# check for null values
null_values = life_data.isnull().sum()
print(null_values)
null_cols = [col for col , value  in null_values.items() if value > 0]
print(null_cols)


# lets then fill the columns with null values with precise or random values

for col in null_cols:
    life_data[col] = life_data[col].fillna(life_data[col].mean())
    

print(life_data.isnull().sum())

# next step is to define targets and features

FEATURES = life_data [
        [
        "Country", "Year", "Status", "Adult Mortality", "infant deaths",
        "Alcohol", "percentage expenditure", "Hepatitis B", "Measles ",
        " BMI ", "under-five deaths ", "Polio", "Total expenditure", "Diphtheria ",
        " HIV/AIDS", "GDP", "Population", " thinness  1-19 years",
        " thinness 5-9 years", "Income composition of resources","Schooling"
    ]
]

TARGET = life_data["Life expectancy "]

CATEGORICAL_COLUMNS = ["Country", "Status"]

# we have to the prepare a column transpormer for encoding the categorical columns and scaling the data
column_transformer = ColumnTransformer(
    transformers= [
        ("cat" , OneHotEncoder(handle_unknown="ignore")  , CATEGORICAL_COLUMNS),
        ("num" , StandardScaler() , [col for col  in FEATURES.columns if col not in CATEGORICAL_COLUMNS])
    ]
)


# lets create a pipeline for the tratining and testing 

pipeline = Pipeline([
    ("preprocessing" , column_transformer),
    ("regressor" , LinearRegression() )
])

# lets then divide our data into training and testing data sets

X_train , X_test , Y_train , Y_test = train_test_split(FEATURES , TARGET , test_size= 0.2 , random_state=42)

# then lets start training our modal
pipeline.fit(X_train , Y_train)

# the we will have to test our modal to see if its able to make predictions
Y_pred = pipeline.predict(X_test)

# check our modal accuracy

print(f"modal accuracy r2_score : {r2_score(Y_test,Y_pred)}")



