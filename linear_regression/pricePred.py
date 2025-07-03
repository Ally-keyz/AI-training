import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.pipeline  import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor



# load the data set
car_data = pd.read_csv("CarPrice.csv")

# read the first 5 rows 
print(car_data.head())

# we have to first drop un wanted columns
car_data.drop(columns="car_ID" , inplace=True)
car_data.drop(columns="CarName" , inplace=True)

# we must then identify null values and clean them
null_values = car_data.isnull().sum()
print(null_values)

null_cols = [col for col , value in null_values.items() if value > 0]

for col in null_cols:
    car_data[col] = car_data[col].fillna(car_data[col].mean())

# we must also then separate the features and the targets

CATEGORICAL_COLUMNS = [
    "fueltype", "aspiration", "doornumber", "carbody",
    "drivewheel", "enginelocation", "enginetype", "cylindernumber"
]

NUMERICAL_COLUMNS = [
    "symboling", "wheelbase", "carlength", "carwidth", "carheight", "curbweight",
    "enginesize", "boreratio", "stroke", "compressionratio",
    "horsepower", "peakrpm", "citympg", "highwaympg"
]

FEATURES = car_data[NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS]

TARGET = car_data["price"]



# we are going to then prepare our column transformer

column_transformer = ColumnTransformer(
    transformers= [
        ("col",OneHotEncoder(handle_unknown="ignore") , CATEGORICAL_COLUMNS),
        ("num" , StandardScaler() , [col for col in FEATURES.columns if col not in CATEGORICAL_COLUMNS])
    ]
)


# make a pipeline for assembling
pipeline = Pipeline([
    ("preprocessing" , column_transformer),
    ("regressor" , RandomForestRegressor(n_estimators=100 , random_state=42))
])

# we will have to split the data into training and testing data
X_train , X_test , Y_train , Y_test = train_test_split(FEATURES , TARGET , test_size=0.2  , random_state=42)

# we have to train the modal with the data
pipeline.fit(X_train,Y_train)

# use the modal to make prediction
Y_pred = pipeline.predict(X_test)

# we then print the modal accuracy score
print(f"Modal acuracy score {r2_score(Y_test,Y_pred)}")

# graph for visualization

plt.figure(figsize=(8, 6))
plt.scatter(Y_test, Y_pred, color='skyblue', alpha=0.6, label="Predictions")
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], color='red', linestyle='--', label="Perfect Prediction")
plt.xlabel("Actual Car Price")
plt.ylabel("Predicted Car Price")
plt.title("Actual vs Predicted Car Prices")
plt.legend()
plt.grid(True)
plt.show()







