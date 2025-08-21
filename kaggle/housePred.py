import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

# load the data set
house_data = pd.read_csv("train.csv", na_values=["NA"])
test_set = pd.read_csv("test.csv",na_values=["NA"])
test_ids = test_set["Id"]

# remove unwanted columns
house_data.drop(columns="Id",inplace=True)
test_set.drop(columns="Id",inplace=True)

# find the missing values and clean them

null_values = house_data.isnull().sum()
null_percent = (null_values / len(house_data)) * 100
print(null_percent[null_percent > 0].sort_values(ascending=False))

test_null_values = test_set.isnull().sum()
tes_null_percent = (test_null_values / len(test_set)) *100

limit = 50
cols_to_drop = null_percent[null_percent > limit].index

test_set.drop(columns=cols_to_drop , inplace=True)
house_data.drop(columns=cols_to_drop , inplace=True)

new_test_null_values = test_set.isnull().sum()
test_null_cols =[col for col, value in new_test_null_values.items() if value > 0]

new_null_values = house_data.isnull().sum()
null_cols = [col for col , value in new_null_values.items() if value > 0]

for col in null_cols:
    if house_data[col].dtype == "object":
        #fill with the frequent value
        house_data[col] = house_data[col].fillna(house_data[col].mode()[0])
    else:
        house_data[col] = house_data[col].fillna(house_data[col].mean())

for col in test_null_cols:
    if test_set[col].dtype == 'object':
        test_set[col] = test_set[col].fillna(test_set[col].mode()[0])
    else:
        test_set[col] = test_set[col].fillna(test_set[col].mean())



# we have now to define features and target 

NUMERICAL_COLUMNS = house_data.select_dtypes(include=["int64","float64"]).columns.tolist()
CATEGORICAL_COLUMNS = house_data.select_dtypes(include=['object']).columns.tolist()
print([NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS])
TARGET = house_data["SalePrice"]
TARGET_COLUMN = "SalePrice"
NUMERICAL_COLUMNS.remove(TARGET_COLUMN)

FEATURES = house_data[NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS]
TEST_FEATURES = test_set[NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS]


# we have to prepare then our column transformer
column_transformer = ColumnTransformer(
    transformers=[
        ("col",OneHotEncoder(handle_unknown="ignore") ,CATEGORICAL_COLUMNS),
        ("num", StandardScaler() , NUMERICAL_COLUMNS)
    ]
)

# THEN lets prepare our pipeline
pipeline = Pipeline([
    ("preprocessing",column_transformer),
    ("regressor",RandomForestRegressor(n_estimators=100 , random_state=42))
])

# lets then separate the testing and training data
X_train , X_test , Y_train , Y_test = train_test_split(FEATURES , TARGET , test_size=0.2 , random_state=42)

# we are going to then train our modal
pipeline.fit(X_train, Y_train)

# make predictions from our modal
Y_pred = pipeline.predict(X_test)
test_pred = pipeline.predict(TEST_FEATURES)


save = pd.DataFrame({
    "Id":test_ids,
    "SalePrice":test_pred
})

save.to_csv("Submission.csv",index=False)

# print out the r2_score
print(f"Modal accuracy score:{r2_score(Y_test,Y_pred)}")


#graph for visualisation
plt.figure(figsize=(10,10))
plt.scatter(Y_test,Y_pred,color ="skyblue",alpha=0.2,label="Predictions")
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], color='red', linestyle='--', label="Perfect Prediction")
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")
plt.legend()
plt.grid(True)
plt.show()

