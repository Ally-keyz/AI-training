import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from catboost import CatBoostRegressor , Pool
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# load the data set
house_data = pd.read_csv("train.csv",na_values=["NA"])
house_data_test = pd.read_csv("test.csv",na_values=['NA'])

#load the first 5 rows 
print(house_data.head())

# DROP UNWANTED COLS
house_data.drop(columns="Id", inplace=True)
house_data_test.drop(columns="Id",inplace=True)
#CHECK FOR NULL VALUES

null_values = house_data.isnull().sum()

# to prevent over filling the data with lots of null we have to limit
null_percentage = (null_values / len(house_data)) * 100

limit = 50
cols_to_drop = null_percentage[null_percentage > limit].index
house_data.drop(columns=cols_to_drop, inplace=True)
house_data_test.drop(columns=cols_to_drop,inplace=True)

new_null_values = house_data.isnull().sum()
null_values_test = house_data_test.isnull().sum()
null_cols = [col for col , value in new_null_values.items() if value > 0]
null_cols_test = [col for col , value in null_values_test.items() if value > 0]

# cleanup the null values
for col in null_cols:
    if house_data[col].dtype == "object":
        house_data[col] = house_data[col].fillna(house_data[col].mode()[0])
    else:
        house_data[col] = house_data[col].fillna(house_data[col].mean())

for col in null_cols_test:
    if house_data_test[col].dtype == 'object':
        house_data_test[col] = house_data_test[col].fillna(house_data_test[col].mode()[0])
    else:
        house_data_test[col] = house_data_test[col].fillna(house_data_test[col].mean())    

# separate the data into features and targets

NUMERICAL_COLS = house_data.select_dtypes(include=["int64",'float64']).columns.tolist()
TARGET = house_data["SalePrice"]
target = "SalePrice"
NUMERICAL_COLS.remove(target)
CATEGORICAL_COLS = house_data.select_dtypes(include=['object']).columns.tolist()
FEATURES = house_data[NUMERICAL_COLS + CATEGORICAL_COLS]
TEST_SET = house_data_test

# split the data into training and testing data sets
X_train , X_test , Y_train , Y_test = train_test_split(FEATURES , TARGET , test_size=0.2 , random_state=42)


# we then have to tell the modal what columns are categorical
training_pool = Pool(X_train, Y_train , cat_features= CATEGORICAL_COLS)
eval_pool = Pool(X_test ,Y_test , cat_features=CATEGORICAL_COLS)

# Initialize our Modal
Modal = CatBoostRegressor(iterations=2000,
                           learning_rate=0.01 ,
                           depth= 8,
                           l2_leaf_reg=6,
                           random_seed=42,
                           early_stopping_rounds=50,
                           eval_metric='R2',
                           verbose=100
)
# Train our modal
Modal.fit(training_pool,eval_set=eval_pool)

#use our modal to make predictions
Y_pred = Modal.predict(X_test)

#Evaluate the modal accuracy
print(f"Modal acurracy score : {r2_score(Y_test,Y_pred)}")








