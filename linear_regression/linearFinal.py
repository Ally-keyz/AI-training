import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score , mean_absolute_error , mean_squared_error
from sklearn.preprocessing import  StandardScaler , OneHotEncoder , OrdinalEncoder
from sklearn.linear_model  import LinearRegression 
from sklearn.ensemble  import RandomForestRegressor

DF = pd.read_csv("Life Expectancy Data.csv")

#load the first 5 columns fo data analysis

head = DF.head()
#print(head)

# we must first look for missing values in our data set

null_values = DF.isnull().sum()
null_cols = [col  for col , value in null_values.items() if value > 0]
print(null_cols)

# clean the missing values data
for col in null_cols:
    DF[col] = DF[col].fillna(DF[col].mean())

# check for the cleaning
print(DF.isnull().sum())

# srub our data into features , categorical cols and the target
FEATURES = DF[
    [
        "Country","Year", "Status", "Adult Mortality", "infant deaths",
        "Alcohol", "percentage expenditure", "Hepatitis B", "Measles ",
        " BMI ", "under-five deaths ", "Polio", "Total expenditure", "Diphtheria ",
        " HIV/AIDS", "GDP", "Population", " thinness  1-19 years",
        " thinness 5-9 years", "Income composition of resources","Schooling"
    ]
]

TARGET = DF["Life expectancy "]

# List of categorical column names only
NOMINAL_COLS = ["Country"]
ORDINAL_COLS = ["Status","Year"]


# prepare aour column transformer

column_transformer = ColumnTransformer(
    transformers= [
        ('cat' , OneHotEncoder(handle_unknown='ignore') , NOMINAL_COLS ),
        ('ordinal' , OrdinalEncoder() , ORDINAL_COLS),
        ('scaler' , StandardScaler() , [col for col in FEATURES.columns if col not in NOMINAL_COLS + ORDINAL_COLS])
    ]
)


pipeline = Pipeline([
    ('preprocessing' , column_transformer),
    ('regressor' , RandomForestRegressor(random_state=42))
])

# split the training and testing data sets

X_train , X_test , Y_train  , Y_test = train_test_split(FEATURES , TARGET , test_size=0.2 , random_state=42)

# fit the data to the model

pipeline.fit(X_train , Y_train)

# analyse the feature importance for later tuning

feature_names_out = []

for name  , transformer , features in column_transformer.transformers_:
    if hasattr(transformer , "get_feature_names_out"):
        feature_names_out.extend(transformer.get_feature_names_out(features))
    else :
        feature_names_out.extend(features) 

coeficients = pipeline.named_steps['regressor'].feature_importances_

feature_importance = pd.DataFrame({'Features':feature_names_out , 'Coeficients': coeficients})
print(feature_importance.sort_values(by='Coeficients' , ascending=False))

# analyse for non important features
thresHold = 0.01
non_important_features = feature_importance.loc[feature_importance['Coeficients'].abs() < thresHold  , 'Features'].tolist()
print(non_important_features)

# after evaluating the model and finding the important features am going to train the model on more important ones

FEATURES = FEATURES.drop(columns=[col for col in non_important_features if col in FEATURES.columns])
NOMINAL_COLS = [col for col in FEATURES.columns if col == "Country"]
ORDINAL_COLS = [col for col in FEATURES.columns if col in ["Status","Year"]]

column_transformer = ColumnTransformer(
    transformers= [
        ('cat' , OneHotEncoder(handle_unknown='ignore') , NOMINAL_COLS ),
        ('ordinal' , OrdinalEncoder() , ORDINAL_COLS),
        ('scaler' , StandardScaler() , [col for col in FEATURES.columns if col not in NOMINAL_COLS + ORDINAL_COLS])
    ]
)


pipeline = Pipeline([
    ('preprocessing' , column_transformer),
    ('regressor' , RandomForestRegressor(random_state=42))
])

X_train , X_test , Y_train  , Y_test = train_test_split(FEATURES , TARGET , test_size=0.2 , random_state=42)
pipeline.fit(X_train , Y_train)

# use the model to make predictions
Y_pred = pipeline.predict(X_test)

# evaluate our model

MSE = mean_squared_error(Y_test , Y_pred)
MAE = mean_absolute_error(Y_test , Y_pred)
R2E  = r2_score(Y_test , Y_pred)


print(f"MSE : {MSE} , MAE : {MAE} , R2E :{R2E}")

plt.scatter(Y_test , Y_pred , alpha=0.5)
plt.xlabel("Actual value")
plt.ylabel("Predicted Life Expectancy")
plt.show()