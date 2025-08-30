import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score , mean_absolute_error , mean_squared_error
from sklearn.preprocessing import StandardScaler , OneHotEncoder , OrdinalEncoder
from sklearn.impute import SimpleImputer


# loading the data set
df  = pd.read_csv("Life Expectancy Data.csv")

# first 5 column for analysis
head = df.head()
print(head)

# check for missing values and clean them
null_values = df.isnull().sum()
null_cols = [col for col , value in null_values.items() if value > 0]
#print(df.isnull().sum())

# drop lows where target is nan
df.dropna(subset=["Life expectancy "] , inplace=True)

# divide the data into features , target and categorical data
FEATURES = df[
    [
        "Country","Year", "Status", "Adult Mortality", "infant deaths",
        "Alcohol", "percentage expenditure", "Hepatitis B", "Measles ",
        " BMI ", "under-five deaths ", "Polio", "Total expenditure", "Diphtheria ",
        " HIV/AIDS", "GDP", "Population", " thinness  1-19 years",
        " thinness 5-9 years", "Income composition of resources","Schooling"
    ]
]

TARGET = df["Life expectancy "]

# creating a function for preprocessing for easy reusability

def preprocessing(features):
    #categorical columns
    NOMINAL_COLS = [col for col in features.columns if col == "Country"]
    ORDINAL_COLS = [col for col in features.columns if col  in ["Status","Year"]]

    num_cols_preprocessor = Pipeline(steps=[
        ('imputer' , SimpleImputer(strategy='median')),
        ('scaler' , StandardScaler())
    ])

    cat_nominal_preprocessor = Pipeline(steps=[
        ('imputer' , SimpleImputer(strategy='most_frequent')),
        ('onehot' , OneHotEncoder(handle_unknown='ignore'))
    ])

    cat_ordinal_preprocessor = Pipeline(steps=[
        ('imputer' , SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder())
    ])

    # creating a column transformer
    column_transformer = ColumnTransformer(
        [
            ('cat' , cat_nominal_preprocessor, NOMINAL_COLS),
            ('ordinal' , cat_ordinal_preprocessor , ORDINAL_COLS),
            ('num' , num_cols_preprocessor , [col for col in features.columns if col not in NOMINAL_COLS + ORDINAL_COLS])
        ]
    )

    return column_transformer

# we are going to create another function for pipeline creation and data spliting

def pipeline_processing(column_transformer):

    # prepare our pipeline
    pipeline = Pipeline([
        ('preprocessing', column_transformer),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    return pipeline


def train_test_spliting(features , target):

    # initialize our train and test spliting using sklearn
    X_train , X_test , Y_train , Y_test = train_test_split(features , target , test_size=0.2  , random_state=42)
    return X_train , X_test , Y_train  , Y_test


column_transformer = preprocessing(FEATURES)
pipeline = pipeline_processing(column_transformer)

X_train , X_test ,Y_train , Y_test = train_test_split(FEATURES , TARGET)

# train and evaluate the model

pipeline.fit(X_train , Y_train)

# feature importance analysis

feature_names = []

for names , transformer , features in column_transformer.transformers_:
    if hasattr(transformer , 'get_feature_names_out'):
        feature_names.extend(transformer.get_feature_names_out(features))
    else:
        feature_names.extend(features)

coeficients = pipeline.named_steps['regressor'].feature_importances_


feature_importances = pd.DataFrame({ 'Features' : feature_names , 'Coeficients' : coeficients})
print(feature_importances.sort_values('Coeficients', ascending=False))

thresHold = 0.01

non_important_features = feature_importances.loc[feature_importances['Coeficients'].abs() < thresHold  , 'Features'].tolist()

FEATURES = FEATURES.drop(columns=[col for col in non_important_features if col in FEATURES])

# retrain the model on the new features
column_transformer = preprocessing(FEATURES)
pipeline = pipeline_processing(column_transformer)
X_train , X_test , Y_train , Y_test = train_test_spliting(FEATURES , TARGET)

pipeline.fit(X_train , Y_train)
# use the model to make prediction
Y_pred = pipeline.predict(X_test)
# evaluate model metrics

MSE = mean_squared_error(Y_test , Y_pred)
MAE = mean_absolute_error(Y_test , Y_pred)
R2SCORE = r2_score(Y_test , Y_pred)

print(f"MSE : {MSE} ,  MAE : { MAE} , ACCURACY : {R2SCORE}")

# use matplot lib for visual analysis of the model perfomance
plt.scatter(Y_test , Y_pred , alpha=0.5)
plt.ylabel("Predicted value")
plt.xlabel("Actual values")
plt.title("Life prediction")
plt.show()


