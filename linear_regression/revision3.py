import pandas as pd
import matplotlib.pylab as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , OneHotEncoder , OrdinalEncoder
from sklearn.metrics import r2_score , mean_absolute_error , mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("Life Expectancy Data.csv")

# read the 5 first rows
head = df.head()
print(head)

# check if there are missing values at the target variable
null = df.isnull().sum()
null_cols = [col for col , value in null.items if value > 0]
print("null columns : \n", null)

# remove the missing foelds from the target input y
df.dropna(subset=['Life Expectancy Data.csv'] , inplace=True)

# distribute the data into features , target 
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

# prepare our column transfrormer for preprocessing

def column_transformer(features):

    nom_cat_cols = [col for col in features.columns  if col in ["Country"]]
    ordinal_cols = [col for col in features.columns if col in ["Year", "Status"]]
    numerical_cols = [col for col in features.columns if col not in nom_cat_cols + ordinal_cols]

    nom_cat_preprocessor = Pipeline(steps=[
        ("impute", SimpleImputer(strategy='median')),
        ("encoder" ,OneHotEncoder(handle_unknown="ignore") )
    ])

    ordinal_cols_preprocessor = Pipeline(steps=[
        ("impute" , SimpleImputer(strategy="most_frequent")),
        ("encoder" , OrdinalEncoder())
    ])

    numerical_cols_preprocessor = Pipeline(steps=[
        ("impute" , SimpleImputer(strategy="median")),
        ("scaler" , StandardScaler())
    ])


    column_processor = ColumnTransformer(
        transformers= [
            ('nom_cat' , nom_cat_preprocessor , nom_cat_cols)
            ('ordinal_cat' , ordinal_cols_preprocessor , ordinal_cols)
            ('num' , numerical_cols_preprocessor , numerical_cols)
        ]
    )

    return column_processor


def pipeline(column_transformer):

    my_pipeline = Pipeline([
        ('preprocessing' , column_transformer),
        ("regressor" , RandomForestRegressor(random_state=42 , max_depth=6 , max_leaf_nodes=6 , n_estimators=200 , max_features="sqrt"))
    ])

    return my_pipeline


def data_spliting(features  , target):

    X_train , X_test , Y_train , Y_test = train_test_split(features , target , test_size=0.2 , random_state=42)
    return X_train , X_test , Y_train , Y_test


# Train our model and evaluate it
columnTransformer = column_transformer(FEATURES)
model = pipeline(columnTransformer)

X_train , X_test , Y_train , Y_test = data_spliting(FEATURES , TARGET)

# fit the data to the model
model.fit(X_train , Y_train)

feature_names = []

for names, transformers , features in columnTransformer.transformers_:
    if hasattr(transformers , "get_feature_names_out"):
        feature_names.extend(transformers.get_feature_names_out(features))
    else: 
        feature_names.extend(features)

print(feature_names)

coeficients = model.named_steps(["regressor"]).feature_importances_
feature_importances = pd.DataFrame({'Features':feature_names , 'Coeficients':coeficients})


