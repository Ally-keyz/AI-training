import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler , LabelEncoder , OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score 
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor

data  = pd.read_csv("work.csv")

#check for null data
#print(data.isnull().sum())
#print(data.head())
#remove unwanted columns
data  = data.drop(columns="ID")
#print(data.head())


#clean some dummy data
data["Levy"] = data["Levy"].replace('-','0')
data["Levy"] = data["Levy"].astype(int)


#clean also data at Milieage because it seems to be numbers mixed with strings
data["Mileage"] = data["Mileage"].str.replace(' km','',regex=False)
data["Mileage"] = data["Mileage"].str.replace(' ' ,'')
data["Mileage"] = data["Mileage"].astype(int)

#clean the data also at engine volume
data["Engine volume"] = data["Engine volume"].str.replace(r'[^\d.]+', '', regex=True)
data["Engine volume"] = data["Engine volume"].astype(float) 



#define the features and the target
FEATURES  = data[[
    "Levy","Manufacturer","Model","Prod. year","Category","Leather interior","Fuel type","Engine volume",
    "Mileage","Cylinders","Gear box type","Drive wheels"
]
]

TARGET = data["Price"]

# we also have to convert our data to all number for our model to undertand because it only undertand numbers

CATEGORICAL_DATA = data[[
    "Manufacturer",
    "Model","Category","Leather interior",
    "Fuel type","Gear box type",
    "Drive wheels"
]]

# start converting our data into numbers using lable encoder

for dat in CATEGORICAL_DATA :
    encoder = LabelEncoder()
    FEATURES.loc[:,dat] = encoder.fit_transform(FEATURES.loc[:,dat])
    
#print the features to review the data
#print(FEATURES)

#split the data into testing and training data sets
X_train , X_test , Y_train , Y_test = train_test_split(FEATURES,TARGET,test_size=0.3,random_state=32)

#scale the data for better accuracy
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#model
model =  RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=32,
    
)

model.fit(X_train,Y_train)

Y_pred = model.predict(X_test)

#check for the model accuracy
print(f"Model accuracy: {r2_score(Y_test,Y_pred)}")

#draw a distribution to show the visualisation

plt.scatter(Y_test,Y_pred,color="red")
plt.plot(Y_pred,Y_pred,'black')
plt.xlabel("tests")
plt.ylabel("prediction")
plt.show()



