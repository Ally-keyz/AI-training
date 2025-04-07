import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# get your data set

car_data = pd.read_csv("car_prices.csv")

# start spliting the data

Features  = car_data[
    ["Year","Mileage (km)","Engine Size (cc)"]
    ]
target = car_data["Price"]

#use the train test split library to split the data

X_train , X_test ,Y_train ,Y_test = train_test_split(Features,target,test_size=0.3,random_state=42)

scaler = StandardScaler()

#scale the data for higher model accuracy
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#model
model = LinearRegression()

#train the model
model.fit(X_train_scaled,Y_train)
#test the model
Y_pred = model.predict(X_test_scaled)

#print the model accuracy
print(f"Model accuracy : {r2_score(Y_test,Y_pred)}")


#plot the graph for visualisation

plt.scatter(Y_test,Y_pred,color="red")
plt.plot(Y_pred,"blue")
plt.xlabel("Test prices")
plt.ylabel("Predicted prices")
plt.title("Prices predictions")
plt.show()