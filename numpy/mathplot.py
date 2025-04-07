import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#read the csv files 

data = pd.read_csv("Students_Grading_Dataset.csv")

#read the first row data
head = data.head()
print(head)

#plotting concepts
df = np.array([1,2,3,4,5,6,6,7,8,9])

plt.plot(df)
plt.show()



