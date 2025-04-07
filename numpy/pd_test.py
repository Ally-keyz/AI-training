import pandas as pd

data = pd.read_csv("Students_Grading_Dataset.csv", header=0).values

#head  = data.head()

saved = pd.DataFrame(data)
saved.to_csv("panda.csv")

print(data)

# more advanced concepts

#reading scpecific columns

df_col = pd.read_csv("Students_Grading_Dataset.csv",usecols=["First_Name"]).values

print(f"--------------------- \n {df_col}")

