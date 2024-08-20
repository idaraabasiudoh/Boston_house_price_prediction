# Install libraries not already in the environment using pip
# !pip install pandas==1.3.4
# !pip install sklearn==0.20.1

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/real_estate_data.csv")
print(data.head())
print(f"Data Shape: {data.shape}\n")
data.isna().sum()

# Data Pre-processing
# Drop rows with missing values
data.dropna(inplace=True)
data.isna().sum()

X = data.drop(columns=["MEDV"])
Y = data["MEDV"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)


# Ccraete and train model
regr_tree = DecisionTreeRegressor(criterion='mse')
regr_tree.fit(X_train, Y_train)

# Model Evaluation
print(f"Model score: {regr_tree.score(X_test, Y_test)}")
prediction = regr_tree.predict(X_test)

print(f"Absolute Mean Error: {(prediction - Y_test).abs().mean()}")