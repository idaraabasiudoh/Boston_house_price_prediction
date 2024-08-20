# Boston Real Estate Price Prediction using Decision Tree Regressor

This project uses a Decision Tree Regressor to predict real estate prices based on various features in the dataset. The dataset is preprocessed to remove any missing values before training the model. The model's performance is evaluated based on the score and the mean absolute error.

## Table of Contents

- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Author](#author)

## Installation

Ensure that you have the required libraries installed in your Python environment. You can install the necessary libraries using `pip` as shown below:

```bash
pip install pandas==1.3.4
pip install sklearn==0.20.1
```

## Data Preprocessing

The dataset used in this project is loaded from a CSV file available online. The dataset is preprocessed to drop any rows containing missing values, ensuring that the model is trained on clean data.

```python
import pandas as pd

# Load the dataset
data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/real_estate_data.csv")
print(data.head())
print(f"Data Shape: {data.shape}\n")

# Check for missing values
print(data.isna().sum())

# Drop rows with missing values
data.dropna(inplace=True)
print(data.isna().sum())
```
## Model Training

The dataset is split into training and testing sets. A Decision Tree Regressor model is then created and trained on the training data.

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Split the dataset into features and target variable
X = data.drop(columns=["MEDV"])
Y = data["MEDV"]

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# Create and train the model
regr_tree = DecisionTreeRegressor(criterion='mse')
regr_tree.fit(X_train, Y_train)
```

## Model Evaluation

The model's performance is evaluated by calculating the score and the mean absolute error on the test set.

```python
# Evaluate the model
print(f"Model score: {regr_tree.score(X_test, Y_test)}")

# Make predictions on the test set
prediction = regr_tree.predict(X_test)

# Calculate the mean absolute error
print(f"Absolute Mean Error: {(prediction - Y_test).abs().mean()}")
```

## Model Evaluation

The model's performance is evaluated by calculating the score and the mean absolute error on the test set.

```python
# Evaluate the model
print(f"Model score: {regr_tree.score(X_test, Y_test)}")

# Make predictions on the test set
prediction = regr_tree.predict(X_test)

# Calculate the mean absolute error
print(f"Absolute Mean Error: {(prediction - Y_test).abs().mean()}")
```

## Author

This project was created by [idaraabasiudoh](https://github.com/idaraabasiudoh).
