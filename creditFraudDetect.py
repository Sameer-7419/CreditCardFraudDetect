#Importing the necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
credit_card_data = pd.read_csv('creditcard.csv')

# Distribution of legit transactions & fraudulant transactions
credit_card_data['Class'].value_counts()
# This dataset is highly unbalanced (0-> Normal transaction 1-> Fraudulant Transaction)

# Separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

legit.shape
fraud.shape

legit.Amount.describe()
fraud.Amount.describe()

# Compare the values for both transactions
credit_card_data.groupby('Class').mean()

# Under-Sampling(Build a sample dataset containing similar distribution of normal and fraudulant transactions)
# Number of fraudulent transactions -> 492
legit_sample = legit.sample(n=492, random_state=1)
new_dataset = pd.concat([legit_sample, fraud], axis=0)

# Checking if sample distribution is similar to original dataset
new_dataset.groupby('Class').mean()

# Splitting the data into features and target variable
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

# Spllitting the data into training and testing data
# 80% training data and 20% testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,stratify=Y,random_state=2)

# Model Training
model = LogisticRegression()

# Training the Logistic Regression model with training data
model.fit(X_train, Y_train)

# Model Evaluation
# Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy on training data : ', training_data_accuracy)

# Accuracy on testing data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy on testing data : ', test_data_accuracy)

# If the value of accuracy score in training data is greater than testing data, then the model is overfitting.
# If the value of accuracy score in training data is less than testing data, then the model is underfitting.
# If the value of accuracy score in training data is equal/near-equal to testing data, then the model is well fitted.