#importing the fundamental libraries
import pandas as pd
import numpy as np
import tensorflow as tf

#importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

#label encoding the 'Gender' column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

#one hot encoding the 'Geography' column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [1])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))

#splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#initializing the ann
ann = tf.keras.models.Sequential()

#adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))

#adding the second hidden layer
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))

#adding the output layer
ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))

#compiling the ann
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#training the ann on the training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

#predicting the result of a single observation
print(f"Single prediction: {ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5} \n")

#predicting the test results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)

#making the confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion matrix: \n {cm} \n")
print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")