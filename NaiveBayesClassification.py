#Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler

#Read the dataset
dataset = pd.read_csv(r'data/iris.csv')
print(dataset.head(5))

#NA values in the dataset
print("Count of NA values:")
print(dataset.isna().sum())

#Filter independent and dependent variables
X = dataset.iloc[:, :4].values
y = dataset.iloc[:, [4]].values

#Preprocess the dataset
#X = StandardScaler().fit_transform(X)
X = Normalizer().fit_transform(X)
#X = MinMaxScaler().fit_transform(X)

#Split data into training and testing split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0, stratify = y)

#Train a classifier
classifier = GaussianNB()
classifier.fit(X_train, y_train.ravel())

#Predcitions
y_pred = classifier.predict(X_test)

#Results
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)
print("Accuracy:", accuracy_score(y_test, y_pred))