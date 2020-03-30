from sklearn.model_selection import train_test_split
# Feature Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



dataset = pd.read_csv("./dataset_report.csv")
print(dataset.head())


meta_headers = ['Cropped Frame','HOG','Original Frame','Laplace of Gaussian','Centroid','Linear Binary Patterns']

# TODO make make hog, log and


meta_data = dataset.filter(meta_headers, axis=1)

dataset.drop(meta_headers, axis=1, inplace=True)

dataset= dataset.replace("Healthy",0)
dataset = dataset.replace("Damaged",1)
dataset = np.array(dataset,dtype=float)
np.random.shuffle(dataset)
splits=10

# data is an array with our already pre-processed dataset examples
kf = KFold(n_splits=splits)
sum = 0



for train, test in kf.split(dataset):
    X_train = dataset[train][:,:-1]
    X_test = dataset[test][:,:-1]
    y_train = dataset[train][:,-1]
    y_test = dataset[test][:,-1]

    regressor = RandomForestClassifier(n_estimators=100, random_state=0, max_depth=10)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    sum = sum + accuracy

    print(confusion_matrix(y_test, y_pred,))
    print(classification_report(y_test, y_pred))
    print(accuracy)


average = sum/splits
print(average)





