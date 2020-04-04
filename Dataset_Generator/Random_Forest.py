from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# todo append column to csv
def appendDFToCSV_void(df, csvFilePath, sep=","):

    if not os.path.isfile(csvFilePath):
        # old_df = []
        # old_df = np.append(old_df, df, axis=1)
        np.savetxt(csvFilePath, df.T, delimiter=sep)
    else:
        # load file csv
        old_df = np.genfromtxt(csvFilePath, delimiter=sep)
        if old_df.ndim  == 1:
            old_df = np.reshape(old_df,(-1,1))

        # append column
        old_df = np.concatenate((df,old_df), axis=1)

        # write back to file
        np.savetxt(csvFilePath, old_df, delimiter=sep)

report_location = "featureselection.csv"
if os.path.isfile(report_location):
    os.remove(report_location)
# NOTE we do not need to scale the features as we are using a decision based algorithm


dataset = pd.read_csv("./dataset_report.csv")


meta_headers = ['Cropped Frame','Original Frame']
meta_data = dataset.filter(meta_headers, axis=1)


dataset.drop(meta_headers, axis=1, inplace=True)

dataset= dataset.replace("Healthy",0)
dataset = dataset.replace("Damaged",1)

dataset = dataset.fillna(0)


# move classes to first column
mid = dataset['True Classification']
dataset.drop(labels=['True Classification'], axis=1,inplace = True)
dataset.insert(0, 'True Classification', mid)


dataset = np.array(dataset,dtype=float)




# Shuffle the data so that the K folds are not to be influenced by the original frame
np.random.shuffle(dataset)


splits=10
kf = KFold(n_splits=splits)
sum = 0

from sklearn import preprocessing

for train, test in kf.split(dataset):
    X_train = dataset[train][:,0:]

    X_test = dataset[test][:,0:]
    y_train = dataset[train][:,0]

    y_test = dataset[test][:,0]

    lab_enc = preprocessing.LabelEncoder()
    training_scores_encoded = lab_enc.fit_transform(y_train)


    regressor = RandomForestClassifier(n_estimators=10000, random_state=0, max_depth=10)
    regressor.fit(X_train, training_scores_encoded)
    y_pred = regressor.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    # print(regressor.feature_importances_)
    # get feature importance
    # importances =
    # print(importances)
    importances = np.reshape(regressor.feature_importances_, (-1, 1))

    appendDFToCSV_void(importances, report_location, sep=",")

    # .to_csv(,mode='a')
    sum = sum + accuracy

    print(confusion_matrix(y_test, y_pred,))
    # print(classification_report(y_test, y_pred))
    # print(accuracy)


average = sum/splits
print(average)





