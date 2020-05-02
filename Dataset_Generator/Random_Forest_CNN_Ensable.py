from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
from sklearn.feature_selection import SelectFromModel
from sklearn import preprocessing
from csv import writer
from skimage.transform import resize
import cv2


'''

Resnet + Random Forest
by Daniel Mcdonough

This is just random forest but it reads the RESNET probability too.
'''


def appendDFToCSV_void(df, csvFilePath, head):
    # if doesnt exist then make one
    if not os.path.isfile(csvFilePath):
        # print(head.shape)
        head = np.reshape(head, (1, -1))
        np.savetxt(csvFilePath, np.reshape(head,(1,-1)), delimiter=",", fmt='%s')
    list_of_elem = np.reshape(df, (-1, 1))
    # Open file in append mode
    with open(csvFilePath, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)




def calcMetrics(confusion_matrix):
    TP = confusion_matrix[0,0]
    FP = confusion_matrix[0,1]
    FN = confusion_matrix[1,0]
    TN = confusion_matrix[1,1]

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)

    fscore = 2*((precision*recall)/(precision+recall))
    FNR = FN/(TP+FN)
    FDR = FP/(TP+FP)
    SPC = TN/(FN+TN)
    ACC = (TP+TN)/(TP+FP+FN+TN)

    return fscore,FNR,FDR,SPC,ACC


splits=10
kf = KFold(n_splits=splits)
accuracy_sum = 0
fscore_sum = 0
FNR_sum = 0
FDR_sum = 0
SPC_sum = 0

report_location = "featureselection_2d.csv"
if os.path.isfile(report_location):
    os.remove(report_location)
# NOTE we do not need to scale the features as we are using a decision based algorithm


dataset = pd.read_csv("./Dataset_ensable_new.csv")

clone = dataset.copy(deep=True)


dataset= dataset.replace("Healthy",1)
dataset = dataset.replace("Damaged",0)

meta_headers = ['Cropped Frame','Original Frame', "HOG","Laplace of Gaussian","Gabor Wavelet","Centroid_x","Centroid_y"]
SIFT = dataset.filter(regex=("SIFT.*"))


LBP = dataset.filter(regex=("Linear Binary Patterns.*"))
dataset.drop(SIFT, axis=1, inplace=True)
dataset.drop(LBP, axis=1, inplace=True)
ZLM = dataset.filter(regex=("Zernlike Moments.*"))
dataset.drop(ZLM, axis=1, inplace=True)
# meta_data = dataset.filter(meta_headers, axis=1)



meta_data = dataset.filter(meta_headers, axis=1)
dataset.drop(meta_headers, axis=1, inplace=True)



dataset = dataset.fillna(0)


# move classes to first column
mid = dataset['True Classification']
# print(mid)
dataset.drop(labels=['True Classification'], axis=1,inplace = True)

head = np.array(dataset.columns.values)

dataset.insert(0, 'True Classification', mid)

from sklearn.utils import shuffle
df = shuffle(dataset)
# df.reset_index(inplace=True, drop=True)


dataset = np.array(df)

pred_classifications = []

# Shuffle the data so that the K folds are not to be influenced by the original frame
# np.random.shuffle(dataset)

X = dataset[:, 1:]
y = dataset[:, 0]

max_features = min(X.shape[1],X.shape[0])

import matplotlib.pyplot as plt
for train, test in kf.split(dataset):
    X_train = dataset[train][:,1:]

    X_test = dataset[test][:,1:]
    y_train = dataset[train][:,0]

    y_test = dataset[test][:,0]


    lab_enc = preprocessing.LabelEncoder()
    training_scores_encoded = lab_enc.fit_transform(y_train)


    regressor = RandomForestClassifier(n_estimators=100, random_state=2, max_depth=10, criterion='gini', max_features=max_features,)
    regressor.fit(X_train, training_scores_encoded)
    y_pred = regressor.predict(X_test)


    for prediction in y_pred:
        pred_classifications.append(prediction)

    '''
    importances = regressor.feature_importances_
    std = np.std([tree.feature_importances_ for tree in regressor.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %s (%f)" % (f + 1, head[indices[f]], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()
    
    '''




    conf_mat = confusion_matrix(y_test, y_pred)

    fscore, FNR, FDR, SPC, ACC = calcMetrics(conf_mat)


    accuracy_sum = accuracy_sum + ACC
    fscore_sum = fscore_sum + fscore
    FNR_sum = FNR_sum + FNR
    FDR_sum = FDR_sum + FDR
    SPC_sum = SPC_sum + SPC


accuracy_average = accuracy_sum/splits
fscore_avg = fscore_sum/splits
FNR_avg = FNR_sum/splits
FDR_avg = FDR_sum/splits
SPC_avg = SPC_sum/splits

print("Average F-score: " + str(fscore_avg))
print("Average Accuracy: " + str(accuracy_average))
print("Average False Negative Rate: " + str(FNR_avg))
print("Average Specificity: " + str(SPC_avg))
print("Average False Discovery Rate: " + str(FDR_avg))


# dataset = pd.read_csv("./Dataset_ensable_new.csv")
clone["Pred Classifications"] = pred_classifications
clone.to_csv("./dataset_ensable_predclasses.csv")

