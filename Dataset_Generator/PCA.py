from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler




'''

PCA by Daniel Mcdonough

This script reads the report produced by the 
classifier and plots it over 
a 2d PCA of the selected features


'''



dataset = pd.read_csv("./dataset_ensable_predclasses.csv")


meta_headers = ['Cropped Frame','Original Frame', "HOG","Laplace of Gaussian","Gabor Wavelet","Centroid_x","Centroid_y"]
SIFT = dataset.filter(regex=("SIFT.*"))
LBP = dataset.filter(regex=("Linear Binary Patterns.*"))
# dataset.drop(SIFT, axis=1, inplace=True)
# dataset.drop(LBP, axis=1, inplace=True)

meta_data = dataset.filter(meta_headers, axis=1)
dataset.drop(meta_headers, axis=1, inplace=True)


dataset= dataset.replace("Healthy",1)
dataset = dataset.replace("Damaged",0)

dataset = dataset.fillna(0)


# move classes to first column
TC = dataset['True Classification']
dataset.drop(labels=['True Classification'], axis=1, inplace = True)

PC = dataset["Pred Classifications"]
dataset.drop(labels=['Pred Classifications'], axis=1, inplace = True)

#convet True classes and predicted classes into 1,2,3,4 based on TP,TN,FP,FN

PC = np.array(PC,dtype=int)
TC = np.array(TC,dtype=int)
color_labels = np.empty(shape=(TC.shape[0],1))

for data_point in range(TC.shape[0]):
    if TC[data_point] == 1 and PC[data_point] == 1:
        # True Positive
        color_labels[data_point] = 0
    elif TC[data_point] == 0 and PC[data_point] == 1:
        # False Negative
        color_labels[data_point] = 1
    elif TC[data_point] == 1 and PC[data_point] == 0:
        # False Positive
        color_labels[data_point] = 2
    else:
        # True Negative
        color_labels[data_point] = 3

print(color_labels)
color_df = dataset.copy(deep=True)
color_df.insert(0, 'Color Classification', color_labels)

color_df = np.array(color_df,dtype=float)
# Initialise the Scaler
scaler = MinMaxScaler()
scaler.fit(color_df)
color_df = scaler.transform(color_df)



# Shuffle the data so that the K folds are not to be influenced by the original frame
np.random.shuffle(color_df)

pca = PCA(n_components=2)

X_train = color_df[:,1:]
X_test = color_df[:,1:]
y_train = color_df[:,0]
y_test = color_df[:,0]

pca.fit(X_train)
X_test = pca.transform(X_test)

# import matplotlib.collections.PathCollection.legend_elements

# print(X_train)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('2 component PCA', fontsize=20)
targets = ['TP', 'FN','FP','TN']
colors = ['g', 'y', 'b','r']


scatter = ax.scatter(X_test[:,0]
           , X_test[:,1]
           , c=y_test
           , alpha= 0.3
           , s=25
           ,cmap= matplotlib.colors.ListedColormap(colors))

# todo make legends

bounds = np.linspace(0,3,4)
cb = plt.colorbar(scatter, spacing='proportional',ticks=bounds)
# cb.set_label('Custom cbar')

# ax.legend(targets[0])
ax.grid()
plt.show()




dataset.insert(0, 'True Classification', TC)

dataset = np.array(dataset,dtype=float)

# Initialise the Scaler
scaler = MinMaxScaler()
scaler.fit(dataset)
dataset = scaler.transform(dataset)



# Shuffle the data so that the K folds are not to be influenced by the original frame
np.random.shuffle(dataset)



pca = PCA(n_components=2)

X_train = dataset[:,1:]
X_test = dataset[:,1:]
y_train = dataset[:,0]
y_test = dataset[:,0]

pca.fit(X_train)
X_test = pca.transform(X_test)


# print(X_train)
fig1 = plt.figure(figsize=(8, 8))
ax1 = fig1.add_subplot(1, 1, 1)
ax1.set_xlabel('Principal Component 1', fontsize=15)
ax1.set_ylabel('Principal Component 2', fontsize=15)
ax1.set_title('2 component PCA', fontsize=20)
targets = ['TP', 'TN']
colors = ['r', 'g']

# for target, color in zip(targets, colors):
scatter = ax1.scatter(X_test[:,0]
           , X_test[:,1]
           , c= y_test
           , s= 25
           , alpha= 0.3
           , cmap= matplotlib.colors.ListedColormap(colors))

bounds = np.linspace(0,1,2)
cb = plt.colorbar(scatter, spacing='proportional',ticks=bounds)
ax1.grid()
plt.show()
