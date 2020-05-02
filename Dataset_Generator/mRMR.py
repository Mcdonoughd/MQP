
import numpy as np
import os
import pandas as pd
import pymrmr

'''
mRMR by Daniel McDonough

Reading a report from the classifier, it prints the top features to use

'''

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

# report_location = "featureselection_1d.csv"
# if os.path.isfile(report_location):
#     os.remove(report_location)
# NOTE we do not need to scale the features as we are using a decision based algorithm


dataset = pd.read_csv("./Dataset_ensable_new.csv")


meta_headers = ['Cropped Frame','Original Frame', "HOG","Laplace of Gaussian","Gabor Wavelet","Centroid_x","Centroid_y"]
meta_data = dataset.filter(meta_headers, axis=1)
SIFT = dataset.filter(regex=("SIFT.*"))
LBP = dataset.filter(regex=("Linear Binary Patterns.*"))
dataset.drop(SIFT, axis=1, inplace=True)
dataset.drop(LBP, axis=1, inplace=True)

ZLM = dataset.filter(regex=("Zernlike Moments.*"))
dataset.drop(ZLM, axis=1, inplace=True)
dataset.drop(meta_headers, axis=1, inplace=True)
dataset.drop(labels=['True Classification'], axis=1,inplace = True)

dataset= dataset.replace("Healthy",-1)
dataset = dataset.replace("Damaged",1)

dataset.fillna(0,inplace=True)
# Print top features based on selection metrics
print(pymrmr.mRMR(dataset, 'MIQ',10))
print(pymrmr.mRMR(dataset, 'MID',10))






