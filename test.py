from skimage import io
import os
import numpy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn import svm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import cv2
from skimage.transform import pyramid_gaussian
import argparse
import time
from sklearn.externals import joblib
from skimage.feature import hog
from hand_rec import prediction,hog_extract,load_images_from_folder,createLabels,find_max,sliding_window

best_clf = joblib.load("clf.sav")
pca = joblib.load("pca.sav")

ap = argparse.ArgumentParser()

ap.add_argument("--image",help="Image path")
ap.add_argument("--folder",help="Folder path")

args = ap.parse_args()

if args.image is not None:
    test_image = io.imread(args.image,as_grey=True)
elif args.folder is not None:
    folder_path = args.folder
else:
    print "No image to test"
    exit()


labels = [i for i in range(1,6)]
no_of_label = 1000/len(labels)


# Test
# test_pred_list = []
# test_d = load_images_from_folder('./trial_data/raw')
# test_data = numpy.array(test_d)
# test_y = numpy.array(createLabels(labels,no_of_label))
# for img in test_data:
#     test_pred_list.append(prediction(img))
#     if(len(test_pred_list)%100==0):
#         print len(test_pred_list)
#     if(len(test_pred_list)==1000):
#         break
#
# print "Test Done, Score:"
# print accuracy_score(test_y,test_pred_list)
# print confusion_matrix(test_y,test_pred_list)

resized_image = cv2.resize(test_image,(320,240))
prediction(best_clf,pca,resized_image,show=True)
