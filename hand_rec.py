print '\nHello \n'

from skimage import io
import os
import numpy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn import svm
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


def train():

    d = load_images_from_folder('./trial_data/cropped')
    data = numpy.array(d)
    n_values,h,w =  data.shape

    X_HOG = numpy.array(hog_extract(data))
    print "Training Data Dimensions: \n"+ str(X_HOG.shape)


    labels = [i for i in range(1,6)]
    no_of_label = n_values/len(labels)

    y = numpy.array(createLabels(labels,no_of_label))

    split = train_test_split(X_HOG,y,test_size=0.25,random_state=42)
    X_train = split[0]
    X_test = split[1]
    y_train = split[2]
    y_test = split[3]

    n_components = 150

    pca = PCA(n_components=n_components, random_state=42).fit(X_train)

    X_train_pca = pca.transform(X_train)
    X_test_pca  = pca.transform(X_test)

    clf = svm.SVC(probability=True)
    params = {'kernel':['rbf','linear','poly'],'C':[0.1,1.0,0.5,0.01]}

    cv = KFold(n_splits=10,random_state=42)

    grid = GridSearchCV(clf,params,cv=cv)
    grid.fit(X_train_pca,y_train)
    print "Train Score: %lf\n"  %grid.best_score_

    best_clf = grid.best_estimator_
    print "Classifier: \n" + str(best_clf)

    clf_filename = 'clf.sav'
    joblib.dump(best_clf, clf_filename)

    pca_filename = 'pca.sav'
    joblib.dump(pca,pca_filename)

    y_pred = best_clf.predict(X_test_pca)
    print "Test Score: %lf\n"  %accuracy_score(y_test,y_pred)
    print "Confusion Matrix: "
    print confusion_matrix(y_test,y_pred)

    print '\n\n' + "Classifier Training Done.\n\n"

    test_folder(best_clf,pca)



images = []

def load_images_from_folder(folder):

    for filename in os.listdir(folder):
        if(filename == '.DS_Store'):
            continue
        if os.path.isdir(os.path.join(folder,filename)):
            load_images_from_folder(os.path.join(folder,filename))
        if os.path.isdir(os.path.join(folder,filename)) == False:
            img = io.imread(os.path.join(folder,filename),as_grey=True)
            if img is not None:
                images.append(img)

    return images


def createLabels(labels,no_of_label):

    y = []
    for i in labels:
        for j in range(no_of_label):
            y.append(str(i))
    return y

def hog_extract(data):

    new_data = []

    for img in data:
        fd = hog(img, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(2, 2), visualise=False)
        new_data.append(fd)

    return new_data

def find_max(prob):
    local_max = 0
    pos = 0
    for i in range(5):
        if prob[0][i] > local_max:
            local_max = prob[0][i]
            pos = i+1
    return (local_max,pos)


def sliding_window(image, stepSize, windowSize):
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def prediction(clf,pca,image,show=False):

    prob_max = 0
    best_window = None
    scale = 1.1
    (winW, winH) = (128, 128)

    for (i, resized) in enumerate(pyramid_gaussian(image,max_layer=3, downscale=scale)):

        if resized.shape[0] < 30 or resized.shape[1] < 30:
            break


        for (x, y, window) in sliding_window(resized, stepSize=10, windowSize=(winW, winH)):

            if window.shape[0] != winH or window.shape[1] != winW:
			    continue


            #window_pca = pca.transform(window.reshape(1,128*128))
            window_hog = hog_extract([window])
            window_pca = pca.transform(window_hog)
            prob = clf.predict_proba(window_pca)
            local_prob_max, pred = find_max(prob)

            if local_prob_max > prob_max:
                prob_max = local_prob_max
                best_window = window_pca
                best_resized = resized
                best_x = x
                best_y = y
                best_prediction = pred

            if show == True:
                clone = resized.copy()
                cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
                cv2.imshow("Window", clone)
                cv2.waitKey(1)
                time.sleep(0.025)

    if show == True:
        print 'The prediction is: '
        print best_prediction
        print 'With a probabiity of: '
        print prob_max
        clone = best_resized.copy()
        cv2.rectangle(clone, (best_x, best_y), (best_x + winW, best_y + winH), (0, 255, 0), 2)
        cv2.imshow("Window", clone)
        cv2.waitKey(0)
    else:
        return str(best_prediction)

def test_folder(clf,pca):
    a = time.clock()
    test_pred_list = []
    test_d = load_images_from_folder('./trial_data/raw')
    test_data = numpy.array(test_d)
    labels = [i for i in range(1,6)]
    no_of_label = 1000/len(labels)
    test_y = numpy.array(createLabels(labels,no_of_label))
    print "Number of images tested: "
    for img in test_data:
        test_pred_list.append(prediction(clf,pca,img))
        if(len(test_pred_list)%100==0):
            print len(test_pred_list)
        if(len(test_pred_list)==1000):
            break

    print "\nTime taken:"
    print time.clock()-a
    print "\nTest Done, Score:"
    print str(accuracy_score(test_y,test_pred_list))+'\n'
    print confusion_matrix(test_y,test_pred_list)


if __name__ == '__main__':
    train()
