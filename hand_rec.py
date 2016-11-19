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

    ap = argparse.ArgumentParser()

    #Folder path must point trial_data (dropbox link in README)

    ap.add_argument("--folder",required=True, help="Folder path for trial_data")

    args = ap.parse_args()
    folder_path = args.folder

    # Cropped directory in trial_data is used for training, and raw directory is used for tesing
    train_folder_path = os.path.join(folder_path,"cropped")
    test_folder_path = os.path.join(folder_path,"raw")


    d = load_images_from_folder(train_folder_path)
    data = numpy.array(d)
    n_values,h,w =  data.shape

    # Extract HOG descriptors from image data
    X_HOG = numpy.array(hog_extract(data))
    print "Training Data Dimensions: \n"+ str(X_HOG.shape)+'\n'

    # Labels for the training data
    labels = [i for i in range(1,6)]
    no_of_label = n_values/len(labels)

    y = numpy.array(createLabels(labels,no_of_label))

    # Splits the dataset into training and testing data in the ratio 3:1
    split = train_test_split(X_HOG,y,test_size=0.25,random_state=42)
    X_train = split[0]
    X_test = split[1]
    y_train = split[2]
    y_test = split[3]

    n_components = 150

    # Dimensional reduction of features using pca from n to 150
    pca = PCA(n_components=n_components, random_state=42).fit(X_train)

    X_train_pca = pca.transform(X_train)
    X_test_pca  = pca.transform(X_test)

    # Setting up a SVM classifier with probability true, so that when a prediction is made by the classifier,
    # even the confidence of the predictions returned
    clf = svm.SVC(probability=True)

    # Setting up the parameter grid for the GridSearchCV and the 10 folds for cross validation
    params = {'kernel':['rbf','linear','poly'],'C':[0.1,1.0,0.5,0.01]}
    cv = KFold(n_splits=10,random_state=42)

    grid = GridSearchCV(clf,params,cv=cv)
    grid.fit(X_train_pca,y_train)
    print "Train Score: %lf\n"  %grid.best_score_

    # Best classifier from GridSearchCV is stored into variable called best_clf
    best_clf = grid.best_estimator_
    print "Classifier: \n" + str(best_clf)+'\n'

    # Trained classifier and pca is being stored in separate modules so they may be imported in test.py
    clf_filename = 'clf.sav'
    joblib.dump(best_clf, clf_filename)

    pca_filename = 'pca.sav'
    joblib.dump(pca,pca_filename)

    y_pred = best_clf.predict(X_test_pca)
    print "Test Score: %lf\n"  %accuracy_score(y_test,y_pred)
    print "Confusion Matrix: "
    print confusion_matrix(y_test,y_pred)

    print '\n\n' + "Classifier Training Done.\n\n"

    # Testing the classifier against the test folder
    test_folder(best_clf,pca,test_folder_path)



images = []


# Loads all images from a directory recursively so all images in child directories also get loaded into images
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

# Extract HOG descriptors for each image in dataset
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

# Function for generating a moving window of windowsize in image with steps equal to stepsize
def sliding_window(image, stepSize, windowSize):
	for y in xrange(0, image.shape[0], stepSize):
		for x in xrange(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


# If show = True, then it shows the window, on the image, which has captured the hand gesture correctly
def prediction(clf,pca,image,show=False):

    prob_max = 0
    best_window = None
    scale = 1.1
    (winW, winH) = (128, 128)

    # Creating an image pyramid using pyramid_gaussian with a depth of four layers which each downscale equals 1.1
    for (i, resized) in enumerate(pyramid_gaussian(image,max_layer=3, downscale=scale)):

        # Break if scaled image is too small
        if resized.shape[0] < 30 or resized.shape[1] < 30:
            break


        for (x, y, window) in sliding_window(resized, stepSize=10, windowSize=(winW, winH)):

            if window.shape[0] != winH or window.shape[1] != winW:
			    continue

            # A prediction is made for every window generated by the sliding window with confidence of prediction
            window_hog = hog_extract([window])
            window_pca = pca.transform(window_hog)
            prob = clf.predict_proba(window_pca)
            local_prob_max, pred = find_max(prob)

            # Storing which prediction for a particular window has the most confidence, and relevant data about that window is saved
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
        # The prediction with the most confidence from all windows is returned.
        return str(best_prediction)

def test_folder(clf,pca,test_folder):
    a = time.clock()
    test_d = load_images_from_folder(test_folder)
    test_data = numpy.array(test_d)
    labels = [i for i in range(1,6)]
    no_of_label = 1000/len(labels)
    test_y = numpy.array(createLabels(labels,no_of_label))

    test_pred_list = []

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
    print "Confusion Matrix: "
    print confusion_matrix(test_y,test_pred_list)


if __name__ == '__main__':
    train()
