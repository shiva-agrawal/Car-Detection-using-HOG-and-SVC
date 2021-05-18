'''
Developer: M.Sc. Shiva Agrawal
Place: Germany
Version: 1.0
Date: 05.07.2019
Description:  Vehicle (here car) detection algorithm using HOG and SVC
              HOG - Histogram of Oriented Gradient ( computer vision algorithm to extract features from an image)
              SVC - Support Vector Classifier (a machine learning classification algorithm)

Steps:
1. Dataset of images with car and without car are imported.
2. Each image is converted into grayscale.
3. HOG features are extracted
4. All the features are accumulated, labelled and split into train and test dataset
5. SVC model is trained using training dataset
6. Confusion matrix and classification report is generated for accuracy of the model
7. the model is saved using pickle
'''

import cv2 as cv
import numpy as np
from skimage.feature import hog
import glob # to parse through the folders for extracting data. here it is used for reading the dataset of car and non car
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report
import pickle

#-----------------------------------------------------------------------------------------------------------------------
# function
#-----------------------------------------------------------------------------------------------------------------------
def getHOGfeatures(image_color):
    # function extracts the total 396 features with 11 orientations from one image
    # It makes a vector of all the features. This happens because the paramter 'feature_vector=True' is set
    image_gray = cv.cvtColor(image_color,cv.COLOR_BGR2GRAY)
    features, hog_image = hog(image_gray,
                              orientations=11,
                              pixels_per_cell=(16,16),
                              cells_per_block=(2,2),
                              transform_sqrt=False,
                              visualise=True,
                              feature_vector=True)

    return features

#-----------------------------------------------------------------------------------------------------------------------
# main script
#-----------------------------------------------------------------------------------------------------------------------

## import the data
car = glob.glob('data/car/**/*.png') # import local path of all the images
no_car = glob.glob('data/no car/**/*.png') # import local path of all the images

## extract features using HOG and accumulate all of them

# 1. for car images
car_hog_featues_acc = []
for car_image in car:
    image_color = cv.imread(car_image)
    car_hog_featues_acc.append(getHOGfeatures(image_color))
    # car_hog_features_acc - a list of length  = total car images and each element of list is an array of 396 HOG features
    # with np.vstack - the list is converted into numpy array where each array of 396 featues are stacked vertically.
    # So now x_car is of size (car images count, 396).
    # Hence it is a 2D matrix in form of numpy 2D array where each row contains features from each image and each column contain one feature.

x_car = np.vstack(car_hog_featues_acc).astype(np.float64)
# as all these images contains car, they are label as 1. Here y_car is of size (car images count, 1)
y_car = np.ones(len(x_car))

# 2. for no car images
nocar_hog_featues_acc = []
for no_car_image in no_car:
    image_color = cv.imread(no_car_image)
    nocar_hog_featues_acc.append(getHOGfeatures(image_color))

# same logic as stated above is applied here
x_nocar = np.vstack(nocar_hog_featues_acc).astype(np.float64)
y_nocar = np.zeros(len(x_nocar))

# combine all the data together with their labels
x = np.vstack((x_car,x_nocar))
y = np.hstack((y_car,y_nocar))
print(x.shape,y.shape)

## Train the SVC (support Vector Classifier) and complete th model

# split the train and test dataset randomly for training and validation of the model
[X_train, X_test, Y_train, Y_test] = train_test_split(x,y,test_size=0.25,random_state= 7)

# train the SVC model
svc_model = LinearSVC()
svc_model.fit(X_train,Y_train)

# metrics : confusion matrix and classification report of the trained model
y_predict = svc_model.predict(X_test)
matrix = confusion_matrix(Y_test,y_predict)
print(matrix)
print(classification_report(Y_test,y_predict))

# save the model
svcModelFileName = 'carDetectorSVCModel.sav'
pickle.dump(svc_model, open(svcModelFileName,'wb'))



