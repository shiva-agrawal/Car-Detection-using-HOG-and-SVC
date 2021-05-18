'''
Developer: M.Sc. Shiva Agrawal
Place: Germany
Version: 1.0
Date: 03.07.2019
Description:  HOG feature extraction demo script
'''

import cv2 as cv
from skimage.feature import hog # for HOG algorithm
from skimage import exposure

# import color image and convert into grayscale
image_color = cv.imread('Truck_free_s.jpg')
cv.imshow('original_image',image_color)
image_gray = cv.cvtColor(image_color,cv.COLOR_BGR2GRAY)
cv.imshow('grayscale image',image_gray)
print(image_gray.shape)


# perform HOG feature extraction with 9 orientation vectors with cell size of 16 x 16
features, hog_image = hog(image_gray,orientations=9,
                          pixels_per_cell=(16,16),cells_per_block=(1,1),transform_sqrt=False, visualise=True,feature_vector=False)

print(features.shape)
print(features[12,12,:])

print(hog_image.shape)
cv.imshow('hog image',hog_image)

# return image after streching or shrinking its intensity levels.
hog_image_rescaled = exposure.rescale_intensity(hog_image,in_range=(0,5))
cv.imshow('hog image rescaled',hog_image_rescaled)
cv.waitKey(0)