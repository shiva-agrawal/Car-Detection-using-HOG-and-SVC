# Car Detection classifier using HOG and SVC - Algorithms development and testing

More than 10,000 images of car and non car are used as dataset to develop the algorithm for binary classifier. 
Generally, CNN are use to develop the algorithms where the images can be directly fed as input. As images are of large size, 
it increases total computation time to train the model.

Another approach (without CNN) using HOG and SVC is developed here. As stated above, image is generally large and each pixel 
cannot be input into the system, so at first phase each image is fed into HOG feature extraction, where only features are extracted.
Then in 2nd phase, this features are accumulated, labelled and fed into Support Vector Classifier to train the model. 

Once the model is trained, its accuracy is examined using confusion matrix and classification report. 
The model is found to be 97% accurate.

Below is the block digram for the steps.

![Block diagram](https://github.com/shiva-agrawal/autonomous_driving/blob/master/02_Perception/Car%20Detection%20using%20HOG%20and%20SVC/processs_flow_diagram.JPG)

For understanding of the HOG feature extraction, a demo script along with input and output image is also added to the repository. 
(Refer Folder - HOG feature extraction_demo)

The same is also shown below.

![HOG](https://github.com/shiva-agrawal/autonomous_driving/blob/master/02_Perception/Car%20Detection%20using%20HOG%20and%20SVC/HOG%20feature%20extraction_demo/HOG_block_diagram.png)

## Folder details:
* HOG feature extraction_demo - folder containing python script and images for HOG algorithm demo.
* Vehicle Detection suing HOG and SVC.py - classifier algorithm
* carDetectorSVCModel.sav - saved model using pickle 
* processs_flow_diagram.JPG - block diagram shown above

## References:
* [HOG - Histogram of Oriented Gradients](https://www.learnopencv.com/histogram-of-oriented-gradients/)
* [SVM - Support Vector Machine](https://scikit-learn.org/stable/modules/svm.html)
