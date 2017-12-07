## **Traffic Sign Recognition** 


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./test_images/test_image_01_sign_1.jpg "Traffic Sign 1"
[image2]: ./test_images/test_image_02_sign_13.jpg "Traffic Sign 2"
[image3]: ./test_images/test_image_03_sign_25.jpg "Traffic Sign 3"
[image4]: ./test_images/test_image_04_sign_14.jpg "Traffic Sign 4"
[image5]: ./test_images/test_image_05_sign_1.jpg "Traffic Sign 5"
[image6]: ./test_images/test_image_06_sign_1.jpg "Traffic Sign 6"
[image7]: ./test_images/test_image_07_sign_28.jpg "Traffic Sign 7"
[image8]: ./test_images/test_image_08_sign_2.jpg "Traffic Sign 8"
[image9]: ./test_images/test_image_09_sign_4.jpg "Traffic Sign 9"
[image10]: ./test_images/test_image_10_sign_4.jpg "Traffic Sign 10"
[image11]: ./test_images/test_image_11_sign_7.jpg "Traffic Sign 11"
[image12]: ./test_images/test_image_12_sign_5.jpg "Traffic Sign 12"
[image13]: ./test_images/test_image_13_sign_8.jpg "Traffic Sign 13"
[image14]: ./test_images/test_image_14_sign_3.jpg "Traffic Sign 14"
[image15]: ./test_images/test_image_15_sign_0.jpg "Traffic Sign 15"
[image16]: ./test_images/test_image_16_sign_39.jpg "Traffic Sign 16"
[image17]: ./test_images/test_image_17_sign_38.jpg "Traffic Sign 17"
[image18]: ./test_images/test_image_18_sign_9.jpg "Traffic Sign 18"
[image19]: ./test_images/test_image_19_sign_28.jpg "Traffic Sign 19"
[image20]: ./test_images/test_image_20_sign_14.jpg "Traffic Sign 20"



---
## Writeup / README


### Data Set Summary & Exploration

#### 1. Basic Summary of Data Set

The numpy library were used to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 12630
* The size of test set is 4410
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

The number of unique classes/labels requires an explanation. This number signifies that we have a total of 43 different possible traffic signs in the given dataset.

#### 2. Exploratory visualization of the dataset.

I proceeded to perform some basic exploration of the data. I printed out the various traffic sign names. It can be seen in my ipython notebook that i have a small function to do this.

I then proceeded to plot a histogram of the various training set traffic signs.

Finally i choose 20 random images from the training set to output the images using matplotlib.

### Design and Test a Model Architecture

#### 1. Image Preprocessing

I initially tried to convert the images to grayscale but after trial and error i came to the conclusion that converting the images to grayscale did not really help my model performance. This came as a surprise.

Next, i decided to normalize by subtracting each data point by 128 and dividing by 128.
I did improve my model performance with this but while researching i came across another method which seemed to give me better performance. I use the mean and standard deviation and preprocess my data in the following way.

X_train = (X_train_pp - np.mean(X_train_pp))/np.std(X_train_pp)

X_valid = (X_valid_pp - np.mean(X_valid_pp))/np.std(X_valid_pp)

X_test  = (X_test_pp  - np.mean(X_test_pp))/np.std(X_test_pp)


I also came across keras preprocessing techniques where there are several preprocessing options available. I will probably try these out at some point of time.

For my submission, i keep things simeple by just performing the normalization.


####  2. Model Architecture

First i tried one run of my project using the default Lenet model. This can still be seen in the project notebook. After having gone through this process, i proceeded to have a model named TrafficNet.

My final model consisted of the following layers:
I used the the same mu and sigma as was suggested in the LeNet model in example lab.

I leveraged 3 convolution layers and 3 fully connected layers in my model.
One thing i notice much to my surprise was when i used dropout, i didn't get any improvement.
Some of the remnants can still be seen in the code.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x64	|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x128	|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 3x3x256 	|
| RELU					|												|
| Max pooling	      	| 1x1 stride, valid padding, outputs 2x2x256	|
| Flatten				| 1024 outputs									|
| Fully connected		| 1024 inputs 120 outputs						|
| RELU					|												|
| Fully connected		| 120  inputs  84 outputs						|
| RELU					|												|
| Fully connected		| 84   inputs  43 outputs						|
| Softmax				| Softmax Cross Entropy with Logits 			|
|						|												|
|						|												|
 


#### 3. Model Training

To train the model i used a batch size of 128 and i used 50 epochs.
I used the AdamOptimzer with a learning rate of 0.001
As mentioned previously i used a mean of 0 and standard deviation of 0.1 for training.
I perform one hot encoding with number of classes equal to 43. The number of 43 is important since this is the number of different traffic signs in the training data set.

A close look at the various individual code cells will show that i essentially leveraged the variosu code cells from the LeNet lab.

I performed training on an Amazaon EC2 GPU instance. 

#### 4. Solution Approach

I first started of with the Lenet architecture itself.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.954
* test set accuracy of 0.963

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3] 
![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8] ![alt text][image9] 
![alt text][image10] ![alt text][image11] ![alt text][image12]
![alt text][image13] ![alt text][image14] ![alt text][image15]
![alt text][image16] ![alt text][image17] ![alt text][image18] 
![alt text][image19] ![alt text][image20]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 




