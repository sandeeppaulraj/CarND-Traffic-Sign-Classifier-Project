## **Traffic Sign Recognition** 


---

[//]: # (Image References)

[image1]: ./test_images/test_image_01_sign_1.jpg "Traffic Sign 1"
[image2]: ./test_images/test_image_02_sign_13.jpg "Traffic Sign 2"
[image3]: ./test_images/test_image_03_sign_25.jpg "Traffic Sign 3"
[image4]: ./test_images/test_image_04_sign_14.jpg "Traffic Sign 4"
[image5]: ./test_images/test_image_05_sign_1.JPG "Traffic Sign 5"
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

The numpy library were used to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 12630
* The size of test set is 4410
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

The number of unique classes/labels requires an explanation. This number signifies that we have a total of 43 different possible traffic signs in the given dataset.

#### 2. Exploratory visualization of the dataset.

I proceeded to perform some basic exploration of the data. I printed out the various traffic sign names. It can be seen in my ipython notebook that i have a small function to do this.

I then proceeded to plot a histogram of the various training set traffic signs.

Finally i chose 20 random images from the training set to output the images using matplotlib.

### Design and Test a Model Architecture

#### 1. Image Preprocessing

I initially tried to convert the images to grayscale but after trial and error I came to the conclusion that converting the images to grayscale did not really help my model performance. This came as a surprise.

Next, i decided to normalize by subtracting each data point by 128 and dividing by 128.
I did improve my model performance with this but while researching I came across another method which seemed to give me better performance. I use the mean and standard deviation and preprocess my data in the following way.

X_train = (X_train_pp - np.mean(X_train_pp))/np.std(X_train_pp)

X_valid = (X_valid_pp - np.mean(X_valid_pp))/np.std(X_valid_pp)

X_test  = (X_test_pp  - np.mean(X_test_pp))/np.std(X_test_pp)


I also came across keras preprocessing techniques where there are several preprocessing options available. I will probably try these out at some point of time.

For my submission, i keep things simple by just performing the normalization.


####  2. Model Architecture

First i tried one run of my project using the default Lenet model. This can still be seen in the project notebook. After having gone through this process, i proceeded to make changes/updates and named my model TrafficNet.

My final model consisted of the following layers:
I used the the same mu and sigma as was suggested in the LeNet model in example lab.

I leveraged 3 convolution layers and 3 fully connected layers in my model.
One thing I noticed much to my surprise was when i used dropout, i didn't get any improvement.
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

To train the model I used a batch size of 128 and I used 50 epochs.
I used the AdamOptimzer with a learning rate of 0.001
As mentioned previously I used a mean of 0 and standard deviation of 0.1 for training.
I perform one hot encoding with number of classes equal to 43. The number of 43 is important since this is the number of different traffic signs in the training data set.

A close look at the various individual code cells will show that I essentially leveraged the various code cells from the LeNet lab.

I performed training on an Amazaon EC2 GPU instance. 

#### 4. Solution Approach

I first started of with the Lenet architecture itself that was suggested in the lab.
LeNet is a great starting point since it has already been extensively used even on grayscale MNIST data
However, with this, i could not achieve a validation set accuracy of even 0.9.I tried various dropout layers but that too did not increase my validation accuracy.

Having modelled a few other CNNs before in other projects, one particular direction which has helped me a lot is starting of with an initial filter size of 32 in the first convolution layer and then progressively increasing it by a power of 2. So the filter size for each convolution layer will be 32,64,128,etc. Along these lines, I experimented with various filter sizes and started seeing an immediate improvement. I finally settled for 3 convolution layers with filter sizes of 64,128 and 256.
The three convolution layers are all followed by a RELU activation layer followed by a MAX Pooling layer.This can be seen in my CNN model architecture. With this I went above the 0.93 threshold that was required for submission. I continued using 3 fully connected layers as well. Adding a fourth convolution layer did not help my validation accuracy. Surprisingly, even adding dropout layers did not help my model. Essentially i settled for three convolution layers and three fully connected layers.
Progressively increasing the filter size makes the network deeper and will help in extracting important features.


My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.954
* test set accuracy of 0.962

All my trial runs were based on just gauging the validation set accuracy. I was satisfied with my validation set accuracy of 0.954 and pleasantly surprised with a test set accuracy of 0.962
Before starting this journey, i set myself a personal target of 0.95 for the validation set accuracy, so i was very satisfied to go a little better. 

Some other things that i should have tried are trying to use "same" padding and also modifying the various kernel size parameters in the various layers.


### Test a Model on New Images

#### 1. Random German Traffic Signs from the internet

I decided to test my model on 20 German traffic signs that i found on the internet using google images to be precise. For most of my projects I test on more data than the prerequisite number of tests required in the project. In this case, I thought "five" was a low number and decided to test on twenty different images. I would also like to highlight the fact that one other reason to use twenty images was to have some "clear" images to gauge how well my model behaved with images from the internet. Some images do have watermarks and other issues which I will mention below.

The issue with the below image is that is has several poles around the sign so this could possibly cause issues

![alt text][image1]

In the below image, the traffic sign is actually above another sign although the sign below is truncated to a very large extent.

![alt text][image2]

The below image is a clear image with only scratches on the sign.

![alt text][image3]

In the below image, notice the blue stripe of another sign.

![alt text][image4]

In the below image part of the sign is missing from the top.

![alt text][image5]

The below image is a clear image

![alt text][image6]

The image below has a watermark

![alt text][image7]


The images below are very clear images.

![alt text][image8]

![alt text][image9] 

![alt text][image10]

![alt text][image11]

![alt text][image12]

![alt text][image13]

![alt text][image14]

![alt text][image15]

![alt text][image16]

![alt text][image17]

Notice the Sun shining in the bottom right hand corner

![alt text][image18] 

The image is different in the sense the left side has an image of a blue sky and the right side has a green tree.

![alt text][image19]

The below image is a clear sign

![alt text][image20]

#### 2. Performance on New Images

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h)	| Speed limit (30km/h)							| 
| Yield 				| Yield 										|
| Road work				| Road work										|
| Stop  				| Stop 							 				|
| Speed limit (30km/h)	| Speed limit (30km/h) 							|
| Speed limit (30km/h)	| Speed limit (30km/h) 							|
| Children crossing		| Children crossing								|
| Speed limit (50km/h)	| Speed limit (30km/h)							|
| Speed limit (70km/h)	| Speed limit (70km/h)							|
| Speed limit (70km/h)	| Speed limit (70km/h)							|
| Speed limit (100km/h)	| Speed limit (100km/h)							|
| Speed limit (80km/h)	| Speed limit (30km/h)							|
| Speed limit (120km/h)	| Speed limit (120km/h)							|
| Speed limit (60km/h)	| Speed limit (60km/h)							|
| Speed limit (20km/h)	| Speed limit (60km/h)							|
| Keep left				| Keep left										|
| Keep Right			| Keep Right									|
| No passing			| No passing									|
| Children crossing		| Children crossing								|
| Stop					| Stop											|


The model was able to correctly guess 17 of the 20 traffic signs, which gives an accuracy of 85%. This is below my test set accuracy of 96.2%.

I am surprised that the 3 differences are with images that represent a speed. An "80" can be mistaken as a "30" but the other 2 discrepanices mean that i need to improve my model. 

#### 3. Model Certainty - Softmax Probabilities

I will choose to discuss 5 test images instead of all the 20 images. 

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1 					| Speed limit (30km/h)   						| 
| 7.52507255e-14		| Speed limit (20km/h) 							|
| 2.34635306e-17		| Roundabout mandatory							|
| 2.17252112e-19		| Speed limit (50km/h)					 		|
| 4.12792529e-21		| Speed limit (120km/h) 						|


For the second image ...

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1 					| Speed limit (70km/h)   						| 
| 2.45097623e-11		| Speed limit (20km/h) 							|
| 2.10769678e-19		| Traffic signals								|
| 7.02596885e-27		| Speed limit (30km/h)			 				|
| 3.58023253e-32		| Stop 											|


For the third image ...

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1 					| Speed limit (60km/h)   						| 
| 6.80272505e-09		| No vehicles 									|
| 1.19147109e-10		| Bicycles crossing								|
| 5.57566979e-11		| Wild animals crossing			 				|
| 4.34783676e-12		| Stop 											|


For the fourth image ...

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1 					| Road work   									| 
| 1.21731280e-11		| Beware of ice/snow 							|
| 7.93273813e-13		| Bicycles crossing								|
| 6.87919163e-15		| Bumpy road					 				|
| 3.84299653e-15		| Priority road					 				|


For the fifth image ...

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1 					| Speed limit (30km/h)  						| 
| 4.64747761e-13		| Wild animals crossing 						|
| 1.36253408e-13		| Speed limit (50km/h)							|
| 3.49214892e-18		| Roundabout mandatory					 		|
| 1.46887163e-23		| Keep left      								|
