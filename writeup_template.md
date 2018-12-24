# **Traffic Sign Recognition** 

## Writeup


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/gayaviswan/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4416
* The size of test set is 12620
* The shape of a traffic sign image is [32,32,3]
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a pie chart showing how the data ...

![Training Set](./pie_train_set.png)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I normalized the image data because it helps bring the image range of intensity values that makes visual sense.

I decided to generate additional data because some the classes had very few data (< 200). I made sure there are atleast 200 data points by flipping and translating the image.

To add more data to the the data set, I used the following techniques because it views the image from a different angle/side. It also trains the model to accept the traffic sign to appear anywhere on the image. 

Here is an example of an original image and an augmented image:

![Flipped Image](./flip.png)
![Translated Image](./translate.png)

The difference between the original data set and the augmented data set is the following 
X, y shapes: (34799, 32, 32, 3) (34799,)
X, y shapes: (34859, 32, 32, 3) (34859,)


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | 1x1 stride , valid padding, outputs 10x10x16	|
| RELU					|												|
| Max Pooling           | 2x2 stride, outputs 5x5x16                    |
| Fully connected		|  Input 400, Output: 120   					|
| RELU					|												|
| Fully connected		|  Input 120, Output: 84    					|
| RELU					|												|
| Fully connected		|  Input 84, Output: 43     					|
| Softmax				|           									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer with batch size of 127, number of epoch of 60 along with learning rate of 0.008

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 
* validation set accuracy of 0.965 
* test set accuracy of 0.942

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

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![image1](./my_images/2.png) 
![image2](./my_images/3.png) 
![image3](./my_images/8.png) 
![image4](./my_images/9.png) 
![image5](./my_images/10.png)

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No Entry      		| No Entry   									| 
| Caution       	    | General Caution  							    |
| Road Work 			| Road Work 									|
| 60 km/h	      		| 60  km/h  					 				|
| Priority Road			| Priority Road      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 0.942

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 73rd cell of the Ipython notebook.

For the first image, the model concludes that it is a No Entry sign (probability of 0.46), and the image does contain No Entry sign. The top five soft max probabilities were


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.46      			| No Entry   									| 
| 0.33  				| Turn left ahead								|
| 0.26  			    | Turn right ahead  							|
| 0.26  	   			| Yield   					 				    |
| 0.24 				    | No passing        							|


For the second image, the model concludes that it is a General Caution sign (probability of 0.43), and the image does contain Caution. The top five soft max probabilities were


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.43      			| General Caution  							    | 
| 0.40  				| Traffic Signals								|
| 0.25  			    | Keep right        							|
| 0.20  	   			| Right-of-way to the next intersection		    |
| 0.18 				    | Go straight or left							|

For the third image, the model concludes that it is a 60 km/h sign (probability of 0.29), and the image does contain 60 km/h sign. The top five soft max probabilities were


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.29      			| Speed Limit (60 km/h) 						| 
| 0.21  				| Speed Limit (80 km/h) 						|
| 0.16  			    | 31 wild animals crossing  					|
| 0.14  	   			| Road Work 				 				    |
| 0.12 				    | Keep Right        							|

For the fourth image, the model concludes that it is a Road work (probability of 0.30), and the image does contain Road work sign. The top five soft max probabilities were


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.30      			| Road Work  									| 
| 0.17  				| Bumpy road   								    |
| 0.15  			    | Keep right   							        |
| 0.12  	   			| Yield   					 				    |
| 0.10 				    | Beware of ice/snow        					|

For the fifth image, the model concludes that it is a priority road (probability of 0.58), and the image does contain Priority road. The top five soft max probabilities were



| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.58      			| Priority road     							| 
| 0.22  				| No entry      								|
| 0.20  			    | Ahead only        							|
| 0.20  	   			| Road work     			 				    |
| 0.18 				    | Yield             							|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


