**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model_arch.png "Nvidia Model"
[image2]: ./examples/model_summary.png "Model Summary"
[image3]: ./examples/center_1.jpg "Center Lane Driving Image"
[image4]: ./examples/left_recovery_1.jpg "Recovery Image"
[image5]: ./examples/right_recovery_1.jpg "Recovery Image"
[image6]: ./examples/three_cameras.png "Three Cameral Images"
[image7]: ./examples/flipped_image.png "Flipped Image"
[image8]: ./examples/bright_image.png "Artificial Brightness Augmentation"
[image9]: ./examples/trans_image.png "Image Translation Augmentation"
[image10]: ./examples/cropped_image.png "Image Cropping"
[video1]: ./examples/run3.mp4 "Project Solution"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md and writeup_report.pdf summarizing the results
* visualize_model.py to explore image processing and model 

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file was used to build and train the model. It contains everything needed to preprocess data, build the model, train and validate the model, and save the model. Comments are throughout to explain each function or section of code.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I adapted the model architecture from the End-To-End Learning network used by The Nvidia Coorperation, found here:
https://arxiv.org/pdf/1604.07316v1.pdf

I decided to use the model because it has proven performance for this type of behavior cloning. The paper does not mention dropout layers or activations, so there were some additions and changes I made to the model architecture to adapt it to work for this specific problem. These will be described in detail later.

I also experiemented with the Inception model from Google, but it was not selected for my final implementation.

####2. Attempts to reduce overfitting in the model

In an attempt to prevent overfitting, dropout layers were added to the model after each convolution and fully connected layer (model.py, Lines 102 and 111). The drouput frequency was tuned to obtain acceptable behavior. 

I also split my data set into seperate training and validation sets to ensure the model was generalizing such that similar performance was seen on both the training and validation sets (model.py, Lines 254, 272-273). The model was then tested using the simulator to make sure it could complete continuous laps of automated driving.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 116). I did tune the dropout rate (selected to be 0.4) and also determened the level of data augmentation needed.

####4. Appropriate training data

Training data was captured using the simulator provided by Udacity. Images and steering angles were captured while I manually drove around the track. I chose a combination of center lane driving, and recovery path driving from the edges of the track to train the model.  

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start from a known CNN that can achieve behavioral cloning for autonomous driving, and continue to make changes and additions until solution was derived that could successfully drive the car around the track autonomously.

My first step was to do some research and find out how other people have accomplished similar tasks using Neural Networks. I found several examples, two of which were Inception Model from Google and the Nvidia Model from The Nvidia Cooperation. While I experiemented with both, I chose the Nvidia model due to its simplicity and smaller number of layers and parameters, as compared to the Inception Model.

I made a few additions to the Nvidia model and developed the rest of my pipeline for training the model. Then I captured a small set of training data to see if the model could make predictions and control the vehicle in the Simulator. I could tell that the model was attempting to do what I needed it to, so I continued with developing the model. I captured a lot more data and split the data into training and validation sets. I found that while the mean squared error of both sets was low, the car still had problems in certain turns when running the model in the simulator. 

To imporve this, I take additional data in those turns, and also found a few methods for augmenting the data to artificially increase the dataset and help the model generalize. I also found that cropping the image more than I orginially did, I got improved behavior.

Finally, I arrived at a solution that was able to drive the vehicle autonomously around the track without leaving the road.

####2. Final Model Architecture

The Model is defined in model.py, lines 75-118. It consists of 5 convolution layers with dropout layers included, and then 3 fully connected layers with dropout, and finally a linear layer with 1 output, the steering angle. The original architecture from Nvidia can be seen here:

![alt text][image1]

I added a cropping layer after the initial normalization layer, to remove some of the uneccessary background noise of the image. After some tuning, I decided to remove 50 pixels from the top, 20 pixes from the bottom, and 60 pixels from the sides of the image, resulting in a 90x200 image size. This image size is different than what was used in the Nvidia paper, which led to different convolution layer depths, and a larger number of neurons for the fully connected layers. I used Exponential Linear Units for activations after each layer and dropout layers were added
Here is a summary of the model architecture:

![alt text][image2]

####3. Creation of the Training Set & Training Process

To train the model, I first captured several laps of center lane driving. Here is an example image of center lane driving:

![alt text][image3]

I then recorded driving from the edges of the track back to the center. My hope was that this would help the model learn what to do if it strays from the middle of the track. Here are examples of the start of a recovery run:

![alt text][image4]
![alt text][image5]

I also decided to use the left and right camera images that are also captured by the simluator, and apply a small correction to the steering angle for these images. Here are the three camera images for a single point

![alt text][image6]

To augment the data sat, I also flipped images and angles thinking that this would provide the same number of left hand and right hand turn examples. For example, here is an image that has then been flipped:

![alt text][image7]

After the collection process, I had ~160,000 images and associated steering angles. I still had some difficulties when testing the model in the simulator (which I later found out was partially due to the limited capability of the computer I was using), so I found additional methods for augmenting the dataset. I used methods for adjusting image brightness and translating the images (model.py, Lines 46-57 and 62-72) mentioned by Vivek Yadav in his post: https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.jfs7vxyyn.

Here are examples of these augmentation techniques:

![alt text][image8]
![alt text][image9]

Then, as part of the model, a layer cropped the image before sending it to the convelution layers. Here is an example of the cropping:

![alt text][image10]

Finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I ran the model for 5 epochs, which showed the largest reductions in loss and proved to be sufficient for prediction in the simulator in autonomous mode. I used an adam optimizer so that manually training the learning rate wasn't necessary.


###Project Summary
The project proved very difficult, mainly because of the added time needed to use AWS to train the model, and my computer not being able to smoothly run the simulator and model predictions to autonomously control the vehicle. In the end, this was a very benefitial project, and I am please with my solution. Below is a video of the vehicle being controlled autonomously using my model:

![alt text][video1]



