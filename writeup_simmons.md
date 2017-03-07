**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_notcar.png
[image2]: ./output_images/Hogs.jpg
[image3]: ./output_images/testboxes.jpg
[image4]: ./output_images/test_perf2.jpg

[video1]: ./project_video_detections.mp4
[video2]: ./project_video_test.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the 4th cell of the "pipeline_developement.ipynb"

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

Here is an example using the first channel of the `YCrCb` color space and HOG parameters of `orientations=15`, `pixels_per_cell=(8, 8)` and `cells_per_block=(1, 1)`:

![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters for Hog, spatial, and color histogram feature extraction, a total of 90 combinations. I extracted features from all of the training examples for each combination, and trained a linear SVC using sklearn.svm.LinearSVC(). I then tested the accuaracy of each combination on 20% of the training images that I kept for validation. With an accuracy of 0.9941, the best performance was achieved by the following combination of parameters:
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 15  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 1 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 64    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

This set of parameters led to a feature vector length of 3840

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using sklearn.svm.LinearSVC() and the parameters found producing the best performance. This was done in the 4th cell of the "pipeline_developement.ipynb". After seperating the training data into a training and validation set, and shuffleing the data, LinearSVC.fit() was used to train the classifier.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In the 5th cell of the same notebook, I experimented with using different window scales to search the image. I used various sizes, and searched different areas of the image based on the size of the window I was using. All of the windows started searching from pixel 400, slightly below the center of the image. The smaller windows only searched the area closer to the center of the image (farther away from the vehicle), and larger windows searched farther down the image, where vehicles would appear larger in the image.

Because I used a fairly aggresive thresholding technique, I chose to overlap the windows in both the x and y directions by 75%, except for the smallest window size of (32x32) which was overlapped by only 50%. I also used a variety of window sizes that overlapped each other as well. This was to hopefully get multiple detections around each vehicle. Here are the parameters I used for producing the windows for searching, and an image depicting these windows on a test image:
window_sizes = [32,64,128,192,256]
y_stops = [496,528,592,640,656]
xy_overlaps = [.5,.75,.75,.75,.75]

![alt text][image3]

This led to a large number of windows, and a long run time for my pipeline, but very good performance.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here is an example test image with positive detections drawn, a heatmap and thresholded heatmap of those detections, and a final image with vehicle bounding boxes drawn.
 
![alt text][image4]

I optimized the performance of my classifier by further adjusting the search window parameters, and the thresholding values used to remove false detections, in order to achieve acceptable perfromance on all of the test images.
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_detections.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In the second cell of the "main.ipynb" there is a function called "process_images" which is my main pipeline for analyzing video images. Within this pipeline, I have multiple layers of filtering. First I take the current image detections, create a heatmap based on those detections, and then apply a threshold to that heatmap, in order to filter out single false positive detections. scipy.ndimage.measurements.label() is then used to group the thresholded detections into seperate "vehicles", and then bounding boxes are fit to these labeled pixels. These single frame vehicle bounding boxes are then saved, in order to be used on the next frame of the video in the following manner.

Using the vehicle bounding boxes from the past 5 video frames, a heatmap is generated. This heatmap is then added to the heatmap generated from the current frame detection windows. A threshold is then applied to this combined heatmap, based on the number of frames used. This ensures that a detection in that area of the image must be found in successive video frames before it will be counted as a valid vehicle detection, and be tracked. 

An example video I used to assess the performance of my filter techniques can be seen here: This video shows the individual frame detections, the combined heatmap, the thresholded heatmap, and finally the vehicle bounding boxes plotted back onto the original image.

Here's a [link to video](./project_video_test.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One problem I faced with my implentation was with computational time. Because I chose to search a variety in sizes of windows with large overlaps, I had to extract large numbers of hog features for each image. This proved to be very time consuming. I attempted to implement the hog sub-sampling method described in the class material, however I ran into problems with the feature length being incorrect. This is one improvment I would need to make in order to take my pipeline further. 

While I was able to successfully filter out false positive detections, I did have problems with the robustness of the true vehicle detections, and specifically the size of the bounding box. The boxes frequently jumped in size from frame to frame and also would sometimes split into 2 bounding boxes. This is one area of improvement I would like to make in the future. 

The pipeline is likely to fail under low light or with cars that are a similar color to the background. Also, because of my filtering, if the relative velocity between the host and target vehicles is large, its possible a confirmed detection will not occur. 

In order to make the pipeline more robust, continued tuning of the filtering techniques and the search windows used could be done. 

