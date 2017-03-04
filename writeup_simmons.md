**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/cal_undistorted.png "Undistorted Calibration Image"
[image2]: ./output_images/undistorted_test.png "Undistorted Test Image"
[image3]: ./output_images/test5_var_grad_ksize15.jpg "Gradient Thresholding Exploration"
[image4]: ./output_images/test5_color_thresh.jpg "Color Thresholding Exploration"
[image5]: ./output_images/combined_thresh_example.png "Combined Threshold Example"
[image6]: ./output_images/straight_lines2warped.png "Warped Image Example"
[image7]: ./output_images/initial_search.png "Initial Lane Line Search"
[image8]: ./output_images/repeat_search.png "Repeat Lane Line Search"
[image9]: ./output_images/lane_visualization.jpg "Lane Visualization"
[video1]: ./output_images/test1.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

All of my initial pipeline development is located in the IPython notebook in ".code_files/pipeline_development.ipynb" The Camera matrix and distortion coefficients are found in cells 3 and 4 of this notebook. 

In the 2nd cell, all of the calibration images that were provided are read in, and the corners of the checkerboard are found using the cv2.findChessboardCorners() function. All of these corner points are saved as "Image Points." I generated the "Onject Points," which are the coordinates of the chessboard corners in the world, assuming no distortion or rotation. This means the coordinates are fixed in the (x, y) plane at z=0. The object points are then the same for each images, and are generated using numpy.mgrid() function. 

These sets of points were then used in the 4th cell of the notebook to find the camera calibration and distortion coefficients. This was done using the cv2.calibrateCamera() function. I then applied the distortion correction to any image I used for further analysis to find lane lines using the cv2.undistort() function. Here is an example of one of the calibration images after distortion correction:

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.

After finding the calibration of the camera, distortion correction is as simple as calling the cv2.undistort() function. Here is an example of a distortion corrected test image:

![alt text][image2]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

My final approach was to use a combination of color transforms, and gradients to obtain a binary image result that captured the lane line pixels. In order to reach the conclusion I did, I explored the various techniques seperately.

In the 5th cell of the note book, I explored gradient thresholding. I used the Sobel operator to approximate the gradient in the x and y directions, found the magnitude and direction of the gradient, and applied various thresholds to those calculations to obtain a binary image. I also looked at applying these gradient techniques to different channels of the RGB and HLS color spaces. I also experimented with applying a gaussian blur to the image before finding the gradient, which had little effect on the outcome. I adjusted the gradient kernal size as well. I generated image sets like the following example to study the effectiveness of each approach I took:

![alt text][image3]

Next, in the 6th cell, I explored color thresholding for lane line identification. I again explored thresholding various color channels of the RGB and HLS color spaces and used different threshold values to optimize the thresholding result. I explored image sets like the following to determine what channels and thresholds were the best for all of the test images:

![alt text][image4]

Finially, in the 7th cell, I explored different combinations of the gradient and color thresholding techniques. I arrived at the following solution, which I felt produced the best results:

combined_binary[((((R_x == 1) & (R_dir == 1)) & ((B_x == 1) & (B_dir == 1))) | ((S_x == 1) | ((S_mag == 1) & (S_dir == 1)))) | (c_S == 1)] = 1

where R_x and R_dir are the Sobel_x and gradient direction threshold results respectively, for the R channel of the RGB image. This is similarly done for the B channel, producing B_x and B_dir. S_x, S_mag, and S_dir are the Sobel_x, gradient magnitude, magnitude direction results using the S channel of the HLS color space of the image. And finally, c_S is the color threshold of the S channel.

Here is an example of an image with the final combined thresholding approach:

![alt text][image5]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The 8th cell in the notebook is where i developed the perpective transform for all of the images. I did this by taking a test image where the lane lines are known to be straight and parallel. I selected 4 points along the lines which formed my source plane and source points. I then picked 4 points to be destination points, knowing the lines formed by these points should be verticle. Here are the points I chose:

ysize = 720
point1 = 217,ysize
point2 = 580,ysize/2+100
point3 = 705,ysize/2+100
point4 = 1110,ysize

src = np.float32([[point1],[point2],[point3],[point4]])
dst = np.float32([[217,ysize],[217,0],[1110,0],[1110,ysize]])

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 580, 460      | 217, 0        | 
| 217, 720      | 217, 720      |
| 1110, 720     | 1110, 720      |
| 705, 460      | 1110, 0        |

I used these points to generate the transformation matrix using the cv2. getPerspectiveTransform() function. Using this tranformation matrix, I warped the image using cv2.warpPerspective() function. I verified that my perspective transform was working as expected by drawing the `src` and `dst` points back on to the warped image to make sure they followed the lane lines:

![alt text][image6]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I use two seperate approaches for identifing lane-line pixels, both of which were descriped in the class material. The first appraoch, which is in the 9th  cell of the notebook, or in the 2nd cell under the intial_lane_line_search() function, is for the first frame of the video, or if no valid lane line was found on the previous image frame. This approach uses a histogram of the bottom half of the warped binary image to find the base of the lane line, and then applys a moving window as it searches up the image to identify the rest of the lane line pixels. Then, a polinomial best fit line is found using the numpy.polyfit() function. Here is an example of this approach:

![alt text][image7]

Once a fit line is found that accurately represents the lane line, I use the repeat_lane_line_search() function (cell 2) to only search for pixels within a margin from that fit line. Here is that approach:

![alt text][image8]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The function for calculating the radius of curvature is in the 2nd cell of the notebook called find_curvature(). First, the polinomial fits had to be converted to the world space. Then the equation provided in the course material is used to calculate the curvature a the lowest point in the image (closest to the vehicle).

The function for locating the position of the vehicle is called find_lateral_offset(). I do this by finding the x value of each polinomial fit at the base of the image, and then proceed as follows:
    #Center of Lane
    center = np.mean([left_fitx,right_fitx])
    # Offset in Pixels
    pixel_offset = img_width/2 - center
    # Offset in Meters
    offset = pixel_offset * xm_per_pix
	
We assume that the vehicle center is at the center of the image. 

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step using function visualize_lane() in cell 2 of the notebook. Here is an example

![alt text][image9]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Once I developed the code I needed, created classes and functions from that code and transfered everything I needed to process the video in "the main.ipynb" notebook. I continued to make adjustments to get good performance on the project video. Overall, I think the performance of the lane finding pipeline is farily good. I had problems with finding accurate polinomial fit representations of the white dashed lane line. Even when I identifed all of the lane line pixed, the fit generated would not accuartely represent the true lane line. 

My attempts to apply sanity checks to the lines found proved to cause more problems than improve performance. I ended up not using them, and only filtered the lines from image to image using:

Lane_line = alpha*current_fit + (1-alpha)*previous_fit

The pipeline is likely to fail if the lane line disappears completely or if there are other lines on the road that are close to vertical. TO make the pipeline more robust, I could get the sanity checks working properly to remove erronous detections.


