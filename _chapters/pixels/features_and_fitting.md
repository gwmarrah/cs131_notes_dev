---
title: Features and Fitting
keywords: RANSAC, Local Invariant Features, Harris Corner Detector
order: 0
authors: Mateo Echeverri (mateoae), Travis Grafton (tjgraft), Jung-Won Ha (jwha23), Stephanie Hu (stephhu1), George Walters-Marrah (gwmarrah), Derian Williams (derianw)
---


- [Conceptual Understanding of RANSAC](#conceptual-understanding-of-ransac)
	- [RANSAC Definition](#ransac-definition)
	- [Challenges Overcome by RANSAC](#challenges-overcome-by-ransac)
	- [RANSAC Algorithm for Line Fitting](#ransac-algorithm-for-line-fitting)
	- [RANSAC Line Fitting Workflow](#ransac-line-fitting-workflow)
	- [Determining the Value of "k"](determining-the-value-of-"k")
- [Optimizing RANSAC](#optimizing-ransac)
- [RANSAC Coding Demo](#ransac-coding-demo) 
- [Conceptual Understanding of Local Invariant Features](#conceptual-understanding-of-local-invariant-features)
	- [Motivation](#motivation)
	- [General Workflow](#general-workflow)
	- [Common Requirements](#common-requirements) 
- [Conceptual Understanding of the Harris Corner Detector](#conceptual-understanding-of-the-harris-corner-detector)
	- [Goal of Harris Corner Detection](#goal-of-harris-corner-detection)
	- [Basics of Corner Detection](#basics-of-corner-detection)
	- [Formulation for Harris Corner Detector](#formulation-for-harris-corner-detector)
- [Harris Corner Detector Matrix](#harris-corner-detector-matrix)
- [Harris Corner Detector Example](#harris-corner-detector-example)

[//]: # (This is how you can make a comment that won't appear in the web page! It might be visible on some machines/browsers so use this only for development.)

[//]: # (Notice in the table of contents that [First Big Topic] matches #first-big-topic, except for all lowercase and spaces are replaced with dashes. This is important so that the table of contents links properly to the sections)

[//]: # (Leave this line here, but you can replace the name field with anything! It's used in the HTML structure of the page but isn't visible to users)
<a name='Topic 1'></a>

# Conceptual Understanding of RANSAC


<a name='RANSAC Definition'></a>
#### RANSAC Definition
RANdom SAmple Consensus is a model fitting method for line detection. It uses an iterative process that takes a random sample of features, fits a line model, and looks for features that are “inliers” for that particular #line model. 


<a name='Challenges Overcome by RANSAC'></a>
#### Challenges Overcome by RANSAC

There are several challenges associated with line fitting:

1. Since there are multiple line models (or instances) in an image it is difficult to determine which points belong to each line model (if any). Extra edge points can lead to clutter that may confuse the algorithm and lead to spurious line detection.
2. A subset of the whole line may be detected by the algorithm, which leads to missing parts of the line. These line segments must be bridged together or extended to form the entire line, even while there are missing edge points.
3. Noisy edge points may not be exactly co-linear even while belonging to the same line. The underlying parameters of the line need to be detected even while using edge points with imprecise coordinates.
4. It is not feasible to check all possible combinations of edge points. This naive approach leads to a performance of O(N^2) which is not scalable to large tasks of line fitting.

RANSAC addresses all of these difficulties by using random sampling and voting as a fitting technique. RANSAC cycles through the features and lets them vote for particular model parameters that they are compatible with. Even though outliers would still vote, their votes are usually not correlated with features that are part of lines, and the resulting model will not have much support.


<a name='RANSAC Algorithm for Line Fitting'></a>
#### RANSAC Algorithm for Line Fitting

The RANSAC loop is repeated for $k$ iterations:
1. Take a random sample of edge points to make a seed group
2. Estimate a line model by computing the parameters of a line from the seed group
3. Find features within a threshold that support this model. Features that support the model are referred to as inliers
4. If there is a sufficient amount of support for the model, compute the least-squares estimate of the model with the inclusion of the inliers.

The model with the most support (largest amount of inliers) is kept as the computed line instance.


<a name='RANSAC Line Fitting Workflow'></a>
#### RANSAC Line Fitting Workflow

Only two points are needed to estimate a line. So, the first task for RANSAC line fitting is randomly selecting two points in the feature space to make an initial (potentially poor) line model. 

<div class="fig figcenter fighighlight">
  <img src="{{ site.baseurl }}/assets/images/initial-line-fit.png">
  <div class="figcaption"> The first task of the RANSAC loop is to sample points from the features of the image (denoted by the blue points). After sampling, parameters for the line model are computed using the seed group.</div>
</div>

After fitting the initial line, inliers (within a prespecified threshold) are detected to collect votes to determine how much support there is for the model. This loop is repeated $k$ times and the line with the most support is kept after every iteration.

---
<div class="fig figcenter fighighlight">
  <img src="{{ site.baseurl }}/assets/images/detect-inliers.png">
  <div class="figcaption"> Once the line model is computed, inliers are detected and the support for the line model is quantified. The line model computed in this particular iteration was quite poor, but may improve in following iterations if a different sample is used.</div>
</div>

---

<a name='Deterimining the Value of k'></a>
#### Determining the Value of "k"

Let $w$ represent the fraction of inliers (points on the line), $n$ represent the points needed to define a model estimate (two in the case of line fitting), and $k$ represent the number of samples chosen. 

The probability that at least one sample will be the true line can be derived through a series of steps: 
1. The probability that a randomly chosen point is on the line is $w$ or the fraction of true inliers. 
2. Therefore, the fraction that a sample of $n$ points are all part of the true line is $w^n$, or the probability that every point in the sample is on the line. 
3. Which means that the probability that at least one point in the sample is not part of the true line is the complement or $1 - w^n$. 
4. We can then compute the probability that $k$ samples will fail by raising the probability of a single sample failing (at least one point in the sample not being on the line) to the $k$ power. This results in the formula $(1-w^n)^k$ which is the probability that every single sample fails to model the true line. 
5. The complement of this is the probability that at least one of the $k$ samples models the true line or $1 - (1 - w^n)^k$.

Using this result, we can calculate the value of $k$ that will give a $0.99$ probability of at least one of the $k$ samples successfully modeling the true line. To do this, plug in $w$ and $k$ (usually given or can be inferred), then set the probability of at least one success equal to $0.99$ and solve for $k$.

## Optimizing RANSAC

Above, we described the general 4 steps that the RANSAC algorithm continuously repeats for a finite number of iterations until it finds a model that best fits the target line(s). Subsequently, at the end of each iteration, there is an additional step that we can take to further improve the model's line-fitting. 

In essence, from a minimal sample of *n* points, RANSAC computes its best estimate and uses that to divide all the data points into "inliners" and outliers. 

---
<div align="center">
  <img src=https://i.imgur.com/xYvm5DS.png width="400" align="center"/>
</div>

---

Before we try to compute another estimate from a new random set of points, we can further improve this current estimate by utilizing the classified inliner data points.

From the current set of classified inliner points, we utilize least squares regression to find a new estimate for the line of best fit. 

---

<div align="center">
  <img src=https://i.imgur.com/D4lz30J.png width="400" align="center"/>
</div>

<sup>Perform least squares regression with the classified inliner points to find a new best fit line. The red line signifies the new model computed from the inline points. The red lines vertically extending from the green data points to the red regression line indicate the least squares regression minimal distance.</sup>

---

As witnessed, this can lead us to further improving our model estimate. Consequently, by recalculating the line of best fit and its corresponding thresholds, this also changes which datapoints are classified as inliners and outliers. So a good course of action is to alternate between model fitting and re-classification of inliner/outlier data points. 

---

<div align="center">
  <img src=https://i.imgur.com/gg21BX8.png
 width="400" align="center"/>
</div>

---
## RANSAC Coding Demo
We will utilize RANSAC to detect the painted lines in this image. 

<div align="center">
  <img src=https://i.imgur.com/dopvJp2.jpg
 width="400" align="center"/>
</div>

We want to first run Canny edge detection to get a list of points. Then we will first use Least Squares Regression as our line estimation algorithm. 

```
img = io.imread("road.jpg", as_gray=True)
edge = feature.canny(img, sigma=5)
y, x = np.nonzero(edge)
reg = LinearRegression().fit(x.reshape(-1,1), y)
print(f"Line equation: y = {reg.coef_[0]:.2f}*x + {reg.intercept_:.2f}")

x_line = np.linspace(0,img.shape[1]-1).reshape(-1,1)
y_line = reg.predict(x_line)

plt.figure(figsize=(20,10))
plt.subplot(1,3,1); plt.imshow(img, cmap="gray"); plt.title("input image", fontsize=20);
plt.subplot(1,3,2); plt.imshow(edge, cmap="gray"); plt.title("edge map", fontsize=20);
plt.subplot(1,3,3); plt.imshow(edge, cmap="gray"); plt.title("least squares line", fontsize=20);
plt.plot(x_line, y_line, 'r')
plt.tight_layout()
```
As evidenced from the edge map, there are a lot of noisy/outlier points in the image. Least squares regression is unable to distinguish from inlier and outlier points, so the ultimate line of best fit it detects is highly inaccurate. 
![](https://i.imgur.com/qu1Ki4Z.png)

Let's replace our line estimator model to be RANSAC. 
```
ransac = RANSACRegressor(base_estimator=LinearRegression())
ransac.fit(x.reshape(-1,1), y)
```
![](https://i.imgur.com/cZgGl5n.png)
The resulting line fit is better than the least squares regression estimate, but it still isn't to the standard that we are expecting. The line estimate is sitting between both street lines rather than exactly predicting just one.  

To understand what went wrong, let's run the code below to see which data points RANSAC classified as inliers. 
```
x_inliers = x[np.nonzero(ransac.inlier_mask_)]
y_inliers = y[np.nonzero(ransac.inlier_mask_)]
print(f"Found {x_inliers.size} inliers.")
plt.figure(figsize=(20,10))
plt.subplot(1,3,1); plt.imshow(edge, cmap="gray"); plt.title("edge map", fontsize=20);
plt.subplot(1,3,2); plt.plot(x_line, y_line, 'r--'); plt.imshow(edge, cmap="gray"); plt.title("line with ransac", fontsize=20)
plt.subplot(1,3,3); plt.plot(x_inliers, y_inliers, 'g.'); plt.imshow(edge, cmap="gray"); plt.title("inliers", fontsize=20)
plt.tight_layout()
```
![](https://i.imgur.com/JVfsaW5.png)

We can see that RANSAC classified both street lines as being inliers, hence why our best fit line was fitted between both lines. By default, `RANSACRegressor()` sets the maximum residual for a data sample to be classified as an inlier to be the MAD (median absolute deviation) of the target values y. However, through the parameter `residual_threshold`, we can specify and refine the threshold. Let's specify the threshold to be 5 pixels to refine the amount of surrounding data points we classify as inliers. 

```
ransac = RANSACRegressor(base_estimator=LinearRegression(), residual_threshold=5)                
```
![](https://i.imgur.com/lCYJJIF.png)


In refining the threshold, we have optimized RANSAC to correctly fit the model to a single line in the image.  

![](https://i.imgur.com/0DHPHod.png)
In viewing the classified inliers, we can see that refining the threshold allowed us to have a more compact list of inliers. 

##Conceptual Understanding of Local Invariant Features

###Motivation
Cross-correlation has proven useful when pinpointing particular patterns within an image, provided that the pixel configuration between the pattern and the query image remains constant. However, global templates used for image matching with cross-correlation lack robustness with regards to scale changes, rotations, viewpoint changes, lighting changes, occlusions, and other similar variations. In turn, rather than concentrating on all pixels of an image, analyzing local invariant features—small image patches that are common across multiple pictures of the same scene—can lead to more accurate image matching and subsequent classification.

<div align="center">
  <img src=https://i.imgur.com/QRDuaER.png
 width="400" align="center"/>
  <div> Example of an image matching task where a global template may not perform well. Source: Lecture 5 slides. </div>
</div>

###General Workflow
To conduct image matching through local invariant features, we should generally follow the steps listed:

1) **Find a set of distinctive keypoints.** Such keypoints should be representative of the object to be identified.

2) **Define a region around each keypoint.** Rather than concentrating on a single pixel where the keypoint is located, using an image patch around the pixel allows us to additionally consider factors such as texture and shape in the neighborhood of the pixel.

3) **Extract and normalize the region content.** Normalization can standardize image patches that may otherwise vary in aspects such as viewpoint, size, and rotation.

4) **Compute a local descriptor from the normalized region.** The content of each standardized image patch can be encapsulated through a computed vector. 

5) **Match local descriptors.** After measuring the distance between the computed vectors corresponding to two separate images, we can declare the images to be similar if the aforementioned distance is below a defined threshold. 

<div align="center">
  <img src=https://i.imgur.com/p1ypDlK.png
 width="400" align="center"/>
  <div> Depiction of the identification and comparison of descriptors for local invariant features. Source: Lecture 5 slides. </div>
</div>

###Common Requirements
Above all, we must address two central problems: detecting the same point independently in separate images of the same scene, which requires a repeatable detector; and subsequently matching each point in one image correctly to the corresponding point in a second image after the points have been localized in both images, which requires a reliable and distinctive descriptor. Local features should then satisfy the requirements listed:

1) **Region extraction needs to be repeatable and accurate.** In order to be useful, the keypoint detector should be somewhat invariant to geometric transformations such as scale changes, translations, and rotations; robust or covariant to out-of-plane transformations; invariant to photometric transformations so that different lighting conditions do not affect descriptors corresponding to image keypoints; and robust to other variations such as noise, blur, and quantization.

<div align="center">
  <img src=https://i.imgur.com/GYMsHSN.png
 width="400" align="center"/>
  <div> Example of feature invariance with geometric transformations. Source: Lecture 5 slides. </div>
</div>

2) **Locality.** Too-large features lead to problems similar to those caused by global template matching. Features must then be local and thus robust to occlusion and clutter.

3) **Quantity.** Detectors should produce a sufficient number of regions so that they are able to cover the particular object of interest.

4) **Distinctiveness.** Keypoint regions should contain a distinctive structure. Regions should not capture overly homogeneous areas with notably flat color, since it will be difficult to match keypoints within such regions due to their homogeneous quality.

5) **Efficiency.** The algorithms used must be efficient so that we are capable of achieving close to real-time performance.

## Conceptual Understanding of the Harris Corner Detector
###Goal of Harris Corner Detection
Harris Corner Detection is a method of finding corners in an image using the concept of keypoint localization. Corners serve as excellent key points because they are repeatable in that they can be seen from multiple viewpoints, distinctive from their neighbors, and, in the region around a corner, the image gradient has two or more dominant orientations.
###Basics of Corner Detection
In order to see corners as keypoints that can be detected, there need to be certain design criteria. First, corner points should be easily recognizable by looking through a small window – they must be local. Second, moving this window in any direction should give a large change in intensity - it should possess good localization. 

<div align="center">
  <img src=https://i.imgur.com/Rh1HrCv.png
 width="600" align="center"/>
  <div> Illustration of shifting a window over different region types. Source: Lecture 5 slides. </div>
</div>

For example, moving this window anywhere in a flat region or along an edge yields no change in intensity. Moving it in any direction on a corner, however, yields significant change in intensity in all directions.

One way to measure intensity change in images is with gradients. In this case, only the magnitude of intensity is important, so the gradient squared is used in each of the horizontal and vertical directions, $I_x^2$ and $I_y^2$. Consider the following cases for placing a window on different region types within an image and summing over the square of the gradients:


*   Corner -  large $\sum I_x^2$,  large $\sum I_y^2$
*   Edge in x-direction - small $\sum I_x^2$, large $\sum I_y^2$
*   Flat - small $\sum I_x^2$, small $\sum I_y^2$


This may look promising, but it cannot be directly applied to corners that are rotated because, in this case, the strength of the gradient magnitudes cannot be predicted. Therefore this measurement of change must be further formalized to fit this more general case.
 
###Formulation for Harris Corner Detector

To reiterate, the goal for the Harris Corner Detector is to localize windows that result in a large change in intensity when shifted in any direction. To develop a formulation of this goal, first consider the single, center pixel of the window $(x, y)$. When shifted by an amount $(u, v)$, taking the difference between the value of the intensity at the original location, $I(x, y)$ and the shifted location, $I(x + u, y + v)$, will yield the measurement of change: $(I(x + u, y + v) – I(x, y))$. This calculation only results in the change for a single pixel, so it is accumulated around the shifted point within the window to yield the change in intensity for the entire window:
 $$E(u,v) = \sum_{x, y} w(x, y)[I(x + u, y + v) – I(x,y)]^2$$
 Where:
  *    $[I(x + u, y + v) – I(x,y)]^2$ is the intensity change for a single pixel
  *    $w(x, y)$ is the window function which helps to weigh the importance of each pixel within the window
 

This measure of change can then be approximated by using a Taylor Expansion to yield a new equation:
 $$E(u, v) \approx \left[\begin{array}{cc} u & v \end{array} \right] M \left[\begin{array}{cc} u \\ v \end{array} \right]$$
where $M$ is a 2x2 matrix computed from summing image derivatives over the window and weighing them:
  $$M = \sum_{x, y} w(x, y) \left[\begin{array}{cc}I_x^2 & I_xI_y \\ I_xI_y & I_y^2\end{array} \right]$$
Note that the matrix of image derivatives consists of both the square of the gradient in the $x$ and $y$ directions, as well as the product of the two.

## Harris Corner Detector Matrix

## Harris Corner Detector Example

import numpy as np
from skimage import io
from skimage.feature import corner_harris, corner_peaks
from matplotlib import pyplot as plt

img = io.imread("mond.jpg", as_gray=True)
'''
The sigma that is passed to the  corner_harris method defines the 
Gaussian used for the window function. A bigger sigma means a bigger
neighborhood for deciding if there is a corner. 
'''
theta_a = corner_harris(img, sigma=2)
plt.figure(figsize=(20,10))
plt.subplot(1,2,1); plt.imshow(img, cmap="gray")
plt.subplot(1,2,2); plt.imshow(theta_a, cmap="jet")
plt.tight_layout()
plt.show()

Clearly, the corner_harris detector didn't catch the corners of the darkest squares in the image. That is because those corners don't have enough constrast (the picel intensities are too similar so the gradients aren't large enough. This can be solved by modifying the image contrast using the adjust_gamma method in scikit.

from skimage import data, exposure, img_as_float
gamma_corrected = exposure.adjust_gamma(img, 0.15)

theta_b = corner_harris(gamma_corrected, sigma=2)
plt.figure(figsize=(20,10))
plt.subplot(1,2,1); plt.imshow(gamma_corrected, cmap="gray")
plt.subplot(1,2,2); plt.imshow(theta_b, cmap="jet")
plt.tight_layout()
plt.show()

However, this doesn't give a list of points. At the end of the day, what we want is to know the coordinate location of the corners. We want the coordinates of the peak values. For that, we find the maximum values of the function.

'''
@min_distance is how far we want the corners to be from each other. Its unit is pixels.
It is good for checking that htere aren't other nearby max values.
@threshold_rel is how strong do we want the corner function to be. E.g.,  "only corners
stronger that certain value are kept."
'''
coords = corner_peaks(theta_b, min_distance=10, threshold_rel=0.02)                   
                      
plt.figure(figsize=(30,15))
plt.subplot(1,3,1); plt.imshow(gamma_corrected, cmap="gray"); plt.title("input", fontsize=30)
plt.subplot(1,3,2); plt.imshow(theta_b, cmap="jet"); plt.title("corner response function", fontsize=30)
plt.subplot(1,3,3); plt.plot(coords[:,1], coords[:,0],'r.'); plt.imshow(theta_a, cmap="gray"); plt.title("peak locations", fontsize=30)
