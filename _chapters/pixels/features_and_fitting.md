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
	- [Determining the Value of "$k$"](determining-the-value-of-"k")
- [Second Big Topic](#topic2)

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

<a name='Deterimining the Value of $k$'></a>
#### Determining the Value of "$k$"

Let $w$ represent the fraction of inliers (points on the line), $n$ represent the points needed to define a model estimate (two in the case of line fitting), and $k$ represent the number of samples chosen. 

The probability that at least one sample will be the true line can be derived through a series of steps: 
1. The probability that a randomly chosen point is on the line is $w$ or the fraction of true inliers. 
2. Therefore, the fraction that a sample of $n$ points are all part of the true line is $w^n$, or the probability that every point in the sample is on the line. 
3. Which means that the probability that at least one point in the sample is not part of the true line is the complement or $1 - w^n$. 
4. We can then compute the probability that $k$ samples will fail by raising the probability of a single sample failing (at least one point in the sample not being on the line) to the $k$ power. This results in the formula $(1-w^n)^k$ which is the probability that every single sample fails to model the true line. 
5. The complement of this is the probability that at least one of the $k$ samples models the true line or $1 - (1 - w^n)^k$.

Using this result, we can calculate the value of $k$ that will give a $0.99$ probability of at least one of the $k$ samples successfully modeling the true line. To do this, plug in $w$ and $k$ (usually given or can be inferred), then set the probability of at least one success equal to $0.99$ and solve for $k$.
