# **Traffic Sign Recognition** 

## Writeup
This a summary of the work done in [Jupyter Notebook](Traffic_Sign_Classifier.html).
---

[//]: # (Image References)

[image1]: ./examples/histogram.png "Histogram"
[image2]: ./examples/dataset.png "Dataset"
[image3]: ./examples/preprocessing.png "Preprocessed Images"
[image4]: ./examples/new_signs.png "New Signs"
[image5]: ./examples/softmax.png "Softmax Probabilities"
[image6]: ./examples/no_entry.png "No Entry"
[image7]: ./examples/featuremaps1.png "Feature Maps 1"
[image8]: ./examples/featuremaps2.png "Feature Maps 2"

### Data Set Summary & Exploration

The dataset used for image classification is the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). It contains 34799 training images, 4410 validation images and 12630 testing examples of 32x32 RGB images of traffic signs. The dataset includes 43 classes of signs:

![Histogram][image1]

The signs included in this dataset can be read from this table:

![Dataset][image2]

As we can see, the lack of lighting and contrast on some of the images makes it almost impossible to identify the signs. This may also cause potential problems in our Neural Network because it indicates that the training set is very inhomogenous. We will see how to rectify this issue in the following.

### Design and Test a Model Architecture

#### 1. Preprocessing

As a first step, I decided to apply a very simple sharpener to the images by subtracting a Gaussian blurred version of the image:
```python
    blur = cv2.GaussianBlur(img, (3,3) ,0);
    image = cv2.addWeighted(img, 2.0, blur, -1.0, 0);
```
Next I applied a [Contrast Limited Adaptive Histogram Equalization](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization#Contrast_Limited_AHE) to the HLS-luminance channel to improve the contrast of the images. This has several advantages to a simple scaling of the whole image. Color values are more or less preserved (while increasing the brightness) and noise is not amplified in regions of similar color values.
```python
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    h, l, s = cv2.split(hls)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3,3))
    lh = clahe.apply(l)
    limg = cv2.merge((h,lh,s))
```
Finally, the image data is converted to grayscale and scaled to lie between 0.1 and 0.9.
The resulting images look like this:

![Preprocessed Images][image3]

The dataset is a lot more homogenous now and even the almost pitch black images (like the 23 Slippery road sign) are now clearly discernible.

In order to make the training more robust with respect to geometric transformations of the images I decided to apply a chain of several transformations to the dataset:

```python
def translate(img):
    del_x = np.random.randint(0,10)-5
    del_y = np.random.randint(0,10)-5
    M = np.float32([[1,0,del_x],[0,1,del_y]])
    dst = cv2.warpAffine(img,M,(32,32))
    return dst

def rotate(img):
    del_theta = np.random.randint(0,60)-30
    M = cv2.getRotationMatrix2D((16,16),del_theta,1)
    dst = cv2.warpAffine(img,M,(32,32))
    return dst

def shear(img):
    del1 = np.random.randint(0,8)-4
    del2 = np.random.randint(0,8)-4
    del3 = np.random.randint(0,8)-4
    del4 = np.random.randint(0,8)-4
    pts1 = np.float32([[0,0],[0,16],[16,16]])
    pts2 = np.float32([[del1,del2],[del3,16+del4],[16,16]])
    M = cv2.getAffineTransform(pts1,pts2)
    dst = cv2.warpAffine(img,M,(32,32))
    return dst
```

Translation should prepare the prediction for off-center images. Rotations and Shears should account for changes in perspective. Translation and Shear were applied twice, so the total size of the dataset was increased by 2^5.

#### 2. CNN Architecture

My final model more or less replicates the [LeNet](http://yann.lecun.com/exdb/lenet/) architecture described in the lecture.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Dropout               |                                               |
| Max pooling	      	| 1x1 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x16	|
| RELU          		|           									|
| Dropout               |                                               |
| Max pooling           | 1x1 stride,  outputs 5x5x16                   |
| Flatten               | outputs 400                                   |
| Fully Connected   	| outputs 120               					|
| RELU                  |                                               |
| Dropout               |                                               |
| Fully Connected       | outputs 84                                    |
| RELU                  |                                               |
| Dropout               |                                               |
| Fully Connected       | outputs 43                                    |
|						|												|
 


#### 3. Training Method

The model was initially trained for 20 epochs with a batch size of 400 at 0.7 keep rate for the dropout modules. Then it was trained for 10 more epochs at 0.9 and 0.95 keep-rate respectively for a total of 40 epochs. 

#### 4. Approach

My final model results were:
* training set accuracy of 91 %
* validation set accuracy of 95 %
* test set accuracy of 94 %

The basic approach was to use the LeNet architecture introduced in the course and then increase the training set to make the model more robust toward geometric transformation, and to find a suitable way of pre-processing the images. The low accuracy in the training set can be explained by the higher geometric complexity introduced by the shear and rotation transformations.

I also tried to improve on the architecture itself but was not successful in finding a significant improvement over the standard LeNet. For instance, instead of working with the greyscaled images one could take into account all color channel (in various color spaces, RGB, HLS etc.). This significantly increased the computation time, however it did not improve the accuracy. 
I also tried to introduce an inception module instoud of one of the convolutions. In fact, this increased the accuracy on the training set but had no effect on the validation and test set, which points to over-fitting.
 

### Test a Model on New Images

#### 1. Test Traffic Signs

![New Signs][image4]

These signs were randomly extracted from Google Street View. Two of these signs are not included in the database, namely sign 3, which is a priority over oncoming traffic sign and sign 7, which is a regular bus stop sign. They were chosen to maybe learn a bit more about which features the model uses for classification.
Sign 8 and 9 are versions of the same Zone 30 sign. 8 is zoomed out and off-center.

#### 2. Classification

Here are the results of the prediction:

| Image			         |     Prediction	        					 | 
|:----------------------:|:---------------------------------------------:| 
| Right-of-way      	 | Right-of-way 							 	 | 
| Priority road     	 | Priority road								 |
| Priority over oncoming | 60 km/h								    	 |
| No entry 	      		 | No entry  					 				 |
| 60 km/h                | 60 km/h                                       |
| Ahead Only 			 | Ahead Only     							     |
| Bus stop               | 60 km/h                                       |
| 30 km/h                | 30 km/h                                       |
| 30 km/h                | 30 km/h                                       |


The model was able to correctly guess all of the signs which were included in the data set. However, while testing other architectures, the second to last sign was most often classified as a 50 km/h. Interestingly, both signs which were not part of the dataset were classified as 60 km/h signs.

#### 3. Softmax probabilities

Here are the top 5 softmax probabilities for all traffic sign: 

![Softmax probabilities][image5]

As we can see, very unique signs like the "No entry" and "Priority road" signs were very easy to identify (softmax probabilities close to 1) while the model still had problems distinguishing different speed limits. On the second to last sign the "30 km/h" sign only scored about "73%". The zoomed in version was guessed correctly at 99.99% accuracy however.
The signs which were not present in the dataset scored very low on all guesses (31 % and lower).

### Visualizing the Neural Network

A detailed visualization of the first two activation layers can be found at the end of the [the Jupyter notebook](Traffic_Sign_Classifier.html). For instance, we can clearly see which layers were activated on the "No Entry sign":

![No Entry][image6]

First layer:
![First activation layer][image7]

Second layer:
![Second activation layer][image8]

