# Final Project Report

CIS 472 Machine Learning

Submitted to: Ph.D. Thien Huu Nguyen

Author: Warren Liu

## Introduction ##

In my final project, I built the image classifier based on the different models, including convolutional neural networks (CNN) and k-nearest neighbors (KNN). I built two models and evaluated their performances.

## Background ##

The image classification is known as classify pictures into different categories (labels), the most popular way to achieve a image classifier is building a convolutional neural network (CNN). Generally, in my understanding, classify by CNN is a way that we extract the features from an image again and again, until we get enough information to determine the label of a picture. A CNN has convolutional layers, pooling layers, and a fully connected layer. The convolutional layer is where we extract features, the pooling layer is built for extracting the "most useful" features again, also is built for reducing the calculations, and the fully connected layer is for outputting the features extracted to categories (labels) we desired. 

In addition, I built a  k-nearest neighbors (KNN) image classifier to compare the performance with CNN model. KNN is an algorithm that use k most-likely pictures to determine the categories of the test image.

### Database ##

The database I used is the fashion MNIST Database. It is a database about clothes in 10 categories. All pictures are in black and white, fit in a 28x28 pixel box. It contains a training set of 60,000 examples, and a test set of 10,000 examples. The datasets are cleaned and pre-processed, which is good and easy to use to train models.

According to the MNIST website: "The original black and white (bilevel) images  from NIST were size normalized to fit in a 20x20 pixel box while preserving their  aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing  technique used by the normalization algorithm. the images were centered in a 28x28  image by computing the center of mass of the pixels, and translating the image so as to  position this point at the center of the 28x28 field".

## Implementation ##

### k-nearest neighbors (KNN) ###

#### Distance ####

One of the most important things in KNN is the distance algorithm, which is used for calculating the distance from the test image to training images. I used two formulas in my implementation. 


$$
Distance = (\sum(x_1-y_1)^2 + (x_2 - y_2)^2 + ... + (x_n - y_n)^2)^\frac{1}{2}
$$


Where X is each pixel index of the test image, Y is each pixel index of the training image.

Euclidean Distance: use the original value in each index (But normally divide each value by 255).

L0 Distance: assign each index to 0 if the value is 0, and assign each index to 1 if the value is greater than 0.

####  Algorithm ####

```python
for each_test_image in all_test_image:
    for each_train_image in all_train_image:
        Caculte the distance between this test image to this train image
        Append to distance list
    Get the most popular label among k shortet distance images
    Evaluate this label with test_image_label
```

### convolutional neural network (CNN) ###

I tried to build different CNN models to train the same dataset.

I selected the Adam optimizer for all models, and the loos function is SparseCategoricalCrossentropy.

#### Models ###

```python
class model_1():
    conv1 = Conv2D(filters, kernel_size, activition='relu')
    flatten = Flatten()
    d1 = Dense(units, activition='relu')
    d2 = Dense(10, activition='softmax')
# Applied gradient
```

```python
class model_2():
    conv1 = Conv2D(filter_size, kernel_size, activition='relu')
    maxPool1 = MaxPooling2D(pool_size)
    conv2 = Conv2D(filter_size, kernel_size, activition='relu')
    maxPool2 = MaxPooling2D(pool_size)
    conv3 = Conv2D(filter_size, kernel_size, activition='relu')
    flatten = Flatten()
    d1 = Dense(units, activition='relu')
    d2 = Dense(10)
```

```python
class model_3():
    conv1 = Conv2D(filter_size, kernel_size, activition='relu')
    maxPool1 = MaxPooling2D(pool_size, padding='same')
    conv2 = Conv2D(filter_size, kernel_size, activition='relu')
    maxPool2 = MaxPooling2D(pool_size, padding='same')
    conv3 = Conv2D(filter_size, kernel_size, activition='relu')
    flatten = Flatten()
    d1 = Dense(units, activition='relu')
    d2 = Dense(10)
```

```python
class model_4():
    conv1 = Conv2D(filter_size, kernel_size, activition='relu')
    maxPool1 = MaxPooling2D(pool_size, padding='same')
    conv2 = Conv2D(filter_size, kernel_size, activition='relu')
    maxPool2 = MaxPooling2D(pool_size, padding='same')
    conv3 = Conv2D(filter_size, kernel_size, activition='relu')
    flatten = Flatten()
    d1 = Dense(units, activition='relu')
    d2 = Dense(10)
    
# Applied gradient
```



## Experiment ##

### KNN

I manually tested different K and different distance algorithms by setting iters for distance_function and k.

```python
for each_distance_fn in distance_fn:
	for each_k in all_k:
        train_with(each_distance, each k)
```

Because it was hard to send my code to GPU, I ran this program on the CPU. It takes average 2s for training 1 test image under Euclidean distance, and 3s for training 1 test image under L0 distance, but there are total 10,000 images in the test set. Therefore, I reduced the test size to 100. The error_time is the count of total wrong predictions.

 Here are the outcomes I got:

| Distance formula |  K   | Error_time | Accuracy | Time    |
| :--------------: | :--: | ---------- | -------- | ------- |
|    Euclidean     |  5   | 13         | 87.0%    | 202.30s |
|    Euclidean     |  7   | 16         | 84.0%    | 213.27s |
|    Euclidean     |  9   | 15         | 85.0%    | 214.63s |
|    Euclidean     |  11  | 17         | 83.0%    | 215.95s |
|        L0        |  5   | 16         | 84.0%    | 315.00s |
|        L0        |  7   | 13         | 87.0%    | 319.48s |
|        L0        |  9   | 15         | 85.0%    | 323.12s |
|        L0        |  11  | 15         | 85.0%    | 300.08s |

### CNN ###

I first manually tested these models by setting them with a most popular value for each hyperparameter. I found that models 2-4 perform about 2%-3% better than model_1.

Then, I used keras-tuner to automatically tuned models 2-4. The hyperparameters I tuned are: Conv2D layer's filter size, dense unit size, with [min_value=32, max_value=128, step=32], and the padding way.

Here is the outcome I got:

```
Trial 30 Complete [00h 01m 14s]
val_accuracy: 0.9229166507720947

Best val_accuracy So Far: 0.9229166507720947
Total elapsed time: 00h 15m 31s

    conv1 size: 96

    conv2 size: 96

    conv3 size: 64

    dense size: 64
```

## Conclusion ##

Based on the outcomes I got, it is easily to conclude that CNN works much better than KNN on image classification. In my understanding, KNN is simply calculating the images' similarity, it leads to a problem that the prediction will be different for two same objects that each object is in the different locations of each image. For instance, the same dog, but one is at the right up corner of a picture, and another is at the left bottom of a picture, the "distance" between these two pictures will be large. However, the CNN is extracting "features" from each image, it ignores the location of object, also the angle, size, etc. I further more trained CNN models on the different MNIST datasets, like the hand-written numbers, and color images. I found that it performed better on the black and white images then it performed on the color images. I believe that it is because the black and white images size is [28, 28], and the color images size is [28, 28, 3], which has a one more color channels dimension. In addition, the hand-written images got the most accurate prediction, which is 98%. I consider that the classification does better on the simply images.

#### Discussion

After doing this project, I consider that the biggest difference between human eyes' recognition and A.I. image classification in nowadays is that human eyes can detect the fourth dimension,  the depth of field. If the camera is able to write the depth of field to a picture, we can firstly extract each object from the image, and then classify each object, which I believe is more simply, faster, and more accurate to classify image.