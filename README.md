# CNN Model with Fashion MNIST Implementation

# Summary

This notebook trains a Convolutional Neural Network model to classify images of clothing, like sneakers and shirts. 
This notebook uses tf.keras, a high-level API to build and train models in TensorFlow.


# Problem

In Deep Learning, we have a lot of features. It is a very challenging task, especially if you don’t know where to start! Having a high number of variables is both a boon and a curse. 
It’s great that we have loads of data for analysis, but it is challenging because of the size.
It’s not workable to analyze every variable at a microscopic level. It might take us days or months to perform any meaningful analysis and we’ll lose a ton of time and money for our business! Not to mention the amount of computational power this will take.
We need a better way to deal with high-dimensional data so we can quickly extract patterns and insights from it


# Solution

To deal with this problem, one of the best methods is convolutional Neural Networks that will reduce the number of features using the filter, max pooling, stride, etc.


# Data

Fashion-MNIST is a dataset of Zalando’s article images—comprising a training set of 60,000 examples and a test set of 10,000 examples. 
Each example is a 28x28 gray scale image associated with a label from 10 classes. 
Zalando intends Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms.

Data Description

- Each image is 28 pixels in height and 28 pixels in width, for 784 pixels in total.
- Each pixel has a single pixel-value associated with it, showing the lightness or darkness of that pixel, with higher numbers meaning 
  darker. The pixel-value is an integer between 0 and 255.
- The training and test data sets have 785 columns.
- The first column comprises the class labels (see above) and represents the article of clothing.
- The rest of the columns contain the pixel-values of the associated image.

# Modeling

Create Convolutional Neural Networks (CNN) with these steps:

1- Define model

The first layer in the model network, Keras.layers.Flatten, transforms the format of the images from a two-dimensional array (of 28 by 28 pixels) to a one-dimensional array (of 28 * 28 = 784 pixels).
This layer unstacks rows of pixels in the image and lining them up and has no parameters to learn; it    only reformats the data.
After flatting the pixels, the network comprises a sequence of two Keras layers.Dense layers. These are fully-connected neural layers.
The first dense layer has 32 nodes (or neurons). The second (and last) layer is a 10-node Softmax layer that returns an array of 10 probability scores that sum to 1.
Each node contains a score that shows the probability that the current image belongs to one of the 10 classes.


2- Compile model

Before the model is ready for training, it needs a few more settings. I added these during the model’s compile step:

- Loss function : 

       This measures how accurate the model is during training. You want to minimize this function, to
       “steer” the model  in the right direction. Here we will use “sparse_categorical_crossentropy”  

- Optimizer :

       This is how we can update the model by optimizer based on the data it sees and its loss function.

- Metrics :

      Used to monitor the training and testing steps. The following example uses accuracy, the fraction of  
      the images that model can classify correctly.


3- Train model

   Training the neural network model requires the following steps:
    - Feed the training data to the model. In this example, the training data is in the x_train and y_train 
      arrays. 
    - The model learns to associate images and labels.
    - You ask the model to make predictions about a test set—in this example, the x_test array. Verify that 
      the predictions match the labels from the y_test array.





# Results

Let’s plot training, and validation’s accuracy and loss.

![ana4](https://user-images.githubusercontent.com/33470542/81485204-58ceb180-9219-11ea-8493-9240503cc8d2.png)

![ana5](https://user-images.githubusercontent.com/33470542/81485219-756ae980-9219-11ea-8d2f-1888dd23da41.png)

Classification Report
We can summarize the performance of our classifier as follows

![ana6](https://user-images.githubusercontent.com/33470542/81485231-9a5f5c80-9219-11ea-88b2-4655916888af.png)

Our classifier is underperforming for class 6 in terms of both precision and recall. 
For class 2, the classifier is slightly lacking precision whereas it is slightly lacking recall (i.e. missed) for class 4.

A subset of correctly predicted classes

![ana7](https://user-images.githubusercontent.com/33470542/81485939-39d31e00-921f-11ea-86d1-18120db46e47.png)


A subset of incorrectly predicted classes

![ana8](https://user-images.githubusercontent.com/33470542/81485951-58391980-921f-11ea-9dc8-b60065e929d7.png)


It looks like the diversity of similar patterns present in multiple classes affects the performance of the classifier, although CNN is a robust architect.


