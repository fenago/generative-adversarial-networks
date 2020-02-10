# %%
'''
## MNIST Handwritten Digit Dataset
The MNIST dataset is an acronym that stands for the Modified National Institute of Standards
and Technology dataset. It is a dataset of 70,000 small square 28 × 28 pixel grayscale images of
handwritten single digits between 0 and 9. The task is to classify a given image of a handwritten
digit into one of 10 classes representing integer values from 0 to 9, inclusively. Keras provides
access to the MNIST dataset via the mnist.load dataset() function. It returns two tuples,
one with the input and output elements for the standard training dataset, and another with the
input and output elements for the standard test dataset. The example below loads the dataset
and summarizes the shape of the loaded dataset.
Note: the first time you load the dataset, Keras will automatically download a compressed
version of the images and save them under your home directory in ∼/.keras/datasets/. The
download is fast as the dataset is only about eleven megabytes in its compressed form.
'''

# %%
# example of loading the mnist dataset
from keras.datasets.mnist import load_data
# load the images into memory
(trainX, trainy), (testX, testy) = load_data()
# summarize the shape of the dataset
print('Train', trainX.shape, trainy.shape)
print('Test', testX.shape, testy.shape)

# %%
'''
Running the example loads the dataset and prints the shape of the input and output
components of the train and test splits of images. We can see that there are 60K examples in
the training set and 10K in the test set and that each image is a square of 28 by 28 pixels.
Train (60000, 28, 28) (60000,)
Test (10000, 28, 28) (10000,)
'''
