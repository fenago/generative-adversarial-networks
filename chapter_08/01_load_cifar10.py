# %%
'''
## CIFAR-10 Small Object Photograph Dataset
CIFAR is an acronym that stands for the Canadian Institute For Advanced Research and
the CIFAR-10 dataset was developed along with the CIFAR-100 dataset (covered in the next
section) by researchers at the CIFAR institute. The dataset is comprised of 60,000 32 × 32
pixel color photographs of objects from 10 classes, such as frogs, birds, cats, ships, airplanes,
etc. These are very small images, much smaller than a typical photograph, and the dataset
is intended for computer vision research. Keras provides access to the CIFAR-10 dataset via
the cifar10.load dataset() function. It returns two tuples, one with the input and output
elements for the standard training dataset, and another with the input and output elements for
the standard test dataset. The example below loads the dataset and summarizes the shape of
the loaded dataset.

Note: the first time you load the dataset, Keras will automatically download a compressed
version of the images and save them under your home directory in ∼/.keras/datasets/. The
download is fast as the dataset is only about 163 megabytes in its compressed form.
'''

# %%
# example of loading the cifar10 dataset
from keras.datasets.cifar10 import load_data
# load the images into memory
(trainX, trainy), (testX, testy) = load_data()
# summarize the shape of the dataset
print('Train', trainX.shape, trainy.shape)
print('Test', testX.shape, testy.shape)


# %%
'''
Running the example loads the dataset and prints the shape of the input and output
components of the train and test splits of images. We can see that there are 50K examples in
the training set and 10K in the test set and that each image is a square of 32 by 32 pixels.
'''
