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
# example of loading and plotting the cifar10 dataset
from keras.datasets.cifar10 import load_data
%matplotlib notebook
from matplotlib import pyplot
# load the images into memory
(trainX, trainy), (testX, testy) = load_data()
# plot images from the training dataset
for i in range(49):
	# define subplot
	pyplot.subplot(7, 7, 1 + i)
	# turn off axis
	pyplot.axis('off')
	# plot raw pixel data
	pyplot.imshow(trainX[i])
pyplot.show()

# %%
'''
Running the example creates a figure with a plot of 49 images from the CIFAR-10 training
dataset, arranged in a 7 × 7 square. In the plot, you can see small photographs of planes, trucks,
horses, cars, frogs, and so on.
'''