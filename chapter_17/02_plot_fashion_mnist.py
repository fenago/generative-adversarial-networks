# %%
'''
## Fashion-MNIST Clothing Photograph Dataset
The Fashion-MNIST dataset is proposed as a more challenging replacement dataset for the
MNIST dataset. It is a dataset comprised of 60,000 small square 28 × 28 pixel grayscale images
of items of 10 types of clothing, such as shoes, t-shirts, dresses, and more. Keras provides access
to the Fashion-MNIST dataset via the fashion mnist.load dataset() function. It returns
two tuples, one with the input and output elements for the standard training dataset, and
another with the input and output elements for the standard test dataset. The example below
loads the dataset and summarizes the shape of the loaded dataset.
Note: the first time you load the dataset, Keras will automatically download a compressed
version of the images and save them under your home directory in ∼/.keras/datasets/. The
download is fast as the dataset is only about 25 megabytes in its compressed form.
'''

# %%
# example of loading the fashion_mnist dataset
from keras.datasets.fashion_mnist import load_data
%matplotlib notebook
from matplotlib import pyplot
# load the images into memory
(trainX, trainy), (testX, testy) = load_data()
# plot images from the training dataset
for i in range(100):
	# define subplot
	pyplot.subplot(10, 10, 1 + i)
	# turn off axis
	pyplot.axis('off')
	# plot raw pixel data
	pyplot.imshow(trainX[i], cmap='gray_r')
pyplot.show()

# %%
'''
Running the example creates a figure with a plot of 100 images from the MNIST training
dataset, arranged in a 10 × 10 square.
'''
