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
'''