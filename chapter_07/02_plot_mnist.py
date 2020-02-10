# %%
'''
The images are grayscale with a black background (0 pixel value) and the handwritten digits
in white (pixel values near 255). This means if the images were plotted, they would be mostly
black with a white digit in the middle. We can plot some of the images from the training dataset
using the Matplotlib library using the imshow() function and specify the color map via the
cmap argument as ‘gray’ to show the pixel values correctly.7.2. MNIST Handwritten Digit Dataset 97

Alternately, the images are easier to review when we reverse the colors and plot the
background as white and the handwritten digits in black. They are easier to view as most of
the image is now white with the area of interest in black. This can be achieved using a reverse
grayscale color map, as follows:
'''


# %%
# example of loading the mnist dataset
from keras.datasets.mnist import load_data
%matplotlib notebook
from matplotlib import pyplot
# load the images into memory
(trainX, trainy), (testX, testy) = load_data()
# plot images from the training dataset
for i in range(25):
	# define subplot
	pyplot.subplot(5, 5, 1 + i)
	# turn off axis
	pyplot.axis('off')
	# plot raw pixel data
	pyplot.imshow(trainX[i], cmap='gray_r')
pyplot.show()


# %%
'''
Running the example creates a plot of 25 images from the MNIST training dataset, arranged
in a 5 × 5 square.

We will use the images in the training dataset as the basis for training a Generative Adversarial
Network. Specifically, the generator model will learn how to generate new plausible handwritten
digits between 0 and 9, using a discriminator that will try to distinguish between real images
from the MNIST training dataset and new images output by the generator model. This is a
relatively simple problem that does not require a sophisticated generator or discriminator model,
although it does require the generation of a grayscale output image.
'''
