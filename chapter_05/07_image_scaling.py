# %%
'''
## Scale Images to the Range [-1,1]
It is recommended to use the hyperbolic tangent activation function as the output from the
generator model. As such, it is also recommended that real images used to train the discriminator
are scaled so that their pixel values are in the range [-1,1]. This is so that the discriminator
will always receive images as input, real and fake, that have pixel values in the same range.
Typically, image data is loaded as a NumPy array such that pixel values are 8-bit unsigned
integer (uint8) values in the range [0, 255]. First, the array must be converted to floating point
values, then rescaled to the required range. The example below provides a function that will
appropriately scale a NumPy array of loaded image data to the required range of [-1,1].
'''

# %%
# example of a function for scaling images
from numpy.random import randint

# scale image data from [0,255] to [-1,1]
def scale_images(images):
	# convert from unit8 to float32
	images = images.astype('float32')
	# scale from [0,255] to [-1,1]
	images = (images - 127.5) / 127.5
	return images

# define one 28x28 color image
images = randint(0, 256, 28 * 28 * 3)
images = images.reshape((1, 28, 28, 3))
# summarize pixel values
print(images.min(), images.max())
# scale
scaled = scale_images(images)
# summarize pixel scaled values
print(scaled.min(), scaled.max())

# %%
'''
Running the example contrives a single color image with random pixel values in [0,255]. The
pixel values are then scaled to the range [-1,1] and the minimum and maximum pixel values are
then reported.
'''