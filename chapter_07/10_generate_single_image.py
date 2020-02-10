# %%
'''
The latent space now defines a compressed representation of MNIST handwritten digits.
You can experiment with generating different points in this space and see what types of numbers
they generate. The example below generates a single handwritten digit using a vector of all 0.0
values.

Note: Your specific results may vary given the stochastic nature of the learning algorithm.
Consider running the example a few times and compare the average performance.
'''

# %%
# example of generating an image for a specific point in the latent space
from keras.models import load_model
from numpy import asarray
%matplotlib notebook
from matplotlib import pyplot
# load model
model = load_model('generator_model_100.h5')
# all 0s
vector = asarray([[0.0 for _ in range(100)]])
# generate image
X = model.predict(vector)
# plot the result
pyplot.imshow(X[0, :, :, 0], cmap='gray_r')
pyplot.show()

# %%
'''
In this case, a vector of all zeros results in a handwritten 9 or maybe an 8. You can then try
navigating the space and see if you can generate a range of similar, but different handwritten
digits.
'''
