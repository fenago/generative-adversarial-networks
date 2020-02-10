# %%
'''
## How to Use the Final Generator Model
Once a final generator model is selected, it can be used in a standalone manner for your
application. This involves first loading the model from file, then using it to generate images. The
generation of each image requires a point in the latent space as input. The complete example of
loading the saved model and generating images is listed below. In this case, we will use the
model saved after 100 training epochs, but the model saved after 40 or 50 epochs would work
just as well
'''

# %%
# example of loading the generator model and generating images
from keras.models import load_model
from numpy.random import randn
%matplotlib notebook
from matplotlib import pyplot

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

# create and save a plot of generated images (reversed grayscale)
def save_plot(examples, n):
	# plot images
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
	pyplot.show()

# load model
model = load_model('generator_model_100.h5')
# generate images
latent_points = generate_latent_points(100, 25)
# generate images
X = model.predict(latent_points)
# plot the result
save_plot(X, 5)

# %%
'''
Running the example first loads the model, samples 25 random points in the latent space,
generates 25 images, then plots the results as a single image.
Note: Your specific results may vary given the stochastic nature of the learning algorithm.
Consider running the example a few times and compare the average performance.
In this case, we can see that most of the images are plausible, or plausible pieces of
handwritten digits
'''