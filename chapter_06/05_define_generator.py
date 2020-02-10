# %%
'''
## Define a Generator Model
The next step is to define the generator model. The generator model takes as input a point
from the latent space and generates a new sample, e.g. a vector with both the input and output6.4. Define a Generator Model 78
elements of our function, e.g. x and x2. A latent variable is a hidden or unobserved variable,
and a latent space is a multi-dimensional vector space of these variables. We can define the size
of the latent space for our problem and the shape or distribution of variables in the latent space.
This is because the latent space has no meaning until the generator model starts assigning
meaning to points in the space as it learns. After training, points in the latent space will
correspond to points in the output space, e.g. in the space of generated samples. We will define
a small latent space of five dimensions and use the standard approach in the GAN literature of
using a Gaussian distribution for each variable in the latent space. We will generate new inputs
by drawing random numbers from a standard Gaussian distribution, i.e. mean of zero and a
standard deviation of one.

- Inputs: Point in latent space, e.g. a five-element vector of Gaussian random numbers.
- Outputs: Two-element vector representing a generated sample for our function (x and
x2).

The generator model will be small like the discriminator model. It will have a single Dense
hidden layer with fifteen nodes and will use the ReLU activation function and He weight
initialization. The output layer will have two nodes for the two elements in a generated vector
and will use a linear activation function. A linear activation function is used because we know
we want the generator to output a vector of real values and the scale will be [-0.5, 0.5] for the
first element and about [0.0, 0.25] for the second element.

The model is not compiled. The reason for this is that the generator model is not fit directly.
The define generator() function below defines and returns the generator model. The size of
the latent dimension is parameterized in case we want to play with it later, and the output
shape of the model is also parameterized, matching the function for defining the discriminator
model.
'''

# %%
# define the generator model
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model

# define the standalone generator model
def define_generator(latent_dim, n_outputs=2):
	model = Sequential()
	model.add(Dense(15, activation='relu', kernel_initializer='he_uniform', input_dim=latent_dim))
	model.add(Dense(n_outputs, activation='linear'))
	return model

# define the discriminator model
model = define_generator(5)
# summarize the model
model.summary()
# plot the model
plot_model(model, to_file='generator_plot.png', show_shapes=True, show_layer_names=True)

# %%
from PIL import Image
from IPython.display import display # to display images

image = Image.open('generator_plot.png')
display(image)