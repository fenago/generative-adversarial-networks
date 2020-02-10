# %%
'''
## Define a Discriminator Model
The next step is to define the discriminator model. The model must take a sample from our
problem, such as a vector with two elements, and output a classification prediction as to whether
the sample is real or fake. This is a binary classification problem.

- Inputs: Sample with two real values.
- Outputs: Binary classification, likelihood the sample is real (or fake).

The problem is very simple, meaning that we don’t need a complex neural network to model
it. The discriminator model will have one hidden layer with 25 nodes and we will use the
ReLU activation function and an appropriate weight initialization method called He weight
initialization. The output layer will have one node for the binary classification using the sigmoid
activation function. The model will minimize the binary cross-entropy loss function, and the
Adam version of stochastic gradient descent will be used because it is very effective. The
define discriminator() function below defines and returns the discriminator model. The
function parameterizes the number of inputs to expect, which defaults to two.
'''

# %%
# define the discriminator model
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model

# define the standalone discriminator model
def define_discriminator(n_inputs=2):
	model = Sequential()
	model.add(Dense(25, activation='relu', kernel_initializer='he_uniform', input_dim=n_inputs))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# define the discriminator model
model = define_discriminator()
# summarize the model
model.summary()
# plot the model
plot_model(model, to_file='discriminator_plot.png', show_shapes=True, show_layer_names=True)

# %%
from PIL import Image
from IPython.display import display # to display images

image = Image.open('discriminator_plot.png')
display(image)