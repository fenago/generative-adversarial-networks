# %%
'''
## Convolutional Neural Network
In this section, we will define a convolutional neural network for image classification. The model
receives black and white 64 X 64 images as input, then has a sequence of two convolutional and
pooling layers as feature extractors, followed by a fully connected layer to interpret the features
and an output layer with a sigmoid activation for two-class predictions. 
'''

# %%
# example of a convolutional neural network
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
visible = Input(shape=(64,64,1))
conv1 = Conv2D(32, (4,4), activation='relu')(visible)
pool1 = MaxPooling2D()(conv1)
conv2 = Conv2D(16, (4,4), activation='relu')(pool1)
pool2 = MaxPooling2D()(conv2)
flat1 = Flatten()(pool2)
hidden1 = Dense(10, activation='relu')(flat1)
output = Dense(1, activation='sigmoid')(hidden1)
model = Model(inputs=visible, outputs=output)


# %%
''' 
Running the example prints the structure of the network. A plot of the model graph is also created and saved to file.
'''

# %%
# summarize layers
model.summary()
# plot graph
plot_model(model, to_file='convolutional_neural_network.png')


# %%
from PIL import Image
from IPython.display import display # to display images

image = Image.open('convolutional_neural_network.png')
display(image)