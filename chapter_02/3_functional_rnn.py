# %%
'''
## Recurrent Neural Network
In this section, we will define a long short-term memory recurrent neural network for sequence
classification. The model expects 100 time steps of one feature as input. The model has a single
LSTM hidden layer to extract features from the sequence, followed by a fully connected layer to
interpret the LSTM output, followed by an output layer for making binary predictions
'''

# %%
# example of a recurrent neural network
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers.recurrent import LSTM
visible = Input(shape=(100,1))
hidden1 = LSTM(10)(visible)
hidden2 = Dense(10, activation='relu')(hidden1)
output = Dense(1, activation='sigmoid')(hidden2)
model = Model(inputs=visible, outputs=output)

# %%
''' 
Running the example prints the structure of the network. A plot of the model graph is also created and saved to file.
'''

# %%
# summarize layers
model.summary()
# plot graph
plot_model(model, to_file='recurrent_neural_network.png')

# %%
from PIL import Image
from IPython.display import display # to display images

image = Image.open('recurrent_neural_network.png')
display(image)