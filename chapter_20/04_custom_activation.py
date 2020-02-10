# %%
'''
## Stacked Discriminator Models With Shared Weights
A final approach is very similar to the prior two semi-supervised approaches and involves creating
separate logical unsupervised and supervised models but attempts to reuse the output layers of
one model to feed as input into another model. The approach is based on the definition of the
semi-supervised model in the 2016 paper by Tim Salimans, et al. from OpenAI titled Improved
Techniques for Training GANs. In the paper, they describe an efficient implementation, where
first the supervised model is created with K output classes and a softmax activation function.
The unsupervised model is then defined that takes the output of the supervised model prior to
the softmax activation, then calculates a normalized sum of the exponential outputs.

To make this clearer, we can implement this activation function in NumPy and run some
sample activations through it to see what happens. The complete example is listed below.
'''

# %%
# example of custom activation function
import numpy as np

# custom activation function
def custom_activation(output):
	logexpsum = np.sum(np.exp(output))
	result = logexpsum / (logexpsum + 1.0)
	return result

# all -10s
output = np.asarray([-10.0, -10.0, -10.0])
print(custom_activation(output))
# all -1s
output = np.asarray([-1.0, -1.0, -1.0])
print(custom_activation(output))
# all 0s
output = np.asarray([0.0, 0.0, 0.0])
print(custom_activation(output))
# all 1s
output = np.asarray([1.0, 1.0, 1.0])
print(custom_activation(output))
# all 10s
output = np.asarray([10.0, 10.0, 10.0])
print(custom_activation(output))

# %%
'''
Remember, the output of the unsupervised model prior to the softmax activation function
will be the activations of the nodes directly. They will be small positive or negative values, but
not normalized, as this would be performed by the softmax activation. The custom activation
function will output a value between 0.0 and 1.0. A value close to 0.0 is output for a small or
negative activation and a value close to 1.0 for a positive or large activation. We can see this
when we run the example.
'''