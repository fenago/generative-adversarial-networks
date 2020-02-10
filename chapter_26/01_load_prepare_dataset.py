# %%
'''
## What Is the CycleGAN?
The CycleGAN model was described by Jun-Yan Zhu, et al. in their 2017 paper titled Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks. The benefit of the CycleGAN model is that it can be trained without paired
examples. That is, it does not require examples of photographs before and after the translation
in order to train the model, e.g. photos of the same city landscape during the day and at night.
Instead, the model is able to use a collection of photographs from each domain and extract
and harness the underlying style of images in the collection in order to perform the translation.
The paper provides a good description of the models and training process, although the official
Torch implementation was used as the definitive description for each model and training process
and provides the basis for the model implementations described below.
'''

# %%
'''
## How to Prepare the Horses to Zebras Dataset
One of the impressive examples of the CycleGAN in the paper was to transform photographs of
horses to zebras, and the reverse, zebras to horses. The authors of the paper referred to this as
the problem of object transfiguration and it was also demonstrated on photographs of apples
and oranges. In this tutorial, we will develop a CycleGAN from scratch for image-to-image
translation (or object transfiguration) from horses to zebras and the reverse. We will refer to
this dataset as horses2zebra.

You will see the following directory structure:
horse2zebra
	- testA
	- testB
	- trainA
	- trainB

The A category refers to horse and B category refers to zebra, and the dataset is comprised
of train and test elements. We will load all photographs and use them as a training dataset.
The photographs are square with the shape 256 × 256 and have filenames like n023814602.jpg.
The example below will load all photographs from the train and test folders and create an array
of images for category A and another for category B. Both arrays are then saved to a new file in
compressed NumPy array format.
'''

# %%
# example of preparing the horses and zebra dataset
from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed

# load all images in a directory into memory
def load_images(path, size=(256,256)):
	data_list = list()
	# enumerate filenames in directory, assume all are images
	for filename in listdir(path):
		# load and resize the image
		pixels = load_img(path + filename, target_size=size)
		# convert to numpy array
		pixels = img_to_array(pixels)
		# store
		data_list.append(pixels)
	return asarray(data_list)

# dataset path
path = 'horse2zebra/'
# load dataset A
dataA1 = load_images(path + 'trainA/')
dataAB = load_images(path + 'testA/')
dataA = vstack((dataA1, dataAB))
print('Loaded dataA: ', dataA.shape)
# load dataset B
dataB1 = load_images(path + 'trainB/')
dataB2 = load_images(path + 'testB/')
dataB = vstack((dataB1, dataB2))
print('Loaded dataB: ', dataB.shape)
# save as compressed numpy array
filename = 'horse2zebra_256.npz'
savez_compressed(filename, dataA, dataB)
print('Saved dataset: ', filename)