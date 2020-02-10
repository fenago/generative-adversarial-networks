# %%
'''
## Satellite to Map Image Translation Dataset
In this tutorial, we will use the so-called maps dataset used in the Pix2Pix paper. This is a
dataset comprised of satellite images of New York and their corresponding Google maps pages.
The image translation problem involves converting satellite photos to Google maps format, or
the reverse, Google maps images to Satellite photos. The dataset is provided on the Pix2Pix
website and can be downloaded as a 255-megabyte zip file.

You will a directory called maps/ with the following structure:

maps
 - train
 - val

The train folder contains 1,097 images, whereas the validation dataset contains 1,099 images.
Images have a digit filename and are in JPEG format. Each image is 1,200 pixels wide and 600
pixels tall and contains both the satellite image on the left and the Google maps image on the
right.
'''

# %%
'''
We can prepare this dataset for training a Pix2Pix GAN model in Keras. We will just work
with the images in the training dataset. Each image will be loaded, rescaled, and split into
the satellite and Google Maps elements. The result will be 1,097 color image pairs with the
width and height of 256 × 256 pixels. The load images() function below implements this. It
enumerates the list of images in a given directory, loads each with the target size of 256 × 512
pixels, splits each image into satellite and map elements and returns an array of each.
'''

# %%
# load, split and scale the maps dataset ready for training
from os import listdir
from numpy import asarray
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed

# load all images in a directory into memory
def load_images(path, size=(256,512)):
	src_list, tar_list = list(), list()
	# enumerate filenames in directory, assume all are images
	for filename in listdir(path):
		# load and resize the image
		pixels = load_img(path + filename, target_size=size)
		# convert to numpy array
		pixels = img_to_array(pixels)
		# split into satellite and map
		sat_img, map_img = pixels[:, :256], pixels[:, 256:]
		src_list.append(sat_img)
		tar_list.append(map_img)
	return [asarray(src_list), asarray(tar_list)]

# dataset path
path = 'maps/train/'
# load dataset
[src_images, tar_images] = load_images(path)
print('Loaded: ', src_images.shape, tar_images.shape)
# save as compressed numpy array
filename = 'maps__256.npz'
savez_compressed(filename, src_images, tar_images)
print('Saved dataset: ', filename)


# %%
'''
Running the example loads all images in the training dataset, summarizes their shape to
ensure the images were loaded correctly, then saves the arrays to a new file called maps 256.npz
in compressed NumPy array format.

Loaded: (1096, 256, 256, 3) (1096, 256, 256, 3)
Saved dataset: maps_256.npz

This file can be loaded later via the load() NumPy function and retrieving each array in
turn. We can then plot some images pairs to confirm the data has been handled correctly.
'''