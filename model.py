import tensorflow as tf
import numpy as np
#import pandas as pd
import cv2
import json
import random
import os
import sklearn
import csv
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

#from pathlib import PurePosixPath
#from collections import deque

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers import Convolution2D, Input, Cropping2D, AveragePooling2D
from keras.optimizers import Adam
#from keras.utils.visualize_util import plot
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3

from keras import backend as K
# fix the image ordering for tensorflow
K.set_image_dim_ordering("tf")

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_integer('epochs', 5, "The number of epochs.")
flags.DEFINE_integer('batch_size', 256, "The batch size.")
flags.DEFINE_integer('all_cameras',0,"Use left and right cameras")
flags.DEFINE_integer('mirror',0,"Mirror images and steering angle")
flags.DEFINE_integer('jitter',0,"Jitter image and steering and adjust brightness")
flags.DEFINE_float('dropout', .60,
				   "Keep dropout probabilities for nvidia model.")
flags.DEFINE_string('cnn_model', 'nvidia',
					"cnn model either nvidia, vgg19, inceptionv3")

cameras = ['left', 'center', 'right']
camera_centre = ['center']
steering_adj = {'left': 0.25, 'center': 0., 'right': -.25}


# load image and convert to RGB
def load_image(log_path, filename):
	filename = filename.strip()
	if filename.startswith('IMG'):
		filename = log_path+'/'+filename
	else:
		# load it relative to where log file is now, not whats in it
		filename = log_path+'/IMG/'+PurePosixPath(filename).name
	img = cv2.imread(filename)
	# return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# randomily change the image brightness
def randomise_image_brightness(image):
	hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	# brightness - referenced Vivek Yadav post
	# https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.yh93soib0

	bv = .25 + np.random.uniform()
	hsv[::2] = hsv[::2]*bv

	return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


# crop camera image to fit nvidia model input shape
def crop_camera(img, crop_height=66, crop_width=200):
	height = img.shape[0]
	width = img.shape[1]

	# y_start = 60+random.randint(-10, 10)
	# x_start = int(width/2)-int(crop_width/2)+random.randint(-40, 40)
	y_start = 60
	x_start = int(width/2)-int(crop_width/2)

	return img[y_start:y_start+crop_height, x_start:x_start+crop_width]


# referenced Vivek Yadav post
# https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.yh93soib0
def jitter_image_rotation(image, steering):
	rows, cols, _ = image.shape
	transRange = 100
	numPixels = 10
	valPixels = 0.4
	transX = transRange * np.random.uniform() - transRange/2
	steering = steering + transX/transRange * 2 * valPixels
	transY = numPixels * np.random.uniform() - numPixels/2
	transMat = np.float32([[1, 0, transX], [0, 1, transY]])
	image = cv2.warpAffine(image, transMat, (cols, rows))
	return image, steering


def build_nvidia_model(img_height=160, img_width=320, img_channels=3, dropout=0.4):
	#Model based on Nvidia paper https://arxiv.org/pdf/1604.07316v1.pdf
	# build sequential model
	model = Sequential()
	img_shape = (img_height, img_width, img_channels)
	#inp = Input(shape=img_shape)
	# normalisation layer
	model.add(Lambda(lambda x: x * 1./127.5 - 1,
					 input_shape=(img_shape),
					 output_shape=(img_shape), name='Normalization'))
	# Crop image to remove background
	model.add(Cropping2D(cropping=((50,20),(0,0)),input_shape=img_shape))

	# convolution layers with dropout
	nb_filters = [24, 36, 48, 64, 64]
	kernel_size = [(5, 5), (5, 5), (5, 5), (3, 3), (3, 3)]
	same, valid = ('same', 'valid')
	padding = [valid, valid, valid, valid, valid]
	strides = [(2, 2), (2, 2), (2, 2), (1, 1), (1, 1)]

	for l in range(len(nb_filters)):
		model.add(Convolution2D(nb_filters[l],
								kernel_size[l][0], kernel_size[l][1],
								border_mode=padding[l],
								subsample=strides[l],
								activation='elu'))
		model.add(Dropout(dropout))

	# flatten layer
	model.add(Flatten())

	# fully connected layers with dropout
	neurons = [100, 50, 10]
	for l in range(len(neurons)):
		model.add(Dense(neurons[l], activation='elu'))
		model.add(Dropout(dropout))

	# logit output - steering angle
	model.add(Dense(1, name='Out'))

	model.compile(optimizer='adam',
				  loss='mse', metrics=['accuracy'])
	return model

def build_vgg19_pretrained(img_height=160, img_width=320, img_channels=3,
					   dropout=.4,crop = 'True'):
	"""Pretrained VGG16 model with fine-tunable last layer
	"""
	img_shape = (img_height, img_width, img_channels)
	input_image = Input(shape=img_shape)
	base_model = VGG19(input_tensor=input_image, include_top=False)

	for layer in base_model.layers:
		layer.trainable = False

	x = base_model.output
	x = AveragePooling2D((2, 2))(x)
	x = Dropout(dropout)(x)
	x = Flatten()(x)
	x = Dense(4096, activation="relu")(x)
	x = Dropout(dropout)(x)
	x = Dense(2048, activation="relu")(x)
	x = Dense(1, activation="linear")(x)

	model = Model(input=input_image, output=x)
	model.compile(optimizer='adam',
				  loss='mse')
	return model

def build_inceptionv3_pretrained(img_height=160, img_width=320, img_channels=3,
					   dropout=.4,crop = 'True'):
	"""Pretrained inceptionv3 model with fine-tunable last layer
	"""
	img_shape = (img_height, img_width, img_channels)
	input_image = Input(shape=img_shape)
	base_model = InceptionV3(input_tensor=input_image, weights='imagenet',include_top=False)

	for layer in base_model.layers:
		layer.trainable = False

	x = base_model.output
	x = AveragePooling2D((2, 2))(x)
	x = Dropout(dropout)(x)
	x = Flatten()(x)
	x = Dense(4096, activation="relu")(x)
	x = Dropout(dropout)(x)
	x = Dense(2048, activation="relu")(x)
	x = Dense(1, activation="linear")(x)

	model = Model(input=input_image, output=x)
	model.compile(optimizer='adam',
				  loss='mse')
	return model

def inspect(model):
	"""Inspect the underlying model by layer name and parameters
	"""
	for layer in model.layers:
		trainable = None
		# merge layer of keras doesn't have trainable?
		try:
			trainable = layer.trainable
		except:
			pass
		print(layer.name, layer.input_shape, layer.output_shape, trainable)
	return


def generator(samples, batch_size=32,all_cameras=0,mirror=0,jitter=0):
	num_samples = len(samples)
	count = 0
	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			for batch_sample in batch_samples:
				temp_c = batch_sample[0].replace('\\','/')
				temp_l = batch_sample[1].replace('\\','/')
				temp_r = batch_sample[2].replace('\\','/')
				name_c = './data/IMG/'+temp_c.split('/')[-1]
				name_l = './data/IMG/'+temp_l.split('/')[-1]
				name_r = './data/IMG/'+temp_r.split('/')[-1]
				#print(batch_sample[0])
				#print(temp_c)
				#print(name_c)
				#print(batch_sample[3])
				if len(name_c) <= 11:
					print('0')
					count += 1
					print('Count:', count)
				#print(name_l)
				#print(name_r)
				if len(name_c) > 11:
					center_image = cv2.cvtColor(cv2.imread(name_c), cv2.COLOR_BGR2YUV)
				
					#plt.figure(figsize=(2,2))
					#plt.imshow(center_image)
					#center_image = cv2.imread(temp_c) #error
					left_image = cv2.cvtColor(cv2.imread(name_l), cv2.COLOR_BGR2YUV)
					right_image = cv2.cvtColor(cv2.imread(name_r), cv2.COLOR_BGR2YUV)
					center_angle = float(batch_sample[3])
					left_angle = center_angle+0.25
					right_angle = center_angle-0.25
					#Mirror images and inverse steering
					center_image_m = np.fliplr(center_image)
					left_image_m = np.fliplr(left_image)
					right_image_m = np.fliplr(right_image)
					center_angle_m = -center_angle
					left_angle_m = -left_angle
					right_angle_m = -right_angle
					if jitter:
						center_image, center_angle = jitter_image_rotation(center_image, center_angle)
						center_image = randomise_image_brightness(center_image)
					images.append(center_image)
					angles.append(center_angle)
					if all_cameras:
						if jitter:
							left_image, left_angle = jitter_image_rotation(left_image, left_angle)
							left_image = randomise_image_brightness(left_image)
							right_image, right_angle = jitter_image_rotation(right_image, right_angle)
							right_image = randomise_image_brightness(right_image)
						images.append(left_image)
						images.append(right_image)
						angles.append(left_angle)
						angles.append(right_angle)
					if mirror:
						if jitter:
							center_image_m, center_angle_m = jitter_image_rotation(center_image_m, center_angle_m)
							center_image_m = randomise_image_brightness(center_image_m)
						images.append(center_image_m)
						angles.append(center_angle_m)
						if all_cameras:
							if jitter:
								left_image_m, left_angle_m = jitter_image_rotation(left_image_m, left_angle_m)
								left_image_m = randomise_image_brightness(left_image_m)
								right_image_m, right_angle_m = jitter_image_rotation(right_image_m, right_angle_m)
								right_image_m = randomise_image_brightness(right_image_m)
							images.append(left_image_m)
							images.append(right_image_m)
							angles.append(left_angle_m)
							angles.append(right_angle_m)

				# trim image to only see section with road
			X_train = np.array(images)
			y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, y_train)
def main(_):
	samples = []

	with open('./data/driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			samples.append(line)
			#print(line[0])
	from sklearn.model_selection import train_test_split
	train_samples, validation_samples = train_test_split(samples, test_size=0.2)
	
	if FLAGS.all_cameras:
		if FLAGS.mirror:
			sample_mul=6
		else:
			sample_mul=3
	else:
		if FLAGS.mirror:
			sample_mul=2
		else:
			sample_mul=1
	
	print('Number of Total Samples:', (len(samples[:])*sample_mul))
	print('Number of Training Samples:', (len(train_samples[:])*sample_mul))
	
	train_generator = generator(train_samples, batch_size=FLAGS.batch_size,all_cameras=FLAGS.all_cameras,mirror=FLAGS.mirror,jitter=FLAGS.jitter)
	validation_generator = generator(validation_samples, batch_size=FLAGS.batch_size,all_cameras=FLAGS.all_cameras,mirror=FLAGS.mirror,jitter=0)

	cnn_model = FLAGS.cnn_model

	# build model and display layers
	if cnn_model == 'nvidia':
		model = build_nvidia_model(dropout=FLAGS.dropout)
	elif cnn_model == 'vgg19':
		model = build_vgg19_pretrained(dropout=FLAGS.dropout)
	else:
		model = build_inceptionv3_pretrained(dropout=FLAGS.dropout)

	print(model.summary())
	inspect(model)

	#plot(model, to_file='model.png', show_shapes=True)

	history_object = model.fit_generator(train_generator, samples_per_epoch=
			sample_mul*len(train_samples), validation_data=validation_generator,
			nb_val_samples=sample_mul*len(validation_samples), nb_epoch=FLAGS.epochs,
			verbose=1)

	### print the keys contained in the history object
	print(history_object.history.keys())

	# save weights and model
	model.save('model.h5')
	#model.save_weights('model.h5')
	with open('model.json', 'w') as modelfile:
		json.dump(model.to_json(), modelfile)
	print('Model Saved')

	### plot the training and validation loss for each epoch
	plt.plot(history_object.history['loss'])
	plt.plot(history_object.history['val_loss'])
	plt.title('model mean squared error loss')
	plt.ylabel('mean squared error loss')
	plt.xlabel('epoch')
	plt.legend(['training set', 'validation set'], loc='upper right')
	plt.show()
	
# parses flags and calls the `main` function above
if __name__ == '__main__':
	tf.app.run()
