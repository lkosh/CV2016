import numpy as np
import keras
import os
from keras_vgg import vgg16

from skimage.io import imshow
from skimage.draw import circle
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from skimage.transform import resize, rescale
from skimage.color import rgb2gray
from keras.layers import Dropout 
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import LearningRateScheduler
from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import ImageDataGenerator
def train_classifier(imgs, bboxes, gt):
	n_classes = 50
	file_name = "vgg16_weights.h5"
	n = len(gt)
	if not os.path.isfile(file_name):
		print 'file %s not found\n' % file_name
		
	ImageNet = vgg16(file_name)
	ImageNet.pop()
	ImageNet.add(Dense(n_classes, activation = 'softmax', name = 'dense_4'))

	
	Features = np.empty((n,3,224,224),float)
	labels = np.empty((n,50))
	count = 0
	for img in imgs:
		x_start, y_start, length, height = bboxes[count]
		#print x_start, y_start
		img = img[y_start : y_start+height, x_start : x_start+length] 
		img = resize(img,(3,224,224))
		Features[count] = img
		labels[count][gt[count]-1] = 1
		count += 1
	sgd = SGD(lr=0.005, momentum=0.9, nesterov=True)
	ImageNet.compile(loss='mean_squared_error', optimizer=sgd)
	ImageNet.fit(Features, labels, nb_epoch=200, validation_split = 0.2)
	ImageNet.save("saved_weights.h5")
	return ImageNet

def predict(model, imgs, bboxes):
	count = 0
	n = len(imgs)
	Features = np.empty((n,3,224,224),float)
	for img in imgs:
		x_start, y_start, length, height = bboxes[count]
		#print x_start, y_start
		img = img[y_start : y_start+height, x_start : x_start+length] 
		img = resize(img,(3,224,224))
		Features[count] = img
		count += 1
	labels = model.predict_classes(Features)
	
    return labels
