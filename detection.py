
import numpy as np
from numpy import zeros, vstack
import keras
#0,1   outer left eyebrow
#2,3   inner left eyebrow
#4,5   inner right eyebrow
#6,7   outer right eyebrow
#8,9   outer left eye
#10,11  center left eye
#12,13 inner left eye
#14,15 inner right eye
#16,17 center right eye
#18,19 outer right eye
#20,21 nose
#22,23 lip left conner
#24,25 lip center
#26,27 lip right conner
#10 ->16 11->17 12-14 13-15 8-18 9-19 0-6 1-7 2-4 3-5 22-26 23-27
from PIL import Image, ImageDraw
from numpy import ravel
import skimage.io as io

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
#from keras.utils.visualize_util import plot
FACEPOINTS_COUNT = 14
class FlippedImageDataGenerator(ImageDataGenerator):
    flip_indices = [
        (0, 6), (1, 7),
        (2, 4), (3, 5), (10, 16), (11, 17),
        (12, 14), (13, 15), (8, 18), (9, 19),
        (22, 26), (23, 27),
        ]

    def next(self):
        X_batch, y_batch = super(FlippedImageDataGenerator, self).next()
        batch_size = X_batch.shape[0]
        indices = np.random.choice(batch_size, batch_size/2, replace=False)
        X_batch[indices] = X_batch[indices, :, :, ::-1]

        if y_batch is not None:
            y_batch[indices, ::2] = y_batch[indices, ::2] * -1

            for a, b in self.flip_indices:
                y_batch[indices, a], y_batch[indices, b] = (
                    y_batch[indices, b], y_batch[indices, a]
                )

        return X_batch, y_batch


def train_detector(imgs, gts):
	n = len(imgs)
	#visualise(imgs,gts,"check")
	
	X = np.empty((n,96,96),float)
	y = np.empty ((n,FACEPOINTS_COUNT*2),int)
	count = 0
	for img in imgs:
		h,l = img.shape[0], img.shape[1]
		img = rgb2gray(img)
		img = resize(img,(96,96))
		imsh = img
		
		k = 0
		X[count] = img
	
		for i in range(FACEPOINTS_COUNT):
			x0,y0 = gts[count][i]
			y1 = y0*96/h
			x1 = x0*96/l
			y[count][k], y[count][1+k] =  x1, y1
			k+=2 
			#print (y1,x1)
			#rr, cc = circle(y1,x1,1)
			#imsh[rr,cc] = 1
			
		#io.imshow(imsh)
		#io.show()
		count += 1
	y = (y - 48) / 48
	y = y.astype(np.float32)
	X = X.astype(np.float32)
	X /= np.std(X, axis = 0)
		
		
		

	start = 0.03
	stop = 0.001
	nb_epoch = 5
	X = X.reshape(-1, 1, 96, 96)
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

	learning_rates = np.linspace(start, stop, nb_epoch)
	model2 = Sequential()

	model2.add(Convolution2D(32, 3, 3, input_shape = (1, 96, 96)))
#	model2.add(BatchNormalization())

	model2.add(Activation('relu'))
#	model2.add(Convolution2D(32, 3, 3, input_shape = (1, 96, 96)))
#	model2.add(Activation('relu'))
	model2.add(MaxPooling2D(pool_size=(2, 2)))
	model2.add(Dropout(0.1)) # !

	model2.add(Convolution2D(64, 2, 2))
#	model2.add(BatchNormalization())

	model2.add(Activation('relu'))
#	model2.add(Convolution2D(128, 3, 3, input_shape = (1, 96, 96)))
#	model2.add(Activation('relu'))
	model2.add(MaxPooling2D(pool_size=(2, 2)))
	model2.add(Dropout(0.2)) # !
	
	model2.add(Convolution2D(128, 2, 2))
#	model2.add(BatchNormalization())

	model2.add(Activation('relu'))
#	model2.add(Convolution2D(512, 3, 3, input_shape = (1, 96, 96)))
#	model2.add(Activation('relu'))
	model2.add(MaxPooling2D(pool_size=(2, 2)))
	model2.add(Dropout(0.3)) # !

	model2.add(Flatten())
	model2.add(Dense(1000))
#	model2.add(BatchNormalization())
	model2.add(Activation('relu'))
	model2.add(Dropout(0.5)) # !
	model2.add(Dense(1000))
#	model2.add(BatchNormalization())

	model2.add(Activation('relu'))
	model2.add(Dense(28))

	change_lr = LearningRateScheduler(lambda epoch: float(learning_rates[epoch]))
	sgd = SGD(lr=start, momentum=0.9, nesterov=True)
	model2.compile(loss='mean_squared_error', optimizer=sgd)
	#flipgen = ImageDataGenerator()
	early_stop = EarlyStopping(patience = 150)
	model2.fit(X, y, nb_epoch=nb_epoch,
					validation_split = 0.2, callbacks = [change_lr, early_stop])
	#flipgen = FlippedImageDataGenerator()

	#model2.fit_generator(flipgen.flow(X_train, y_train),
        #                     samples_per_epoch=X_train.shape[0],
        #                     nb_epoch=nb_epoch,
        #                     validation_data=(X_val, y_val),
        #                     callbacks=[change_lr, early_stop])


	return model2


def detect(model, imgs):
	count = 0
	print(imgs[0].shape)
	X = np.empty((len(imgs),96,96))
	h = []
	l = []
	for img in imgs:
		img = rgb2gray(img)
		h.append(img.shape[0])
		l.append( img.shape[1])
		img = resize(img,(96,96))
	
		
		X[count] = img
		count += 1 
	X = X.astype(np.float32)
	X = X.reshape(-1, 1, 96, 96)

	pr = model.predict(X)
	pr = np.reshape(pr,(pr.shape[0],14,2))
	pr = pr*48 + 48
	#print("shape of output array ",pr.shape)
	count = 0
	for i in range(len(imgs)):
		
		for j in range(FACEPOINTS_COUNT):
			pr[i][j][0] = pr[i][j][0] * h[count]/96
			pr[i][j][1] = pr[i][j][1] * l[count]/96
			#pr[i][j][0],pr[i][j][1] = pr[i][j][1], pr[i][j][0]
			#print (h[count])
			#print("img: ",i,"fp:", j, "y ", pr[i][j][0], "x ",pr[i][j][1])
		count += 1
	return pr
