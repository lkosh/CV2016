import numpy 
import numpy as np
import math
from numpy import ones,dstack,zeros
import skimage.io as io
from skimage.io import imread, imsave
from skimage import data
from skimage import img_as_float
from skimage.transform import rescale
from skimage.segmentation import clear_border
def alignlyr(pic1,pic2,a):
	min=1e+10
	shiftx,shifty = 0,0
	h, w = pic1.shape[0]-2*a, pic1.shape[1]-2*a
	for k in range(-a,a+1):
		for j in range (-a,a+1):
			s = mse(pic1,pic2,k,j,h,w,a)
			#s=metrika(pic1,pic2,k,j,a)
			#s=crosscorr(pic1,pic2,k,j)
			if min>s:
				min=s
				shiftx=k
				shifty=j
			
	return (shiftx,shifty)
def mse(l1, l2, x, y, h, w, a):#№№№ помощь
    tmp = ((np.roll(np.roll(l1, x, 0), y, 1) - l2)**2)
    return tmp[a:h+a, a:w+a].sum()


def align(image):
	#print(image.shape);
	height, width = int(image.shape[0]/3), int(image.shape[1])
	#h = ones(4,dtype = int)
	h,w = 0,0
	x = zeros(4,dtype = int)
	y = zeros(4,dtype = int)
	#w = zeros(4,dtype = int)
	for i in range(1,4):
		h = int(9*height/10)
		w = int(9*width/10)
		x[i]  = (i-1)*height + int(height/20)
		y[i] = int(width/20)
		
	pic1 = image[x[1]:x[1]+h, y[1]:y[1]+w]
	pic2 = image[x[2]:x[2]+h, y[2]:y[2]+w]
	pic3 = image[x[3]:x[3]+h, y[3]:y[3]+w]	
	n,h1,w1 = 1,h,w
	#print(pic1.shape,type(pic2))
	while (h1 > 200) | (w1 > 200):
		h1, w1, n = int(h1/2), int(w1/2), n+1

	
	
	k = ones((n+1),  int)


	k[n] = 15	
	if n > 1: k[n-1] = 4 
	
	if n > 2: k[n-2] = 2

	while n > 0:
			 
		picsmall1 = rescale(pic1, 1/2.**(n-1))
		picsmall2 = rescale(pic2, 1/2.**(n-1))
		picsmall3 = rescale(pic3, 1/2.**(n-1))
		shiftx12,shifty12=alignlyr(picsmall1,picsmall2,k[n])
		shiftx32,shifty32=alignlyr(picsmall3,picsmall2,k[n])
	
		x[1], y[1] = x[1]-shiftx12*2**(n-1), y[1]-shifty12*2**(n-1)
		pic1 = image[x[1]:x[1]+h, y[1]:y[1]+w]
		x[3], y[3] = x[3]-shiftx32*2**(n-1), y[3]-shifty32*2**(n-1)
		pic3 = image[x[3]:x[3]+h, y[3]:y[3]+w]
		n-=1

	picres=dstack((pic3,pic2,pic1))
	
	return picres
