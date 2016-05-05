from numpy import zeros
import math
from math import atan2
from skimage.data import camera
from skimage.filters import threshold_otsu
from numpy import array, dstack, roll
from skimage.transform import rescale
import numpy as np
import matplotlib.pyplot as plt
from numpy import zeros
import scipy.ndimage
from sys import argv, stdout, exit
from os.path import basename
from glob import iglob
from skimage.io import imread, imsave
from skimage.filters import threshold_otsu, threshold_adaptive,threshold_li,threshold_yen,threshold_isodata
from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank, sobel_h, sobel_v
from skimage.util import img_as_ubyte
from skimage import data, exposure, img_as_float
from skimage.filters.rank import median
from skimage.morphology import disk
from skimage.transform import resize
from skimage.measure import regionprops
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing, square
from skimage.measure import regionprops

import skimage
from scipy import sqrt, pi, arctan2, cos, sin
from scipy.ndimage import uniform_filter

from skimage.color import rgb2gray
def extract_hog(img,ROI):
	#print("first")
	#cellRows = cellCols = 5
	#binCount = 4
#	BlockRowCells = 3
#	BlockColCells = 3
	orientations=8
	pixels_per_cell=(5, 5)#5,5 - 0.9844
	cells_per_block=(3, 3)#3,3
	img = resize(img,(50,50))
	image = rgb2gray(img)

	image = np.atleast_2d(image)
	#hist = hog(img,binCount,(cellCols,cellRows),(BlockRowCells,BlockColCells))
	#hist = np.divide(hog,np.linalg.norm(hog))
	gx = roll(image, 1, axis = 1) - roll(image, -1, axis = 1)
	gx[:,0],gx[:,-1] = 0,0;
	
	gy = roll(image, 1, axis = 0) - roll(image, -1, axis = 0)
	gy[-1,:],gy[0,:] = 0,0;
	matr = np.square(gx) + np.square(gy)
	matr = np.sqrt(matr)
	orientation = arctan2(gy, (gx + 1e-15)) * (180 / pi) + 90

	imx, imy = image.shape
	cx, cy = pixels_per_cell
	bx, by = cells_per_block

	n_cellsx = int(np.floor(imx // cx))  # number of cells in i
	n_cellsy = int(np.floor(imy // cy))  # number of cells in j

    
	or_hist = np.zeros((n_cellsx, n_cellsy, orientations))
	for i in range(orientations):
		
	
		condition = orientation < 180 / orientations * (i + 1)

		tmp = np.where(condition,orientation, 0)
		condition = orientation >= 180 / orientations * i
		tmp = np.where(condition,tmp, 0)
	
		cond2 = tmp > 0
		temp_mag = np.where(cond2, matr, 0)

		or_hist[:,:,i] = uniform_filter(temp_mag, size=(cx, cy))[cx/2::cx, cy/2::cy].T
	numbx = (n_cellsx - bx) + 1
	numby = (n_cellsy - by) + 1
	normb = np.zeros((numbx, numby, bx, by, orientations))

	for i in range(numbx):
		for j in range(numby):
			block = or_hist[i:i + bx, j:j + by, :]
			eps = 1e-5
			normb[i, j, :] = block / sqrt(block.sum() ** 2 + eps)

    
   
	return normb.ravel()
	

	
	
