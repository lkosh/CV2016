from numpy import zeros
from skimage.data import camera
from skimage.filters import threshold_otsu
from numpy import array, dstack, roll
from skimage.transform import rescale
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import binary_fill_holes
from PIL import ImageStat, Image, ImageEnhance
from numpy import zeros, histogram, interp,array
from skimage.filters.rank import autolevel
import math
import scipy.ndimage
from scipy.ndimage.filters import gaussian_filter, median_filter
from sys import argv, stdout, exit
from os.path import basename
from glob import iglob
from skimage.io import imread, imsave, imshow
import skimage.io as io
from skimage.filters import threshold_otsu, threshold_adaptive,threshold_li,threshold_yen,threshold_isodata
from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank
from skimage.util import img_as_ubyte
from skimage import data, exposure, img_as_float,  img_as_ubyte
from skimage.exposure import is_low_contrast, adjust_log
from skimage.filters.rank import median
from skimage.morphology import dilation, rectangle, white_tophat,selem, remove_small_objects, disk
from skimage.transform import resize
from skimage.measure import regionprops
from skimage.segmentation import clear_border
from skimage.measure import label
from skimage.morphology import closing, square, opening, erosion, diamond
from skimage import feature
from skimage.measure import regionprops
import skimage
from scipy.spatial import distance
from itertools import islice
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
def generate_template(digit_dir_path):
	#x=camera()
	#x=resize(x,(80,40))
	#x=x-x
	x = np.zeros((80,40))
	for i in iglob(digit_dir_path + '/*.bmp'):
		img = imread(i)
		img_correction = binarize(img)
		img=resize(img_correction,(80,40))
		x=x+img
	x=correct2(x)
	
	x[:,:]+=1
	
	
	for region in regionprops(x):
		if region.image.shape[0]==80:
			return region.image[1:-1,1:-1]
def correct2(image,adj1=0.51,adj2=0.4):
		if brightness(image)<=94:#<=94--68 84
			image = exposure.adjust_log(image, adj1)#0.51

			image = gaussian_filter(image,adj2)#0.4
		#binary = histeq(image)
		else:
			image = gaussian_filter(image,0.35)		
		
		binary = threshold_adaptive(image, 27,offset = 0)	
		cleared = binary.copy()
		clear_border(cleared)
		label_image = label(cleared)
		borders = np.logical_xor(binary, cleared)
		label_image[borders] = -1
		
		return label_image

def isodata(image):
	
	image = gaussian_filter(image,0.35)
	binary = threshold_adaptive(image, 27,offset = 0)
	cleared = binary.copy()
	clear_border(cleared)
	label_image = label(cleared)
	borders = np.logical_xor(binary, cleared)
	label_image[borders] = -1
	return label_image
def binarize(img):
	imag = gaussian_filter(img,0.35)
		
	binary = threshold_adaptive(imag, 27,offset = 0)#28
	return binary

	
def brightness( image ):
		# hist,bin = exposure.histogram(image)
		# stat = ImageStat.Stat(hist.tolist())

		# return stat.rms
		return image.sum()/(image.shape[0]*image.shape[1])
def correct(image,adj1=0.54,adj2=0.480):#0.4
		#if brightness(image)<=94:#<=94--68 84
		image = exposure.adjust_log(image, 0.54)#0.51

		image = gaussian_filter(image,0.46)#0.4
		#binary = histeq(image)
		#else:
		#	image = gaussian_filter(image,0.7)#0.35
		#binary = feature.canny(image/255.)

		#binary = binary_fill_holes(binary)
	
		
		binary = threshold_adaptive(image, 27,offset = 0)
		#binary = remove_small_objects(binary,3,connectivity = 2)
		binary = closing(binary,rectangle(2,3))
		
		#if show_img:
	
		#io.imshow(binary)
		#io.show()
		
		cleared = binary.copy()
		#clear_border(cleared)
		label_image = label(cleared,background = 1,connectivity =1)
		#borders = np.logical_xor(binary, cleared)
		#label_image[borders] = -1
		'''fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
		for region in regionprops(label_image):

    # skip small images
			if num(region):
    # draw rectangle around segmented coins
				minr, minc, maxr, maxc = region.bbox
				rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
										  fill=False, edgecolor='red', linewidth=2)
				ax.add_patch(rect)
		#plt.show()
		ax.imshow(label_image)
		io.show()'''
		return label_image
def quality (pic1,pic2,pic3,h,prin = 0):
	if pic1.bbox[3]>pic2.bbox[1] or pic2.bbox[3]>pic3.bbox[1]:
		return 10**20
	
	x1 = (pic1.bbox[0] - pic1.bbox[2]) - (pic2.bbox[0]-pic2.bbox[2])
	x2 = (pic2.bbox[0] - pic2.bbox[2]) - (pic3.bbox[0]-pic3.bbox[2])
	c1 = (pic1.bbox[1] - pic1.bbox[3]) - (pic2.bbox[1]-pic2.bbox[3])
	c2 = (pic2.bbox[1] - pic2.bbox[3]) - (pic3.bbox[1]-pic3.bbox[3])
	
	#q1 = (1.7-(pic1.bbox[2]-pic1.bbox[0])/(pic1.bbox[3]-pic1.bbox[1]))**2
	#q2 = (1.7-(pic2.bbox[2]-pic2.bbox[0])/(pic2.bbox[3]-pic2.bbox[1]))**2
	#q3 = (1.7-(pic3.bbox[2]-pic3.bbox[0])/(pic3.bbox[3]-pic3.bbox[1]))**2
	#if max(q1,q2,q3)>0.5 :
	#	return 10000
	
	#dy  = max(y1,y2)
#	if dx>10 or dy>15:
	#if abs(x1)>5 or abs(x2)>5:
	#	return 10000
	y1 = pic1.bbox[2]#bottom line of number
	y2 = pic2.bbox[2]
	y3 = pic3.bbox[2]
	
	k = max(abs(y1-y2),abs(y2-y3))
	if prin:
		print(k)
	#if k >6 :
	#	return 100000
	d1 = pic1.bbox[3]-pic2.bbox[1]#distance between regions
	d2 = pic2.bbox[3] - pic3.bbox[1]
	if abs(d2-d1)>10:
	#		print(d1,d2)
		return 10**20
	
	

	
	h1 = pic1.bbox[2] -pic1.bbox[0] 
	h2 = pic2.bbox[2] - pic2.bbox[0]
	h3 = pic3.bbox[2] - pic3.bbox[0]
	if prin:
		print(h1,h2,h3)
	m = max(abs(h1-h2),abs(h2-h3))
	start = pic1.bbox[1] 
	#if m>6:
	#	return 10000
	return (k+abs(x1)+abs(x2)+abs(d2-d1) + abs(y1-y2)+abs(y2-y3)+2*abs(d1)+2*abs(d2)+2*(h-min(h1,h2,h3))	)
	##return (dx+dy+d2-d1)/4

def align(prev1,prev2,prev3):
	return tuple(sorted([prev1, prev2, prev3], key=lambda prev: prev.coords[0,][1]))
	
def num(pic):
	return pic.image.shape[0]<45 and pic.image.shape[0]>10 and pic.image.shape[1]<40 and pic.image.shape[1]>6

def number(digit_templates,region):
	min  = 10000
	num = -1
	for j in range(0,10,1):
		
		x = digit_templates[j].shape[1]
		y = digit_templates[j].shape[0]
		sum = ((digit_templates[j] - region) ** 2).sum() / (x * y)
		
		if (sum)<min and sum>0:
			min = sum	
			num = j
		if sum == 0:
			num = j
			
	return num
			
def recognize (img,digit_templates,adj1=0.3,adj2=0.14, drawed=False):
	
	
	img = correct(img,adj1,adj2)
	x = camera()
	x = x-x
	rp = regionprops(x)
	 
	#io.imshow(img)
	#io.show()
	
	reg_props = regionprops(img)
	if not rp == reg_props:
		pic1,pic2,pic3 = reg_props[0],reg_props[0],reg_props[0]
	#pic=[]
		res1,res2,res3 = align(pic1,pic2,pic3)
	#pic.append(pic1,pic2,pic3)
		min = 10**10#quality(pic1,pic2,pic3,img.shape[0])

		reg_props = [reg for reg in reg_props if num(reg)]
		
		for index1, pic1 in enumerate(reg_props):
			for index2, pic2 in islice(enumerate(reg_props), index1 + 1, None):
				for pic3 in islice(reg_props, index2 + 1, None):
					
					pic1,pic2,pic3 = align(pic1,pic2,pic3)
					
					q = quality(pic1,pic2,pic3,img.shape[0])
					if q<min:
						res1,res2,res3 = pic1,pic2,pic3
						min = q
			
		if drawed:
			fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
			for region in (res1,res2,res3):
					minr, minc, maxr, maxc = region.bbox
					rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
											  fill=False, edgecolor='red', linewidth=2)
					ax.add_patch(rect)
			ax.imshow(img)
			io.show()			
		'''fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
		for region in (res1,res2,res3):

    # skip small images
		
    # draw rectangle around segmented coins
				minr, minc, maxr, maxc = region.bbox
				rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
										  fill=False, edgecolor='red', linewidth=2)
				ax.add_patch(rect)
		#plt.show()
		ax.imshow(img)
		io.show()
		#	return 100000
		d1 = res1.bbox[3]-res2.bbox[1]#distance between regions
		d2 = res2.bbox[3] - res3.bbox[1]
	#if abs(d2-d1)>3:
	#	return 10000
	
	
		print(abs(d1),abs(d2))'''
		r1 = res1.image
		r2 = res2.image
		r3 = res3.image

		r1=resize(r1,(78,38))
		r2=resize(r2,(78,38))
		r3=resize(r3,(78,38))
#		print(quality(res1,res2,res3,img.shape[0]))
		
		u1=number(digit_templates,r1)
		u2=number(digit_templates,r2)
		u3=number(digit_templates,r3)
		gt=[]
		gt.append((u1,u2,u3))

		#print(gt)
		return gt[0]
		
	else:
		return[0,0,0]
	
	
