from numpy import zeros
import numpy
from skimage.data import camera
import numpy as np
import math
from numpy import ones,dstack,zeros,roll, sqrt, square, gradient
import skimage.io as io
from skimage.io import imread, imsave, imshow
from skimage import data, color
import scipy
from scipy import ndimage
from scipy.ndimage import filters
from skimage import img_as_float, img_as_int
from skimage.transform import rescale
from skimage.segmentation import clear_border
def img_transpose(img):
	im_width,im_height = img.shape[0],img.shape[1]
	cost = np.zeros(img.size)
	im_arr = np.reshape(img,(im_height,im_width))
	
	#print(img.shape)
	im_arr = np.transpose(img)
	#im_arr = numpy.reshape(img,(im_height,im_width))
	#im_arr = numpy.transpose(im_arr)
	
	return im_arr
def find_horizontal_seam(img):
	x,y = img.shape[1],img.shape[0]
	cost = img.copy()
	
	for x in range(1,img.shape[1]):
		for y in range(img.shape[0]):
			if y==0:
				min_val= min(cost[y,x-1],cost[y+1,x-1]) 
			elif y==img.shape[0]-1:
				min_val = min(cost[y-1,x-1],cost[y,x-1]) 
			else:
				min_val = min(cost[y-1,x-1],cost[y,x-1],cost[y+1,x-1]) 
			cost[y,x] += min_val
			
	#io.imshow(cost)
	#io.show()
	x,y = img.shape[1],img.shape[0]
	dl = np.zeros(img.shape[1],dtype = 'int')
	#print(matr[y-1,0])
	dl[-1] = np.argmin(cost[:, -1])
	
	#print(dl[y-1])
	#print(matr[1,dl[y-1]+1])
	#i=y-2
	path=[]
	
	path.append((dl[x-1],x-1))
	for i in range(x-2,-1,-1):
		top    = cost[dl[i + 1] - 1, i] if dl[i + 1] > 0     else 10**50
		middle = cost[dl[i + 1], i]
		bottom = cost[dl[i + 1] + 1, i] if dl[i + 1] < y - 1 else 10**50
		dl[i] = np.argmin([top, middle, bottom]) + dl[i + 1] - 1
		"""if dl[i+1]>=1 and dl[i+1]<=y-2:
			
			dl[i] = np.argmin([cost[dl[i+1]-1,i],cost[dl[i+1],i],cost[dl[i+1]+1,i]])
			#print(dl[i]);
			dl[i]+=dl[i+1]-1# if dl[i] == 0 -> dl[i] = dl[i+1]-1
			#print(dl[i]);  
		elif dl[i+1] == 0:
			
			dl[i] = np.argmin([cost[dl[i+1],i],cost[dl[i+1]+1,i]])
			dl[i]+=dl[i+1]
		elif dl[i+1] == y-1:
			
			dl[i] = np.argmin([cost[dl[i+1]-1,i],cost[dl[i+1],i]])
			dl[i]+=dl[i+1]-1"""
		path.append((dl[i],i))
	#print(path)
	return path

def find_vertical_seam(img):
	x,y = img.shape[1],img.shape[0]
	im_arr = img.copy()
	cost = np.zeros(img.shape)
	
	cost[:1,:] = im_arr[:1, :]
	for y in range(1,img.shape[0]):
		for x in range(img.shape[1]):
			if x==0:
				min_val= min(cost[y-1,x],cost[y-1,x+1]) 
			elif x==img.shape[1]-1:
				min_val = min(cost[y-1,x-1],cost[y-1,x]) 
			else:
				min_val = min(cost[y-1,x-1],cost[y-1,x],cost[y-1,x+1]) 
			cost[y,x] = im_arr[y,x] + min_val
	#io.imshow(cost)
	#io.show()
	x,y = img.shape[1],img.shape[0]
	dl = np.zeros(img.shape[0],dtype = 'int')
	minimum = cost[y-1,0]
	#print(matr[y-1,0])
	for i in range(x):
		if minimum>cost[y-1,i]:
			minimum = cost[y-1,i]
			dl[y-1] = i
		
	#print(dl[y-1])
	#print(matr[1,dl[y-1]+1])
	#i=y-2
	path=[]
	
	path.append((y-1,dl[y-1]))
	for i in range(y-2,-1,-1):
		left   = cost[i, dl[i + 1] - 1] if dl[i + 1] > 0     else 10**50
		middle = cost[i, dl[i + 1]]
		right  = cost[i, dl[i + 1] + 1] if dl[i + 1] < x - 1 else 10**50
		dl[i] = np.argmin([left, middle, right]) + dl[i + 1] - 1
		"""if dl[i+1]>=1 and dl[i+1]<=x-2:
			
			dl[i] = np.argmin([cost[i,dl[i+1]-1],cost[i,dl[i+1]],cost[i,dl[i+1]+1]])
			#print(dl[i]);
			dl[i]+=dl[i+1]-1# if dl[i] == 0 -> dl[i] = dl[i+1]-1
			#print(dl[i]);  
		elif dl[i+1] == 0:
			
			dl[i] = np.argmin([cost[i,dl[i+1]],cost[i,dl[i+1]]+1])
			dl[i]+=dl[i+1]
		elif dl[i+1] == x-1:
			
			dl[i] = np.argmin([cost[i,dl[i+1]-1],cost[i,dl[i+1]]])
			dl[i]+=dl[i+1]-1"""
		path.append((i,dl[i]))
	#print(path)
	return path

def delete_horizontal_seam(img,seam):
	h,w = img.shape[0],img.shape[1]
	path = set(seam)
	seen = set()
	flag = 0
	res = img
	#res = np.ones((img.shape[0],img.shape[1]-1,3),dtype= int)
	for x in range(w):
		for y in range(h):
			if (y,x) not in path and not flag:
				res[y,x] = img[y,x]
			elif (y,x) in path :
				flag = 1
			elif flag:
				res[y-1,x] = img[y,x]
		flag = 0
 
	
	res = res[:-1,:]
	return res
def delete_vertical_seam(img,seam):

	h,w = img.shape[0],img.shape[1]
	path = set(seam)
	seen = set()
	flag = 0
	res = img
	#res = np.ones((img.shape[0],img.shape[1]-1,3),dtype= int)
	for y in range(h):
		for x in range(w):
			if (y,x) not in path and not flag:
				res[y,x] = img[y,x]
			elif (y,x) in path :
				flag = 1
			elif flag:
				res[y,x-1] = img[y,x]
		flag = 0
 
	io.imshow(res)
	res = res[:,:-1]
	return res	

def add_vertical_seam(img,seam):
	
	h,w = img.shape[0],img.shape[1]
	path = set(seam)
	seen = set()
	flag = 0
	#res =  np.array(img,dtype = 'uint8')
	res = np.zeros(img.shape,dtype ='uint8')
	if img.ndim == 3:
		res = np.resize(res,(h,w+1,3))
	else:
		res = np.resize(res,(h,w+1))
	for (y,x) in path:
		#print(img[y,x],img[y,x-1])
		if x < res.shape[1]-2:
			a = np.array(img[y,x+1])
		else:
			a = np.array(img[y,x-1])
		a=np.vstack((a,np.array(img[y,x])))
		
		res[y,x] = np.mean(a,axis = 0)
		#print("res",res[y,x])
	#res = np.ones((img.shape[0],img.shape[1]-1,3),dtype= int)
	for y in range(h):
		for x in range(w+1):
			if (y,x) not in path and not flag:
				res[y,x] = img[y,x]
			if (y,x) in path :
				flag = 1
				'''if res.ndim == 3:
					if x<w-1:
						res[y,x] = vector_avg(img[y,x],img[y,x+1])
					else:
						res[y,x] = vector_avg(img[y,x],img[y,x-1])
					
				else:
					if x<w-1:
						res[y,x] = (img[y,x]+img[y,x+1])/2
					else:
						res[y,x] = (img[y,x]+img[y,x-1])/2'''

			elif flag:
				res[y,x] = img[y,x-1]
		flag = 0
	
	return res	
def vector_avg (u, v):
	
	"""
	Returns the component average between each vector
	@u: input vector u
	@v: input vector v
	"""
	w = v
	#print(u[0],v[0])
	w[0] = (u[0]+v[0])//2
	w[1] = (u[1]+v[1])//2
	w[2] = (u[2]+v[2])//2
	#print("w  ",w,"u ",u,"v ",v)
	return w
	'''w = list(u)
	for i in range(len(u)):
		w[i] = (u[i] + v[i]) / 2
	return tuple(w)'''
def add_horizontal_seam(img,seam):
	
	h,w = img.shape[0],img.shape[1]
	path = set(seam)
	seen = set()
	flag = 0
	res = np.zeros(img.shape,dtype ='uint8')

	if img.ndim == 3:
		res = np.resize(res,(h+1,w,3))
	else:
		res = np.resize(res,(h+1,w))
	for (y,x) in path:
		#print(res[y,x],res[y-1,x])
		if y != 0:
			a = np.array(img[y-1,x])
		else:
			a = np.array(img[y+1,x])
		a=np.vstack((a,np.array(img[y,x])))
		#print(a)
		res[y,x] = np.mean(a,axis = 0)
		#print(res[y,x])
	#res = res-res
	#res = np.ones((img.shape[0],img.shape[1]-1,3),dtype= int)
	for x in range(w):
		for y in range(h+1):
			if (y,x) not in path and not flag:
				res[y,x] = img[y,x]
			elif (y,x) in path :
				if y+1!=h:
					flag = 1
					#if res.ndim == 3:
						#print("res",res.ndim,res.dtype)
						#res[y,x] = vector_avg(img[y,x],img[y+1,x])
					#else:
					#	res[y,x] = (img[y,x]+img[y+1,x])/2
				else:
					flag = 1
				#	res[y,x] = img[y,x]
			elif flag:
				res[y,x] = img[y-1,x]
		flag = 0
	
	return res	
def grad(img):
	#img = img.astype('float64')
	#print(img.dtype)
    
	x = img.shape[1]
	y = img.shape[0]
	
	dx = roll(img, 1, axis = 1) - roll(img, -1, axis = 1)
	dx[:,0],dx[:,-1] = 0,0;
	
	dy = roll(img, 1, axis = 0) - roll(img, -1, axis = 0)
	dy[-1,:],dy[0,:] = 0,0;
	
	img = np.square(dx) + np.square(dy)
	img = np.sqrt(img)
	#img = img.astype('int64')
	#print(img.dtype)
	
	##print(img.shape,type(img))
	#print(img)
	return img
	#img = np.gradient(img)

def mask_obj(en,emask):

	en = en + emask*(en.shape[0]*en.shape[1]*256)

	return en
def v_seam(seam,path):
	
	for (y,x) in path:
		seam[y,x] = 1;
	return seam
def add_energy(mask,path):
	
	for (y,x) in path:
		mask[y,x] += 20;
	return mask
def seam_carve(img, mode, mask=None):
	#img = imread('pic_01.png')
	#mask = imread('mask_01.png')
	
	#print(mask)
	#img = img.astype('int64')
	res=img.copy()

	img = np.array(img, dtype='float64')
	#print(img.shape,img.ndim)
	
	if not mask == None:
		#print("mask  ",mask.shape,type(mask))
		mask = mask.astype('float64')
		#io.imshow(mask)
		#io.show()
	#im = color.rgb2gray(img)

	

	x = img.shape[1]
	y = img.shape[0]
	#im = np.zeros((img.shape[0],img.shape[1]), dtype='int64')
	r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
	im = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
	
	en = grad(im)
	#en = filters.sobel(im)
	
	if not mask == None:
		en = mask_obj(en,mask)
		#print(np.amax(mask),np.amin(mask))
	#io.imshow(img)
	#io.show()

	seam_mask = np.zeros((img.shape[0],img.shape[1]),dtype = 'float64')
	#io.imshow(en)
	#io.show()
	if mode == "horizontal expand":
		path = find_vertical_seam(en)
		seam_mask = v_seam(seam_mask,path)
		res = add_vertical_seam(res,path)
		#io.imshow(res)
		#io.show()
		if not mask == None:
			mask = add_vertical_seam(mask,path)
			mask = add_energy(mask,path)
	if mode == "vertical expand":
		path = find_horizontal_seam(en)
		seam_mask = v_seam(seam_mask,path)
		res = add_horizontal_seam(res,path)
		if not mask == None:
			mask = add_horizontal_seam(mask,path)
			mask = add_energy(mask,path)
	if mode == "vertical shrink":
		path = find_horizontal_seam(en)
		seam_mask = v_seam(seam_mask,path)
		res = delete_horizontal_seam(res,path)
		if not mask == None:
			mask = delete_horizontal_seam(mask,path)
	if mode == "horizontal shrink":		 
		path = find_vertical_seam(en)
		seam_mask = v_seam(seam_mask,path)
		res = delete_vertical_seam(res,path)	
		if not mask == None:		
			mask = delete_vertical_seam(mask,path)
	#io.imshow(res)
	#io.show()
	#print(res.dtype)
	#io.imshow(seam_mask)
	#io.show()
	#io.imsave('pic.png',res)
	#io.imsave('mask.png',mask)
	return res,mask,seam_mask

'''img = io.imread('pic_01.png')
mask = io.imread('mask_01.png')
mask = ((mask[:,:,0]!=0)*(-1) + (mask[:,:,1]!=0)).astype('uint8')

seam_carve(img,mask,"horizontal expand")
'''
