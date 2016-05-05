from numpy import zeros, vstack
import skimage
from skimage import feature
import math
from skimage.feature import hog
from skimage.io import imshow
import skimage.io as io
import numpy as np
import random
from skimage.color import rgb2gray
from sklearn import datasets, svm
from skimage.transform import resize, rescale
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.neighbors import KNeighborsClassifier

def intersect(rectbb,rectgt):
	(x_gt_from, y_gt_from, x_gt_to, y_gt_to) = rectgt
	(x_bb_from, y_bb_from, x_bb_to, y_bb_to) = rectbb
	if (min(x_bb_to, x_gt_to) <= max(x_bb_from, x_gt_from)) or \
	    		(min(y_bb_to, y_gt_to) <= max(y_bb_from, y_gt_from)):
		return False

	intersection = \
		(min(x_bb_to, x_gt_to) - max(x_bb_from, x_gt_from)) * \
		(min(y_bb_to, y_gt_to) - max(y_bb_from, y_gt_from))

	union = \
		(x_bb_to - x_bb_from) * (y_bb_to - y_bb_from) + \
		(x_gt_to - x_gt_from) * (y_gt_to - y_gt_from) - intersection
	sx = (x_bb_to - x_bb_from) * (y_bb_to - y_bb_from)
	sy = (x_gt_to - x_gt_from) * (y_gt_to - y_gt_from)
	if intersection >= 0.3*sx or intersection >= 0.3*sy:
		
		return True
	return (intersection / float(union) >= 0.5)

def delmin(objects,prob):
	
	objects = [i for x , i in sorted(zip(prob,objects),reverse = True)]
	prob.sort(reverse = True)
	i=j=0
	obj = objects[0]
	tmp  = obj
	while  	i < len(objects):
		obj = objects[i]
		
		j=i+1
		
		while j<len(objects):
			tmp = objects[j]
			fl = 0
			if intersect(tmp,obj):
				#print("delete",j)
				objects.remove(objects[j])
				prob.remove(prob[j])
				fl=1
				
			#if tmp != objects[-1]:
			if not fl:	
				j+=1
			
			#print(i,j)
			
		i+=1
		#print("len",len(objects))
	return objects,prob


def detect(model, img, j=0):
	print("second function")
	stepy = stepx = 8
	
	#patch = img[0:coordy,0:coordx]
	#pr.sort(key lambda = pr[0]) sort by probability
	objects = []
	prob = []
	img = rgb2gray(img)
	im = img
	#io.imshow(img)
	#io.show()
	cx = 64//8 -2
	cy = 128//8 -2

	for n in (3/4,1/2,5/12,1/3,3/10,1/4):
#	for n in (1/2,1/4):
		img = rescale(im,n)
		h0 = hog(img,feature_vector = False)
		if j>3:
			figS,axS = plt.subplots(ncols = 1,nrows = 1, figsize = (6,6))
			axS.imshow(img)
		coordx = -stepx
		coordy = -stepy
		#print(img.shape)
		while coordy + stepy +128< img.shape[0]:
			coordy += stepy
			coordx = -stepx 
			while coordx + stepx +64< img.shape[1]:
				coordx += stepx
				#print(coordx,coordy)
				patch = img[coordy:(coordy+128),coordx:(coordx+64)]
				#print(coordy,coordx)
				'''cellRows = cellCols = 5
				binCount = 8
				BlockRowCells = 3
				BlockColCells = 3
				h = hog(patch,binCount,(cellCols,cellRows),(BlockRowCells,BlockColCells))'''
				h = h0[coordy//8:coordy//8+cy,coordx//8:coordx//8+cx]
				
				h = np.ravel(h)
				pr = model.predict_proba(h)
				
				#pr.sort(key lambda = pr[0],reverse = True)
				#ans = pr[1]# ans = most probavle class ?
				#print(type(pr))
				#print(pr.ndim)
				#print(pr[0][0])
				#print(pr[0][1])
				if 0.6<pr[0][1] :
					objects.append([coordx//n,coordy//n,(coordx+64)//n,(coordy+128)//n])
					prob.append(pr[0][1])
					if j>3:
						minc,minr,maxc,maxr = coordx,coordy,coordx+64,coordy+128
						#print("patch")
						#print(pr[0][1])
						#print(minc,minr,maxc,maxr)
						rect = mpatches.Rectangle((minc,minr), maxc - minc, maxr-minr,fill = False, edgecolor = 'red',linewidth = 2)
						axS.add_patch(rect)
						#print(n," rescale", pr[0][1])
						
						
		if j>3:
			plt.show()	
			
#	if prob != []:
#		objects,prob = delmin(objects,prob)
	if j>0:	
		fig,ax = plt.subplots(ncols = 1,nrows = 1, figsize = (6,6))
		ax.imshow(im)
		
	a = np.zeros((0,5))
	objects = [i for x , i in sorted(zip(prob,objects),reverse = True)]
	prob.sort(reverse = True)
	for i in range(len(prob)):
		a = np.vstack((a,[objects[i][0],objects[i][1],objects[i][2],objects[i][3],prob[i]]))
		#print(objects[i][1], objects[i][3], objects[i][0],objects[i][2])
		
		if j>0 :
			
			minc,minr,maxc,maxr = objects[i][0],objects[i][1],objects[i][2],objects[i][3]
			rect = mpatches.Rectangle((minc,minr), maxc - minc, maxr-minr,fill = False, edgecolor = 'red',linewidth = 2)
			ax.add_patch(rect)
			
			ax.imshow(im)
			#print("prob")
			#print(prob[i])
			
					
		#	p = im[objects[i][1]: objects[i][3], objects[i][0]:objects[i][2]]
		#	io.imsave("class2/"+str(j)+"pict"+ str(i)+str(prob[i])+"patch.png" ,p)
		#io.imshow(p)
		#io.show()
		#io.imsave("class/"+str(j)+"image" + str(i) + "patch.png",p)
		#io.show()
		#print(prob[i])
	if j>0:
		plt.show()
	return a
	
def train_detector(imgs, gt):
	
	print("first function")
	features = np.empty((0,6804),float)
	positive = np.empty((0,6804),float)
	negative = np.empty((0,6804),float)
	labels = np.empty((0,1),int)
	count = 0
	imc=0
	'''print(gt[0])
	print(gt[1])
	print(gt[2])
	'''
	for img in imgs:
		#io.imshow(img)
		#io.show()
		#imc+=1
		#print(imc)
		img = rgb2gray(img)

		
		xmin = np.zeros((len(gt[count]),),int)
		xmax = np.zeros((len(gt[count]),),int)
		ymin = np.zeros((len(gt[count]),),int)
		ymax = np.zeros((len(gt[count]),),int)
		for i in range(len(gt[count])):
			xmin[i],ymin[i],xmax[i],ymax[i] = gt[count][i]#???
			patch = img[ymin[i]:ymax[i],xmin[i]:xmax[i]]
			patch = resize(patch,(128,64))
			h = hog(patch)
			features=vstack((features,h))
			labels = vstack((labels,1))
			flippatch = np.fliplr(patch)
			h = hog(flippatch)
			features=vstack((features,h))
			#positive = vstack((positive,h))
			labels = vstack((labels,1))

		#print(xmin[i],ymin[i],xmax[i],ymax[i])
			for shifty in range(-5,5,5):
				if ymax[i]+shifty>=img.shape[0] or ymin[i]+shifty<=0:
						break
				for shiftx in range(-5,5,5):
					
					if xmax[i]+shiftx>=img.shape[1] or xmin[i]+shiftx<=0:
						break
						
					npatch = img[ymin[i]+shifty:ymax[i]+shifty,xmin[i]+shiftx:xmax[i]+shiftx]
					p1 = [xmin[i],ymin[i],xmax[i],ymax[i]]
					p2 = [xmin[i]+shiftx,ymin[i]+shifty,xmax[i]+shiftx,ymax[i]+shifty]
					if not intersect(p1,p2):
							break
					npatch = resize(patch,(128,64))
					
					#io.imsave("positive/" + "img" + str(count) + "positive" + str(i) + ".png",patch)
					#patch = np.resize(patch,(128,64))
					#io.imshow(patch)
					#io.show()
					h = hog(npatch)
					features=vstack((features,h))
					#positive = vstack((positive,h))
					
					labels = vstack((labels,1))
					flippatch = np.fliplr(npatch)
					h = hog(flippatch)
					features=vstack((features,h))
					#positive = vstack((positive,h))
					labels = vstack((labels,1))
		found = 0
			
		while found<4:
			fl = 1
			x1 = random.randint(0,img.shape[1] - 64)
			y1 = random.randint(0,img.shape[0] - 128)
			x2 = x1 + 64
			y2 = y1 + 128
			for i in range(len(gt[count])):
				x_overlap = max(0,min(x2,xmax[i]) - max(x1,xmin[i]))
				y_overlap = max(0,min(y2,ymax[i]) - max(y1,ymin[i]))
				intersection = x_overlap*y_overlap
				if intersection/((xmax[i]- xmin[i])* (ymax[i]-ymin[i]) + 128*64 - intersection)>0.1:
					fl = 0
			if fl:
				found +=1
				
				font = img[y1:y2,x1:x2]
				#io.imshow(font)
				#io.show()
				h = hog(font)
				#io.imsave("negative" + "img" + str(count)+ "negative" + str(found) + ".png",font)
				
				features = vstack((features,h))
				#negative = vstack((negative,h))
				labels = vstack((labels,0))
		count += 1
		
	#print(features)
	lbls = labels
	labels = np.ravel(labels)
	#print(features.shape,labels.shape)
	clf=svm.SVC(C=2,kernel='linear',gamma='auto', probability = True )
	#clf = KNeighborsClassifier(n_neighbors = 2,algorithm = 'ball_tree')
	clf.fit(features, labels)
	

		#print(times)
		#for img in imgs:
			#if len(gt[count]) ==0: 
			
				#a = detect(clf,img)[:,:4]#a stores all positive results of detection
				#img = rgb2gray(img)
				#for k in range(a.shape[0]):
					
					#xmin,ymin,xmax,ymax = a[k]
				
					#bg = img[ymin:ymax,xmin:xmax]
					
					#bg = resize(bg,(128,64))
					
					#h = hog(bg)
					#features = vstack((features,h))
					#lbls = vstack((lbls,0))
			#count+=1
		#labels = np.ravel(lbls)			

	for times in range(1):
		count = 0
		print(times)
		for img in imgs:
			
			img = rgb2gray(img)
			a = detect(clf,img)[:,:4]#a stores all positive results of detection
			for k in range(a.shape[0]):
				fl = 0
				xmin,ymin,xmax,ymax = a[k]
				found = [xmin,ymin,xmax,ymax]	
				for i in range(len(gt[count])):
					x1,y1,x2,y2 =  gt[count][i]
					#print("gt:")
					#print(x1,y1,x2,y2)
					person = [x1,y1,x2,y2]
					if intersect(person,found):
						fl = 1
				if not fl:#adding new background patch
					bg = img[ymin:ymax,xmin:xmax]
					#io.imsave("bootstrap/"+str(count)+"pict"+ str(i)+"patch"+"iteration" + str(times)+".png",bg)

					bg = resize(bg,(128,64))
					
					h = hog(bg)
					features = vstack((features,h))
					lbls = vstack((lbls,0))
			count+=1
		labels = np.ravel(lbls)
		print(lbls.shape)
		clf.fit(features,labels)	 	
				 

	return clf
