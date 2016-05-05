from numpy import ones
import numpy as np
from sklearn import svm
def fit_and_classify(train_features, train_labels, test_features,k='linear',g=0.1,const =1):
	#print("Second function")
	#clf=svm.SVC(C=1,gamma=0.10000000000000001,kernel='linear') #0.9846
	clf=svm.SVC(C=4,kernel='rbf',gamma=0.2 )# rbf 0.2,c=2 - 0.9885, 0.2,4 = 0.9888
	#clf=svm.SVC(C=const,gamma = g,kernel=k,degree = d)
	clf.fit(train_features, train_labels)
	vect=list()
	for i in test_features:
		vect.append(clf.predict(i)[0])
	return vect
