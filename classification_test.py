import numpy as np
import sys
import time
from skimage.io import imread
from classification import train_classifier, predict

from sklearn import cross_validation

def load_data(path):
    if path[-1] != '/':
        path += '/'
    data = open(path+'gt.txt','r').readlines()
    imgs = []
    bboxes = np.zeros((len(data), 4))
    labels = np.zeros(len(data),dtype = np.int)
    for i, line in enumerate(data):
        line_split = (line[:-1] if line[-1] == '\n' else line).split(' ')
        bboxes[i,:] = [float(x) for x in line_split[2:6]]
        labels[i] = int(line_split[1])
        imgs.append(imread(path+"images/"+line_split[6],plugin="matplotlib"))
    return (imgs,bboxes,labels)

if len(sys.argv) < 3:
    print("Usage %s train_folder test_folder" % sys.argv[0])
    sys.exit(1)
start_time = time.time()
train_dir = sys.argv[1]
test_dir = sys.argv[2]
train_imgs, train_bboxes, train_gt = load_data(train_dir)
train_bb_gt = zip(train_bboxes, train_gt)
X_train,X_test,y_train,y_test = cross_validation.train_test_split(train_imgs,train_bb_gt, 
test_size = 0.1, train_size = 0.05,random_state = 2	)
train_imgs = X_train
test_imgs = X_test
train_bboxes, train_gt = zip(*y_train)
test_bboxes, test_gt = zip(*y_test)

model = train_classifier(train_imgs, train_bboxes, train_gt)
del train_imgs, train_bboxes, train_gt
#test_imgs, test_bboxes, test_gt = load_data(test_dir)
predicted_labels = np.array(predict(model, test_imgs, test_bboxes))
score = 1 - np.count_nonzero(test_gt-predicted_labels)/float(len(test_gt))
print("Score: %.4f" % score)
end_time = time.time()
print("Running time: %.2f s (%.2f minutes)" %
      (round(end_time - start_time, 2), round((end_time - start_time) / 60, 2)))
