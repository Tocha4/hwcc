import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from skimage import transform
from multiprocessing import Process, Queue

def load_img(count, image):
    
#    print(image)
    img = cv2.imread('../train/'+ image)
    img = transform.resize(img, output_shape=(600,1000), mode='symmetric')
#    q.put(img)
    

def batch_generator(X,y, batch_size=64, shuffle=False, random_seed=None, stop=None):

    idx = np.arange(y.shape[0])
    
    if shuffle:
        rng = np.random.RandomState(random_seed)
        rng.shuffle(idx)
        X = X[idx]
        y = y[idx]
    
    end = X.shape[0]
    if stop != None:
        end = stop
    
    for i in range(0,end, batch_size):

        yy = y[i: i+batch_size]
        XX = np.zeros(shape=(len(yy), 600,1000,3))

        for count,image in enumerate(X[i:i+batch_size]):

#            load_img(count, image)
            img = cv2.imread('../train/'+ image)
            img = transform.resize(img, output_shape=(600,1000), mode='symmetric')
            XX[count] = img
        yield (XX, yy)


def get_input():
        ## Input 
    fname = pd.read_csv('../train.csv')
    yn = fname.Id.unique()
    yn = pd.DataFrame(yn, columns=('Id',))
    yn['digits'] = yn.index
    
    new = pd.merge(fname,yn, on='Id')
    X = np.array(new.Image)
    y = np.array(new.digits)  
    return X,y


if __name__=='__main__':
    
    fname = pd.read_csv('../train.csv')
    yn = fname.Id.unique()
    yn = pd.DataFrame(yn, columns=('Id',))
    yn['digits'] = yn.index
    
    new = pd.merge(fname,yn, on='Id')
    X = np.array(new.Image)
    y = np.array(new.digits)
    
    batch_gen = batch_generator(X,y,batch_size=10, shuffle=True, stop=55)
    
#    X = np.zeros(shape=(10,600,1000,3))
    for n,i in enumerate(batch_gen):
        x_batch,y_batch = i
#        print(y)
        cv2.imshow('img', x_batch[-1])
        key = cv2.waitKey(0)
        if key==113:
            break
