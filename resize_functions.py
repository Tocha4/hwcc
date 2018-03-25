import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from skimage import transform


fname = pd.read_csv('train.csv')


fig = plt.figure(figsize=(15,12))

mode = {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}
n = 0
for m in mode:
    for image in [i for i in fname.Image[:5]]:   
        
        fig.add_subplot(5,5,n+1)
        frame = cv2.imread('./train/'+image)    
        img_shape = frame.shape
        frame = transform.resize(frame, output_shape=(600,1000), mode=m)    
        plt.imshow(frame)
        plt.title(img_shape)
        n += 1