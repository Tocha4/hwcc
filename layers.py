import tensorflow as tf
import numpy as np
import pandas as pd
from skimage import transform
import cv2
import matplotlib.pyplot as plt

def tensor_info(*tensors):
    for tensor in tensors:
        print(tensor.name, tensor.get_shape())


def make_conv2d(name, input_tensor, output_imgs=32, ks=(3,3), activ=tf.nn.relu, st=(1,1)):
    with tf.name_scope(name):
        conv_w_bias = tf.layers.conv2d(input_tensor, filters=output_imgs, kernel_size=ks, activation=activ, strides=st, name='conv_l1')
        tensor_info(conv_w_bias)


def make_full_connected():
    pass
    

    




if __name__=='__main__':    
    
    ## Input 
    fname = pd.read_csv('train.csv')
    img = cv2.imread('./train/'+ fname.Image[7])
    img = transform.resize(img, output_shape=(600,1000), mode='symmetric')
    img = np.array([img])
    
    
    ## Creating the graph
    g2 = tf.Graph()
    with g2.as_default():
        input_tensor = tf.placeholder(dtype=tf.float32, shape=(None, 600,1000, 3), name='input')
        conv_w_bias = make_conv2d(name='first_layer', input_tensor=input_tensor, output_imgs=32, ks=(3,3), activ=tf.nn.relu, st=(1,1))
    
    
    ## Run the Session
    with tf.Session(graph=g2) as sess:
        file_writer = tf.summary.FileWriter('./logs/conv2d_auto', g2)
        sess.run(tf.global_variables_initializer())
        new_imgs = sess.run('first_layer/conv_l1/Relu:0', feed_dict={'input:0': img})
        file_writer.close()
        
        
    ## Show results  
    fig = plt.figure()
    for n in range(12):
        fig.add_subplot(3,4,n+1)
        plt.imshow(new_imgs[0,:,:,n])
        
        
        
        
        
        
        
        
        
        
        