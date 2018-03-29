import tensorflow as tf
import numpy as np
import pandas as pd
from skimage import transform
import cv2
import matplotlib.pyplot as plt
import time
from resize_functions import batch_generator, get_input

def tensor_info(*tensors):
    for tensor in tensors:
        print(tensor.name, tensor.get_shape())


def make_conv2d(name, input_tensor, output_imgs=32, ks=(3,3), activ=tf.nn.relu, st=(1,1), pool_ksize=[2,2]):
#    with tf.name_scope(name):
    conv_w_bias = tf.layers.conv2d(input_tensor, filters=output_imgs, kernel_size=ks, activation=activ, strides=st, name=name+'_conv')
    h1_pool = tf.nn.max_pool(conv_w_bias, ksize=[1,2,2,1], strides=[1,*pool_ksize,1], padding='SAME', name=name+'_pool')
    tensor_info(conv_w_bias, h1_pool)
    return h1_pool


def make_full_connected(name, input_tensor, output=10000, activation=None):
    input_shape = input_tensor.get_shape().as_list()[1:]
    n_units = np.prod(input_shape)
    pool_flat = tf.reshape(input_tensor, shape=[-1, n_units], name=name+'_pool_flat')
    fc = tf.layers.dense(pool_flat, output, name=name+'_full_conected', activation=activation)
    tensor_info(pool_flat, fc)
    return fc

def build_model():   

    input_tensor = tf.placeholder(dtype=tf.float32, shape=(None, 600,1000, 3), name='input')
    tf_y = tf.placeholder(dtype=tf.int32, shape=[None], name='tf_y')
    tf_y_onehot = tf.one_hot(indices=tf_y, depth=4251, dtype=tf.float32, name='tf_y_onehot')
    # Layers
    conv_w_bias_0 = make_conv2d(name='zero_layer', input_tensor=input_tensor, output_imgs=10, ks=(3,3), activ=tf.nn.relu, st=(1,1), pool_ksize=[3,3])
    conv_w_bias_1 = make_conv2d(name='first_layer', input_tensor=conv_w_bias_0, output_imgs=20, ks=(3,3), activ=tf.nn.relu, st=(1,1), pool_ksize=[2,2])
    conv_w_bias_2 = make_conv2d(name='second_layer', input_tensor=conv_w_bias_1, output_imgs=40, ks=(3,3), activ=tf.nn.relu, st=(1,1), pool_ksize=[2,2])
    conv_w_bias_3 = make_conv2d(name='third_layer', input_tensor=conv_w_bias_2, output_imgs=60, ks=(3,3), activ=tf.nn.relu, st=(1,1), pool_ksize=[2,2])
    conv_w_bias_4 = make_conv2d(name='fourth_layer', input_tensor=conv_w_bias_3, output_imgs=80, ks=(3,3), activ=tf.nn.relu, st=(1,1), pool_ksize=[2,2])
    fc = make_full_connected(name='first', input_tensor=conv_w_bias_4, activation=tf.nn.relu)
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    fc_drop = tf.layers.dropout(fc, rate=keep_prob, training=True, name='fc_drop')
    tensor_info(keep_prob,fc_drop)
    fc_2 = make_full_connected(name='second',output=4251, input_tensor=fc_drop, activation=None)
    
    # Vorhersage
    predictions = {'probabilities': tf.nn.softmax(fc_2, name='probabilities'),
                   'labels': tf.cast(tf.argmax(fc_2, axis=1), dtype=tf.int32, name='labels')}
    
    ## Visualisierung des Graphen mit TensorBoard
    ## Verlustfunktion und Optimierung
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=fc_2, labels=tf_y_onehot),
                                        name='cross_entropy_loss')
    
    # Optimierung:
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    optimizer = optimizer.minimize(cross_entropy_loss, name='train_op')
    
    # Berechnung der Korrektklassifizierungsrate
    correct_predictions = tf.equal(predictions['labels'], tf_y, name='correct_preds')
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')       
    tensor_info(tf_y_onehot,cross_entropy_loss, correct_predictions, accuracy)


if __name__=='__main__':    
    

    ## Creating the graph
    g2 = tf.Graph()
    with g2.as_default():
#        build_model()
#        saver = tf.train.Saver()
        
        new_saver = tf.train.import_meta_graph('../model/20180328_epoch_{}.meta'.format(2))
        
        

    X,y = get_input()

    with tf.Session(graph=g2) as sess:
        
#        file_writer = tf.summary.FileWriter('../logs/conv2d_auto', g2)
        
#        sess.run(tf.global_variables_initializer())

        new_saver.restore(sess, '../model/20180328_epoch_{}'.format(2))
        
        
        
        

# Training        
        epochs = 10
        start = time.time()
        for i in range(1,epochs+1):
            batch_gen = batch_generator(X,y, batch_size=50, shuffle=True, random_seed=123, stop=None)
            epoch_loss = 0.0
            progress = 0
            for x_batch, y_batch in batch_gen:
                loss,_ = sess.run(['cross_entropy_loss:0', 'train_op'], feed_dict={'input:0': x_batch,
                                                                                 'tf_y:0':y_batch,
                                                                                 'keep_prob:0': 0.5})
                epoch_loss += loss                
                progress += 1
            print('Loss: {:.2f} Epoch: {} Time: {:.1f}min'.format(epoch_loss, i, (time.time()-start)/60), end=' ')


# Validation
            validation_set = True
            if validation_set:
                batch_gen_valid = batch_generator(X,y, batch_size=100, shuffle=True, random_seed=123, stop=100)
                epoch_loss = 0.0
                for x_batch, y_batch in batch_gen_valid:
                    accuracy = sess.run('accuracy:0', feed_dict={'input:0': x_batch,'tf_y:0':y_batch,'keep_prob:0': 0})
            print('Validation: {:.2f}'.format(accuracy))       
        
        
        
        
        
        
        
        
        
        
        new_saver.save(sess, '../model/20180328_epoch_{}'.format(i))
#        file_writer.close()
        
        

        
        
        
        
        
        
        
        
        