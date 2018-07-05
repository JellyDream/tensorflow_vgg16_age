# coding=utf-8
# -*- coding: utf-8 -*-


import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg

import os
#os.environ['CUDA_VISIBLE_DEVICES']='0'

os.environ['TF_CPP_MIN_LOG_LEVEL']='0'

class Model(object):

    @staticmethod
    def inference(x, drop_rate):
        '''第1个隐层:224x224x64'''
        with tf.variable_scope('hidden1'):
            #卷积层
            conv = tf.layers.conv2d(x, filters=64, kernel_size=[3, 3], padding='same')
            #规范化
            norm = tf.layers.batch_normalization(conv)
            #激活函数
            activation = tf.nn.relu(norm)
            #dropout层
            dropout = tf.layers.dropout(activation, rate=drop_rate)
            hidden1 = dropout
        '''第2隐层:224x224x64'''
        with tf.variable_scope('hidden2'):
            conv = tf.layers.conv2d(hidden1, filters=64, kernel_size=[3, 3], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden2 = dropout
        '''第3隐层:'''
        with tf.variable_scope('hidden3'):
            conv = tf.layers.conv2d(hidden2, filters=128, kernel_size=[3, 3], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            dropout = tf.layers.dropout(activation, rate=drop_rate)
            hidden3 = dropout
        '''第4隐层'''
        with tf.variable_scope('hidden4'):
            conv = tf.layers.conv2d(hidden3, filters=128, kernel_size=[3, 3], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden4 = dropout
            
        '''第5隐层'''   
        with tf.variable_scope('hidden5'):
            conv = tf.layers.conv2d(hidden4, filters=256, kernel_size=[3, 3], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            dropout = tf.layers.dropout(activation, rate=drop_rate)
            hidden5 = dropout
        
        '''第6隐层'''   
        with tf.variable_scope('hidden6'):
            conv = tf.layers.conv2d(hidden5, filters=256, kernel_size=[3, 3], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            dropout = tf.layers.dropout(activation, rate=drop_rate)
            hidden6 = dropout
        
        '''第7隐层'''  
        with tf.variable_scope('hidden7'):
            conv = tf.layers.conv2d(hidden6, filters=256, kernel_size=[3, 3], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden7 = dropout   
        
        '''第8隐层'''
        with tf.variable_scope('hidden8'):
            conv = tf.layers.conv2d(hidden7, filters=512, kernel_size=[3, 3], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            dropout = tf.layers.dropout(activation, rate=drop_rate)
            hidden8 = dropout    

        '''第9隐层'''  
        with tf.variable_scope('hidden9'):
            conv = tf.layers.conv2d(hidden8, filters=512, kernel_size=[3, 3], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            dropout = tf.layers.dropout(activation, rate=drop_rate)
            hidden9 = dropout 
            
        '''第10隐层'''
        with tf.variable_scope('hidden10'):
            conv = tf.layers.conv2d(hidden9, filters=512, kernel_size=[3, 3], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden10 = dropout    
            
        '''第11隐层'''
        with tf.variable_scope('hidden11'):
            conv = tf.layers.conv2d(hidden10, filters=512, kernel_size=[3, 3], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            dropout = tf.layers.dropout(activation, rate=drop_rate)
            hidden11 = dropout     

        '''第12隐层'''
        with tf.variable_scope('hidden12'):
            conv = tf.layers.conv2d(hidden11, filters=512, kernel_size=[3, 3], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            dropout = tf.layers.dropout(activation, rate=drop_rate)
            hidden12 = dropout
            
        '''第13隐层'''
        with tf.variable_scope('hidden13'):
            conv = tf.layers.conv2d(hidden12, filters=512, kernel_size=[3, 3], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden13 = dropout
        
        flatten = tf.reshape(hidden13, [-1, 7 * 7 * 512])
        
        '''第14隐层''' 
        #全连接
        with tf.variable_scope('hidden14'):
            dense = tf.layers.dense(flatten, units=4096, activation=tf.nn.relu)
            hidden14 = dense
        
        '''第15隐层''' 
        with tf.variable_scope('hidden15'):
            dense = tf.layers.dense(hidden14, units=4096, activation=tf.nn.relu)
            hidden15 = dense
        
        '''第16隐层'''   
        with tf.variable_scope('hidden16'):
            dense = tf.layers.dense(hidden15, units=101)
            hidden16 = dense
        
        age_logits = hidden16
        
        return age_logits

        
    @staticmethod
    def loss(age_labels, age_logits):
        #计算每一个softmax产生的误差,并加和
        loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=age_labels, logits=age_logits))
        return loss
        
        
        
###-------
if __name__=='__main__':
    im = mpimg.imread('images/cat.jpg')/255.
    im_4d = im[np.newaxis]
    x = tf.convert_to_tensor(im_4d, dtype=tf.float32)
    
    label_logits = Model.inference(x, drop_rate=0.)
    label_predictions = tf.argmax(label_logits, axis=1)
    
