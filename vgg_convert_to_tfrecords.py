# coding=utf-8
# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import numpy as np

def convert_to_tfrecords(txtfile):
    with open(txtfile) as f:
    
        dataset_type = txtfile.split('.')[0].split('/')[1]

        SAVE_PATH = 'data/{0}.tfrecords'.format(dataset_type)
    

        writer = tf.python_io.TFRecordWriter(SAVE_PATH)
        count=0
        while True:
            try:
                count+=1
                
                imagefile_and_label = os.path.join('data/',next(f))
                imagefile, label = imagefile_and_label.split()
                
                print('Processing {0}: {1},  filename:{2}'.format(dataset_type,count,imagefile))
                
                image = mpimg.imread(imagefile)

                image = cv2.resize(image,(224,224)).tobytes()

                label = int(label)
                
                feature = {
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                }
                example = tf.train.Example(features = tf.train.Features(feature = feature))
                writer.write(example.SerializeToString())
    
            except StopIteration:
                break
        writer.close()
            
            
if __name__=='__main__':
    file_list = ['train.txt','val.txt','test.txt']
    for txtfile in file_list:
        txtfile = os.path.join('data/',txtfile)
        convert_to_tfrecords(txtfile)
