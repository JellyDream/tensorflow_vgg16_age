# coding=utf-8
# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt

import os
#os.environ['CUDA_VISIBLE_DEVICES']='0'

os.environ['TF_CPP_MIN_LOG_LEVEL']='0'

# 预处理
def preprocess(image):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.multiply(tf.subtract(image, 0.5), 2)
    image = tf.reshape(image, [224, 224, 3])
    return image

# 产生batch
def build_batch(path_to_tfrecords_file, num_examples, batch_size, shuffled):
    filename_queue = tf.train.string_input_producer([path_to_tfrecords_file], num_epochs=None)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    
    image = tf.decode_raw(features['image'], tf.uint8)
    
    #
    image = preprocess(image)
    label = tf.cast(features['label'], tf.int32)
    
    min_queue_examples = int(0.4 * num_examples)
    
    if shuffled:
        image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                          batch_size=batch_size,
                                                          num_threads=2,
                                                          capacity=min_queue_examples + 3 * batch_size,
                                                          min_after_dequeue=min_queue_examples)
    else:
        image_batch, label_batch = tf.train.batch([image, label],
                                                  batch_size=batch_size,
                                                  num_threads=2,
                                                  capacity=min_queue_examples + 3 * batch_size)
    #返回值解释:
    # image_batch是batch x 224 x 224 x 3
    # label_batch是batch
    return image_batch, label_batch
    
    
    
    
if __name__=='__main__1':
    image_batch, label_batch = build_batch('data/test.tfrecords',
                                           num_examples=4744,
                                           batch_size=32,
                                           shuffled=False)
    
    sess = tf.InteractiveSession()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    image_val,label_val = sess.run([image_batch,label_batch])

    coord.request_stop()
    coord.join(threads)
    sess.close()
    
    print(image_val.shape)
    print('*'*20)
    print(label_val)
    # plt.imshow(image_val[0])
    # plt.show()
    
