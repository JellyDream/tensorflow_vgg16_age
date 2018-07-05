# coding=utf-8
# -*- coding: utf-8 -*-

import tensorflow as tf
import get_batch
from vgg_model import Model

NUM_EXAMPLES = 4744

path_to_restore_checkpoint_file = './logs/train/model.ckpt-22000'
path_to_tfrecords_file = 'data/test.tfrecords'

def batch_inference(path_to_tfrecords_file = 'data/test.tfrecords',
                    num_examples = NUM_EXAMPLES,
                    batch_size=10):
    num_batches = num_examples // batch_size
    
    with tf.Graph().as_default():
        image_batch, label_batch = get_batch.build_batch(path_to_tfrecords_file,
                                                 num_examples=num_examples,
                                                 batch_size=batch_size,
                                                 shuffled=False)
        label_logits = Model.inference(image_batch, drop_rate=0.0)
        label_predictions = tf.to_int32(tf.argmax(label_logits, axis=1))
        
        sub = tf.abs(label_predictions-label_batch)
        sub_mean = tf.reduce_mean(sub)

        label_softmax = tf.nn.softmax(label_logits)
        label_regression = tf.reduce_sum(label_softmax*tf.constant(range(101),dtype=tf.float32),1)
        sub1 = tf.abs(label_regression-tf.to_float(label_batch))
        sub1_mean = tf.reduce_mean(sub1)
        #sess
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            restorer = tf.train.Saver()
            restorer.restore(sess, path_to_restore_checkpoint_file)
            
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            L = []
            for i in range(num_batches):
                label_regression_val = sess.run(label_regression)
                #print(label_regression_val)
                sub1_mean_val = sess.run(sub1_mean)
                L.append(sub1_mean_val)
                print('process:',i)
                print(float(sum(L))/len(L))
                #	
            	#sub_mean_val = sess.run(sub_mean)
                #L.append(sub_mean_val)
                #print('process:',i)
                #print(float(sum(L))/len(L))
            
            return sum(L)/len(L)
            
            coord.request_stop()
            coord.join(threads)

                                                 
if __name__=='__main__':
    print(batch_inference())
                                                 
                                                 
