# coding=utf-8
# -*- coding: utf-8 -*-

import tensorflow as tf
import get_batch
from vgg_model import Model

import os

# os.environ['TF_CPP_MIN_LOG_LEVEL']='0'

class Evaluator(object):
    def __init__(self, path_to_eval_log_dir):
        self.summary_writer = tf.summary.FileWriter(path_to_eval_log_dir)
        
    def evaluate(self, path_to_checkpoint, path_to_tfrecords_file, num_examples, global_step):
        batch_size = 2
        num_batches = num_examples // batch_size
    
        with tf.Graph().as_default():
            image_batch, label_batch = get_batch.build_batch(path_to_tfrecords_file,
                                                             num_examples=num_examples,
                                                             batch_size=batch_size,
                                                             shuffled=False)
                                                                    
            label_logits = Model.inference(image_batch, drop_rate=0.0)
			
			
            label_predictions = tf.argmax(label_logits, axis=1)
            
            labels = label_batch
            predictions = label_predictions
          
            accuracy, update_accuracy = tf.metrics.accuracy(
                labels=labels,
                predictions=predictions
            )
            
            tf.summary.image('image', image_batch)
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.histogram('variables',
                                 tf.concat([tf.reshape(var, [-1]) for var in tf.trainable_variables()], axis=0))
            summary = tf.summary.merge_all()
            
            with tf.Session() as sess:
                sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)

                restorer = tf.train.Saver()
                restorer.restore(sess, path_to_checkpoint)

                for _ in range(num_batches):
                    sess.run(update_accuracy)

                accuracy_val, summary_val = sess.run([accuracy, summary])
                self.summary_writer.add_summary(summary_val, global_step=global_step)

                coord.request_stop()
                coord.join(threads)
        return accuracy_val
