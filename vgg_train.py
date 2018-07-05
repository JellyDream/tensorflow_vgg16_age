# coding=utf-8
# -*- coding: utf-8 -*-

import tensorflow as tf
import os
#os.environ['CUDA_VISIBLE_DEVICES']='0'

os.environ['TF_CPP_MIN_LOG_LEVEL']='0'

from datetime import datetime
import time

import get_batch
from vgg_model import Model
from vgg_evaluator import Evaluator



NUM_TRAIN_EXAMPLES = 33440
NUM_VAL_EXAMPLES = 10560


def train(path_to_train_tfrecords_file,
          num_train_examples,
          path_to_val_tfrecords_file,
          num_val_examples,
          path_to_train_log_dir,
          path_to_restore_checkpoint_file,
          training_options):
    

    batch_size = training_options['batch_size']
    initial_patience = training_options['patience']
    num_steps_to_show_loss = 100
    num_steps_to_check = 1000
    
    with tf.Graph().as_default():

        # image_batch.shape=(batch, 224, 244, 3)
        image_batch, label_batch = get_batch.build_batch(path_to_train_tfrecords_file,
                                               num_examples=num_train_examples,
                                               batch_size=batch_size,
                                               shuffled=True)
                                               
        label_logtis = Model.inference(image_batch, drop_rate=0.5)
        loss = Model.loss(label_batch, label_logtis)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(training_options['learning_rate'], 
                                                   global_step=global_step,
                                                   decay_steps=training_options['decay_steps'], 
                                                   decay_rate=training_options['decay_rate'], 
                                                   staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        
        tf.summary.image('image', image_batch)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('learning_rate', learning_rate)
        summary = tf.summary.merge_all()
        
        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(path_to_train_log_dir, sess.graph)
            evaluator = Evaluator(os.path.join(path_to_train_log_dir, 'eval/val'))

            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            saver = tf.train.Saver()
            if path_to_restore_checkpoint_file is not None:
                assert tf.train.checkpoint_exists(path_to_restore_checkpoint_file), \
                    '%s not found' % path_to_restore_checkpoint_file
                saver.restore(sess, path_to_restore_checkpoint_file)
                print ('Model restored from file: %s' % path_to_restore_checkpoint_file)

            print ('Start training')
            patience = initial_patience
            best_accuracy = 0.0
            duration = 0.0

            while True:
                start_time = time.time()
                _, loss_val, summary_val, global_step_val, learning_rate_val = sess.run([train_op, loss, summary, global_step, learning_rate])
                duration += time.time() - start_time
                
                # num_steps_to_show_loss=100
                if global_step_val % num_steps_to_show_loss == 0:
                    examples_per_sec = batch_size * num_steps_to_show_loss / duration
                    duration = 0.0
                    print ('=> %s: step %d, loss = %f (%.1f examples/sec)' % (
                        datetime.now(), global_step_val, loss_val, examples_per_sec))

                if global_step_val % num_steps_to_check != 0:
                    continue

                summary_writer.add_summary(summary_val, global_step=global_step_val)

                print ('=> Evaluating on validation dataset...')
                path_to_latest_checkpoint_file = saver.save(sess, os.path.join(path_to_train_log_dir, 'latest.ckpt'))
                accuracy = evaluator.evaluate(path_to_latest_checkpoint_file, 
                                              path_to_val_tfrecords_file,
                                              num_val_examples,
                                              global_step_val)
                print ('==> accuracy = %f, best accuracy %f' % (accuracy, best_accuracy))

                if accuracy > best_accuracy:
                    path_to_checkpoint_file = saver.save(sess, os.path.join(path_to_train_log_dir, 'model.ckpt'),
                                                         global_step=global_step_val)
                    print ('=> Model saved to file: %s' % path_to_checkpoint_file)
                    patience = initial_patience
                    best_accuracy = accuracy
                else:
                    patience -= 1

                print ('=> patience = %d' % patience)
                if patience == 0:
                    break

            coord.request_stop()
            coord.join(threads)
            print ('Finished')


    
def main(_):
    path_to_train_tfrecords_file = 'data/train.tfrecords'
    path_to_val_tfrecords_file = 'data/val.tfrecords'
    path_to_train_log_dir = './logs/train'
    path_to_restore_checkpoint_file = None
        


    training_options = {
        'batch_size' : 16,
        'learning_rate' : 1e-2,
        'patience' : 200,
        'decay_steps':10000,
        'decay_rate':0.9
    }
    
    train(path_to_train_tfrecords_file,
          NUM_TRAIN_EXAMPLES,
          path_to_val_tfrecords_file,
          NUM_VAL_EXAMPLES,
          path_to_train_log_dir,
          None,
          training_options)

if __name__ == '__main__':
    tf.app.run(main=main)
    
    
