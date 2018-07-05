# tensorflow_vgg16_age
Age estimation based on the  VGG16 implemented with tensorflow

## Dependencies
*   Python 2.7/3.5/3.6
*   tensorflow 1.2 or higher version 
*   maybe you should install Anaconda!

## Running step
1. running "python vgg_convert_to_tfrecords.py" to convert images and lables to tfrecords
2. running "python vgg_train.py" to start training
3. running "python vgg_inference_batch.py" to evaluate test set
