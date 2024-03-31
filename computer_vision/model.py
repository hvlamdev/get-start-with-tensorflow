# Import and configure logging
import logging
import google.cloud.logging as cloud_logging
from google.cloud.logging.handlers import CloudLoggingHandler
from google.cloud.logging_v2.handlers import setup_logging

cloud_logger = logging.getLogger('cloudLogger')
cloud_logger.setLevel(logging.INFO)
cloud_logger.addHandler(CloudLoggingHandler(cloud_logging.Client()))
cloud_logger.addHandler(logging.StreamHandler())

# Import TensorFlow
import tensorflow as tf

# Import numpy
import numpy as np

# Import tensorflow_datasets
import tensorflow_datasets as tfds

# Define, load and configure data
# split: specify which splits of the dataset is to be loaded.
# as_supervised= True: ensure that the loaded tf.data.Dataset will have a 2-tuple structure (input, label).
# ds_train and ds_test are of type tf.data.Dataset. 
# ds_train has 60,000 images which will be used for training the model.
# ds_test has 10,000 images which will be used for evaluating the model.
(ds_train, ds_test), info = tfds.load('fashion_mnist', split=['train', 'test'], with_info=True, as_supervised=True)

# Values before normalization
image_batch, labels_batch = next(iter(ds_train))

# See the min and max values of training images for item 0.
print("Before normalization ->", np.min(image_batch[0]), np.max(image_batch[0]))

# Batch size is a term used in machine learning and refers to the number of training examples utilized in one iteration.
# Define batch size
BATCH_SIZE = 32


# When training a neural network, for various reasons it's easier if you scale the pixel values to the range between 0 to 1.
# This process is called 'normalization'. 
# Since the pixel values for FashionMNIST dataset are in the range of [0, 255], you will divide the pixel values by 255.0 to normalize the images.
# Normalize and batch process the dataset
# Pixel values type is tf.uint8, so you need to cast them to tf.float32.
ds_train = ds_train.map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y)).batch(BATCH_SIZE)
ds_test = ds_test.map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y)).batch(BATCH_SIZE)


# Re-print the min and max values of an image in the train dataset:
# Examine the min and max values of the batch after normalization
image_batch, labels_batch = next(iter(ds_train))
print("After normalization ->", np.min(image_batch[0]), np.max(image_batch[0]))


# Define the model
# Sequential: This defines a SEQUENCE of layers in the neural network.
# Flatten: Our images are of shape (28, 28), i.e, the values are in the form of a square matrix.
# Flatten takes that square and turns it into a one-dimensional vector.
# Dense: Adds a layer of neurons.

# Each layer of neurons needs an activation function to decide if a neuron should be activated or not. 
# Relu effectively means if X>0 return X, else return 0. It passes values 0 or greater to the next layer in the network.
# Softmax takes a set of values, and effectively picks the biggest one so you don't have to sort to find the largest value. 
# For example, if the output of the last layer looks like [0.1, 0.1, 0.05, 0.1, 9.5, 0.1, 0.05, 0.05, 0.05], it returns [0,0,0,0,1,0,0,0,0].
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(64, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# Compile the model
# - Optimizer: an algorithm that modifies the attributes of the neural network like weights and learning rate. This helps in reducing the loss and improving accuracy.
# - Loss indicates the model's performance by a number. If the model is performing better, loss will be a smaller number. Otherwise loss will be a larger number.
# - Notice the metrics= parameter. This allows TensorFlow to report on the accuracy of the training after each epoch by checking the predicted results against the known answers(labels). 
# It basically reports back on how effectively the training is progressing.
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

model.fit(ds_train, epochs=5)


# Evaluate the test set:
cloud_logger.info(model.evaluate(ds_test))

# ---------------------------------------------------
# Save the model to it can resume where it left off and avoid long training times
# Save the entire model as a SavedModel.
model.save('saved_model')

# Reload a fresh Keras model from the saved model
new_model = tf.keras.models.load_model('saved_model')

# Summary of loaded SavedModel
new_model.summary()


# Save the entire model to a HDF5 file.
model.save('my_model.h5')

# Recreate the exact same model, including its weights and the optimizer
new_model_h5 = tf.keras.models.load_model('my_model.h5')

# Summary of loaded h5 model
new_model_h5.summary()

# ---------------------------------------------------
# Callbacks
# A callback is a powerful tool to customize the behavior of a Keras model during training, evaluation, or inference.
# You can define a callback to stop training as soon as your model reaches a desired accuracy on the training set.


# Print out max value to see the changes
image_batch, labels_batch = next(iter(ds_train))
t_image_batch, t_labels_batch = next(iter(ds_test))
up_logger.info("training images max " + str(np.max(image_batch[0])))
up_logger.info("test images max " + str(np.max(t_image_batch[0])))