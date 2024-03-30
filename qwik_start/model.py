# Mechine Learning with TensorFlow
# The progress use Answer and Data to create a rule to predict the result


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


# Prepare the data (y = 3x + 1)
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

# Design a model 
# we will create the simplest possible neural network.
# It has 1 layer, and that layer has 1 neuron.
# The neural network's input is only one value at a time. Hence, the input shape must be [1].
model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])

# Compile the model
# Must specify 2 functions, a loss and an optimizer.
# Loss function measures the guessed answers against the known correct answers and measures how well or how badly it did.
# E.g. When the computer is trying to learn this relationship, it makes a guess...maybe y=10x+10

# Optimizer function makes another guess. Based on how the loss function went, it will try to minimize the loss.
# At  At this point, maybe it will come up with something like y=5x+5. While this is still pretty bad, it's closer to the correct result (i.e. the loss is lower).

# The model repeats this for the number of epochs you specify.
model.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.MeanSquaredError())

# Train the neural network
# To train the neural network to 'learn' the relationship between the Xs and Ys, you will use model.fit.
# This function will train the model in a loop where it will make a guess, measure how good or bad it is 
# (aka the loss), use the optimizer to make another guess, etc. 
# It will repeat this process for the number of epochs you specify, which in this lab is 500.
model.fit(xs, ys, epochs=500)


# Use the model
# E.g. The result is 31.00664
# Neural networks deal with probabilities. It calculated that there is a very high probability that the relationship between X and Y is Y=3X+1. 
# But with only 6 data points it can't know for sure. As a result, the result for 10 is very close to 31, but not necessarily 31.
cloud_logger.info(str(model.predict([10.0])))