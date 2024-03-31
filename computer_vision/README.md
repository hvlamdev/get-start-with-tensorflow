## Check if TensorFlow package is present. Run the following command in the terminal.
python -c "import tensorflow;print(tensorflow.__version__)"

## Install additional packages
### Upgrade the pip3
pip3 install --upgrade pip


### Install google-cloud-logging
/usr/bin/python3 -m pip install -U google-cloud-logging --user

### Install pylint
/usr/bin/python3 -m pip install -U pylint --user

### Upgrade tensorflow version
pip install --upgrade tensorflow


## Run model
python model.py

## Run callback
python callback_model.py

## Experiment with different values for the dense layer.
Go to # Define the model section, change 64 to 128 neurons:

## Run updated_model
python updated_model.py

## Consider the effects of additional layers in the network. What will happen if you add another layer between the two dense layers?
### In updated_model.py, add a layer in the # Define the model section.
### Replace your model definition with the following:
```
# Define the model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(64, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(64, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
```

### ## Run updated_model
python updated_model.py

## Before you trained your model, you normalized the pixel values to the range of [0, 1]. What would be the impact of removing normalization so that the values are in the range of [0, 255], like they were originally in the dataset? 
### Give it a try -- in the # Normalize the data section, remove the map function applied to both the training and test datasets.
```
# Define batch size
BATCH_SIZE = 32


# Normalizing and batch processing of data
ds_train = ds_train.batch(BATCH_SIZE)
ds_test = ds_test.batch(BATCH_SIZE)

# Define the model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(64, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
```

# Add this code at the end of updated_model.py to print the max value of the first image in batch 0. Without normalization, the max value will be in the range of [0, 255].
```
# Print out max value to see the changes
image_batch, labels_batch = next(iter(ds_train))
t_image_batch, t_labels_batch = next(iter(ds_test))
up_logger.info("training images max " + str(np.max(image_batch[0])))
up_logger.info("test images max " + str(np.max(t_image_batch[0])))
```


# What happens if you remove the Flatten() layer, and why?
# In the # Define the model section, remove "tf.keras.layers.Flatten()":
```
# Define the model
model = tf.keras.models.Sequential([tf.keras.layers.Dense(64, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
```

You get an error about the shape of the data. This is expected.

The details of the error may seem vague right now, but it reinforces the rule of thumb that the first layer in your network should be the same shape as your data. Right now, the input images are of shape 28x28, and 28 layers of 28 neurons would be infeasible. So, it makes more sense to flatten that 28,28 into a 784x1.

Instead of writing all the code to handle that yourselves, you can add the Flatten() layer at the beginning. When the arrays are loaded into the model later, they'll automatically be flattened for you.

# Notice the final (output) layer. Why are there 10 neurons in the final layer? What happens if you have a different number than 10?
Find out by training the network with 5.
Replace the # Define the model section with the following to undo the change you made in the previous section:
```
# Define the model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(64, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
```
Change the number of neurons in the last layer from 10 to 5:
```
# Define the model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(64, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(5, activation=tf.nn.softmax)])
```
Save and run updated_model.py.
=> What happens: You get an error as soon as it finds an unexpected value.
Another rule of thumb -- the number of neurons in the last layer should match the number of classes you are classifying for. In this case, it's the digits 0-9, so there are 10 of them, and hence you should have 10 neurons in your final layer.