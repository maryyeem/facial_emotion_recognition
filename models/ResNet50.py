
#IMPORTS ###################


import numpy as np
import pandas as pd
from tensorflow.python.lib.io import file_io
from skimage.transform import resize
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping, Callback, LearningRateScheduler
from sklearn.metrics import *

import skimage
from skimage.transform import rescale, resize


import pydot

# PARAMETERS ##########################################################################################################################################


# Size of the images
img_height, img_width = 197, 197

# Parameters
num_classes         = 7     # ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
epochs_top_layers   = 5
epochs_all_layers   = 100
batch_size          = 128


#DATASET ############################################

# Folder where logs and models are saved
folder = 'C:/Users/Maryem/Desktop/P2M/logs/RESNET50'

# Data paths
train_dataset	= 'C:/Users/Maryem/Desktop/fer2013/train'
eval_dataset 	= 'C:/Users/Maryem/Desktop/fer2013/validation'


#MODEL ##########################################

# Create the based on ResNet-50 architecture pre-trained model
    # model:        Selects one of the available architectures vgg16, resnet50 or senet50
    # include_top:  Whether to include the fully-connected layer at the top of the network
    # weights:      Pre-training on VGGFace
    # input_shape:  Optional shape tuple, only to be specified if include_top is False (otherwise the input
    #               shape has to be (224, 224, 3) (with 'channels_last' data format) or (3, 224, 224) (with
    #               'channels_first' data format). It should have exactly 3 inputs channels, and width and
    #               height should be no smaller than 197. E.g. (200, 200, 3) would be one valid value.
# Returns a keras Model instance
base_model = ResNet50(
    include_top = False,
    weights     = 'imagenet',
    input_shape = (img_height, img_width, 3))

# Places x as the output of the pre-trained model
x = base_model.output

# Flattens the input. Does not affect the batch size
x = Flatten()(x)

# Add a fully-connected layer and a logistic layer
# Dense implements the operation: output = activation(dot(input, kernel) + bias(only applicable if use_bias is True))
    # units:        Positive integer, dimensionality of the output space
    # activation:   Activation function to use
    # input shape:  nD tensor with shape: (batch_size, ..., input_dim)
    # output shape: nD tensor with shape: (batch_size, ..., units)
x = Dense(1024, activation = 'relu')(x)
predictions = Dense(num_classes, activation = 'softmax')(x)

# The model we will train
model = Model(inputs = base_model.input, outputs = predictions)
# model.summary()

# DATA PREPARATION ########################
def custom_preprocessing(x):
    x_reshaped = x.reshape(48,48)
    return x_reshaped

#create an ImageDataGenerator and generate batches of data
def get_datagen(dataset, aug=False):
    if aug: #check if augmentation is enabled
        datagen = ImageDataGenerator(
                            rescale=1./255, #rescale the pixel values to [0,1]
                            featurewise_center=False,
                            featurewise_std_normalization=False,
                            rotation_range=10,  # Randomly rotate images by up to 10 degrees
                            width_shift_range=0.1, # Randomly shift images horizontally by up to 10% of the width
                            height_shift_range=0.1, # Randomly shift images vertically by up to 10% of the height
                            zoom_range=0.1, # Randomly zoom into images by up to 10%
                            horizontal_flip=True) #randomly flip images horizontally
    else:
        datagen = ImageDataGenerator(rescale=1./255)

    return datagen.flow_from_directory(
            dataset,
            target_size=(197, 197), 
            color_mode='rgb',
            shuffle = True, 
            class_mode='categorical',
            batch_size=batch_size) #size of the batch 

train_generator  = get_datagen(train_dataset, True)
dev_generator    = get_datagen(eval_dataset)


# TRAINING ###############################

# UPPER LAYERS TRAINING ###############################################################################################################################

# First: train only the top layers (which were randomly initialized) freezing all convolutional ResNet-50 layers
for layer in base_model.layers:
    layer.trainable = False

# Compile (configures the model for training) the model (should be done *AFTER* setting layers to non-trainable)
    # optimizer:    String (name of optimizer) or optimizer object
        # lr:       Float >= 0. Learning rate
        # beta_1:   Float, 0 < beta < 1. Generally close to 1
        # beta_2:   Float, 0 < beta < 1. Generally close to 1
        # epsilon:  Float >= 0. Fuzz factor
        # decay:    Float >= 0. Learning rate decay over each update
    # loss:     String (name of objective function) or objective function
    # metrics:  List of metrics to be evaluated by the model during training and testing
model.compile(
    optimizer   = Adam(learning_rate = 1e-3, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08, decay = 0.0), 
    loss        = 'categorical_crossentropy', 
    metrics     = ['accuracy'])

# This callback writes a log for TensorBoard, which allows you to visualize dynamic graphs of your training and test metrics, 
# as well as activation histograms for the different layers in your model
    # log_dir:          The path of the directory where to save the log files to be parsed by TensorBoard
    # histogram_freq:   Frequency (in epochs) at which to compute activation and weight histograms for the layers of the model
    #                   If set to 0, histograms won't be computed. Validation data (or split) must be specified for histogram visualizations
    # write_graph:      Whether to visualize the graph in TensorBoard. The log file can become quite large when write_graph is set to True
    # write_grads:      Whether to visualize gradient histograms in TensorBoard. histogram_freq must be greater than 0
    # write_images:     Whether to write model weights to visualize as image in TensorBoard
# To visualize the files created during training, run in your terminal: tensorboard --logdir path_to_current_dir/Graph
tensorboard_top_layers = TensorBoard(
	log_dir         = folder + '/logs_top_layers',
	histogram_freq  = 0,
	write_graph     = True,
	write_images    = True)

# Train the model on the new data for a few epochs (Fits the model on data yielded batch-by-batch by a Python generator)
    # generator:        A generator or an instance of Sequence (keras.utils.Sequence) object in order to avoid duplicate data when using multiprocessing
    #                   The output of the generator must be either {a tuple (inputs, targets)} {a tuple (inputs, targets, sample_weights)}
    # steps_per_epoch:  Total number of steps (batches of samples) to yield from generator before declaring one epoch finished and starting the next epoch
    #                   It should typically be equal to the number of unique samples of your dataset divided by the batch size 
    # epochs:           Integer, total number of iterations on the data
    # validation_data:  This can be either {a generator for the validation data } {a tuple (inputs, targets)} {a tuple (inputs, targets, sample_weights)}
    # callbacks:        List of callbacks to be called during training (to visualize the files created during training, run in your terminal:
    #                   tensorboard --logdir path_to_current_dir/Graph)
model.fit(
    x           = train_generator,
    steps_per_epoch     = len(train_generator),  # samples_per_epoch / batch_size
    epochs              = epochs_top_layers,                            
    validation_data     = dev_generator,
    callbacks           = [tensorboard_top_layers])


# FULL NETWORK TRAINING ###############################################################################################################################

# At this point, the top layers are well trained and we can start fine-tuning convolutional layers from ResNet-50

# Fine-tuning of all the layers
for layer in model.layers:
    layer.trainable = True

# We need to recompile the model for these modifications to take effect (we use SGD with nesterov momentum and a low learning rate)
    # optimizer:    String (name of optimizer) or optimizer object
        # lr:       float >= 0. Learning rate
        # momentum: float >= 0. Parameter updates momentum
        # decay:    float >= 0. Learning rate decay over each update
        # nesterov: boolean. Whether to apply Nesterov momentum
    # loss:     String (name of objective function) or objective function
    # metrics:  List of metrics to be evaluated by the model during training and testing
model.compile(
    optimizer   = SGD(learrning_rate = 1e-4, momentum = 0.9, decay = 0.0, nesterov = True),
    loss        = 'categorical_crossentropy', 
    metrics     = ['accuracy'])

# This callback writes a log for TensorBoard, which allows you to visualize dynamic graphs of your training and test metrics, 
tensorboard_all_layers = TensorBoard(
    log_dir         = folder + '/logs_all_layers',
    histogram_freq  = 0,
    write_graph     = True,
    write_grads     = False,
    write_images    = True)

# Reduce learning rate when a metric has stopped improving
	# monitor: 	Quantity to be monitored
	# factor: 	Factor by which the learning rate will be reduced. new_lr = lr * factor
	# patience:	Number of epochs with no improvement after which learning rate will be reduced
	# mode: 	One of {auto, min, max}
	# min_lr:	Lower bound on the learning rate
reduce_lr_plateau = ReduceLROnPlateau(
	monitor 	= 'val_acc',
	factor		= 0.5,
	patience	= 3,
	mode 		= 'auto',
	min_lr		= 1e-8)

# Stop training when a monitored quantity has stopped improving
	# monitor:		Quantity to be monitored
	# patience:		Number of epochs with no improvement after which training will be stopped
	# mode: 		One of {auto, min, max}
early_stop = EarlyStopping(
	monitor 	= 'val_acc',
	patience 	= 10,
	mode 		= 'auto')
	

# Save the model after every epoch
	# filepath:       String, path to save the model file
	# monitor:        Quantity to monitor {val_loss, val_acc}
	# save_best_only: If save_best_only=True, the latest best model according to the quantity monitored will not be overwritten
	# mode:           One of {auto, min, max}. If save_best_only = True, the decision to overwrite the current save file is made based on either
	#			      the maximization or the minimization of the monitored quantity. For val_acc, this should be max, for val_loss this should
	#			      be min, etc. In auto mode, the direction is automatically inferred from the name of the monitored quantity
	# period:         Interval (number of epochs) between checkpoints

# We train our model again (this time fine-tuning all the resnet blocks)
model.fit(
    x           = train_generator,
    steps_per_epoch     = len(train_generator),  # samples_per_epoch / batch_size 
    epochs              = epochs_all_layers,                        
    validation_data     = dev_generator,
    callbacks           = [reduce_lr_plateau,early_stop,tensorboard_all_layers])

# SAVING ##############################################################################################################################################

# Saving the model in the workspace
model.save(folder + '/ResNet-50.keras')
# Save model.h5 on to google storage
with file_io.FileIO('ResNet-50.keras', mode='rb') as input_f:
    with file_io.FileIO(folder + '/ResNet-50.keras', mode='wb') as output_f:  # w+ : writing and reading
        output_f.write(input_f.read())


# print('\n# Evaluate on dev data')
# results_dev = model.evaluate_generator(dev_generator, 3509 // BS)
# print('dev loss, dev acc:', results_dev)

# print('\n# Evaluate on test data')
# results_test = model.evaluate_generator(test_generator, 3509 // BS)
# print('test loss, test acc:', results_test)

# # list all data in history
# print(history.history.keys())
# # summarize history for accuracy
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'dev'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'dev'], loc='upper left')
# plt.show()
