
#IMPORTS ###################
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


import tensorflow as tf 
from tensorflow.python.lib.io import file_io
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, Callback
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import *

import skimage
from skimage.transform import rescale, resize


import pydot

#PARAMETERS ###########################################

EPOCHS = 50
BS = 128
DROPOUT_RATE = 0.5
FROZEN_LAYER_NUM = 170

ADAM_LEARNING_RATE = 0.001
SGD_LEARNING_RATE = 0.01
SGD_DECAY = 0.0001

img_height, img_width= 96,96


#DATASET ############################################

# Folder where logs and models are saved
folder = 'C:/Users/Maryem/Desktop/P2M/logs/RESNET50'

# Data paths
train_dataset	= 'C:/Users/Maryem/Desktop/fer2013/train.csv'
eval_dataset 	= 'C:/Users/Maryem/Desktop/fer2013/fer2013_eval.csv'


#MODEL ##########################################

#create the base pre-trained resnet50 model
    #include_top: include the top fully connected layer --> set to false
    #weights: pre-training on imagenet
    #input_shape: the shape of the input image
    
resnet50=tf.keras.applications.resnet50.ResNet50(include_top=False,weights='imagenet',input_shape=(img_height, img_width, 3),pooling='avg')
#we take last_layer the output of our pre-trained model

#freezing layers
batch_norm_indices = [2, 6, 9, 13, 14, 18, 21, 24, 28, 31, 34, 38, 41, 45, 46, 53, 56, 60, 63, 66, 70, 73, 76, 80, 83, 87, 88, 92, 95, 98, 102, 105, 108, 112, 115, 118, 122, 125, 128, 132, 135, 138, 142, 145, 149, 150, 154, 157, 160, 164, 167, 170]
for i in range(FROZEN_LAYER_NUM):
    if i not in batch_norm_indices:
        resnet50.layers[i].trainable = False
        

last_layer=resnet50.get_layer('avg_pool').output

#add a flatten layer
x=Flatten(name='flatten')(last_layer)

#add a dropout layer 
x=Dropout(DROPOUT_RATE)(x)

#add a fully connected layer with tha activation function relu
x=Dense(4096, activation='relu',name='fc6')(x)
x=Dropout(DROPOUT_RATE)(x)
x = Dense(1024, activation='relu', name='fc7')(x)
x = Dropout(DROPOUT_RATE)(x)


#defining the output layer 
    #activation function: softmax
    #7 units for the 7 classes
out=Dense(7,activation='softmax',name='classifier')(x)

#defining the final model to be trained
model=Model(resnet50.input,out)

#optimizer
optim=tf.keras.optimizers.Adam(learning_rate=ADAM_LEARNING_RATE,beta_1=0.9,beta_2=0.999,epsilon=1e-08)
sgd = tf.keras.optimizers.SGD(learning_rate=SGD_LEARNING_RATE, momentum=0.9, nesterov=True)
rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc',mode='max',factor=0.5, patience=10, min_lr=0.00001, verbose=1)

#compiling
model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])

# DATA PREPARATION ########################

# #create an ImageDataGenerator and generate batches of data
# def get_datagen(dataset, aug=False):
#     if aug: #check if augmentation is enabled
#         datagen = ImageDataGenerator(
#                             rescale=1./255, #rescale the pixel values to [0,1]
#                             featurewise_center=False,
#                             featurewise_std_normalization=False,
#                             rotation_range=10,  # Randomly rotate images by up to 10 degrees
#                             width_shift_range=0.1, # Randomly shift images horizontally by up to 10% of the width
#                             height_shift_range=0.1, # Randomly shift images vertically by up to 10% of the height
#                             zoom_range=0.1, # Randomly zoom into images by up to 10%
#                             horizontal_flip=True) #randomly flip images horizontally
#     else:
#         datagen = ImageDataGenerator(rescale=1./255)

#     return datagen.flow_from_directory(
#             dataset,
#             target_size=(197, 197), 
#             color_mode='rgb',
#             shuffle = True, 
#             class_mode='categorical',
#             batch_size=BS) #size of the batch 
# DATA PREPARATION ####################################################################################################################################

# Preprocesses a numpy array encoding a batch of images
    # x: Input array to preprocess
def preprocess_input(x):
    x -= 128.8006 # np.mean(train_dataset)
    return x

# Function that reads the data from the csv file, increases the size of the images and returns the images and their labels
    # dataset: Data path
def get_data(dataset):
    file_stream = file_io.FileIO(dataset, mode='r')
    data = pd.read_csv(file_stream)
    pixels = data['pixels'].tolist()
    images = np.empty((len(data), img_height, img_width, 3))
    i = 0

    for pixel_sequence in pixels:
        single_image = [float(pixel) for pixel in pixel_sequence.split(' ')]  # Extraction of each single
        single_image = np.asarray(single_image).reshape(48, 48) # Dimension: 48x48
        single_image = resize(single_image, (img_height, img_width), order = 3, mode = 'constant') # Dimension: 139x139x3 (Bicubic)
        ret = np.empty((img_height, img_width, 3))  
        ret[:, :, 0] = single_image
        ret[:, :, 1] = single_image
        ret[:, :, 2] = single_image
        images[i, :, :, :] = ret
        i += 1
    
    images = preprocess_input(images)
    labels = to_categorical(data['emotion'])

    return images, labels    

# Data preparation
train_data_x, train_data_y  = get_data(train_dataset)
val_data  = get_data(eval_dataset)

# Generate batches of tensor image data with real-time data augmentation. The data will be looped over (in batches) indefinitely
# rescale:          Rescaling factor (defaults to None). Multiply the data by the value provided (before applying any other transformation)
# rotation_range:   Int. Degree range for random rotations
# shear_range:      Float. Shear Intensity (Shear angle in counter-clockwise direction as radians)
# zoom_range:       Float or [lower, upper]. Range for random zoom. If a float, [lower, upper] = [1-zoom_range, 1+zoom_range]
# fill_mode :       Points outside the boundaries of the input are filled according to the given mode: {"constant", "nearest", "reflect" or "wrap"}
# horizontal_flip:  Boolean. Randomly flip inputs horizontally
train_datagen = ImageDataGenerator(
    rotation_range  = 10,
    shear_range     = 10, # 10 degrees
    zoom_range      = 0.1,
    fill_mode       = 'reflect',
    horizontal_flip = True)

# Takes numpy data & label arrays, and generates batches of augmented/normalized data. Yields batcfillhes indefinitely, in an infinite loop
    # x:            Data. Should have rank 4. In case of grayscale data, the channels axis should have value 1, and in case of RGB data, 
    #               it should have value 3
    # y:            Labels
    # batch_size:   Int (default: 32)
train_generator = train_datagen.flow(
    train_data_x,
    train_data_y,
    batch_size  = BS)


# TRAINING ###############################

tensorboard= TensorBoard(
    log_dir=folder+'/logs',
    histogram_freq=0,
    write_graph= True,
    write_images= True)

reduce_lr= ReduceLROnPlateau(
    monitor = 'val_loss',
    factor= 0.4,
    patience = 5,
    mode = 'auto',
    min_lr= 1e-6)

early_stop=EarlyStopping(
    monitor = 'val_loss',
    patience = 20,
    mode = 'auto')

import warnings
class ModelCheckpoint(Callback):

    def __init__(self, filepath, folder, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1):
        super(# It seems like there is a typo in the code where `ModelChec` is mentioned. It should be
        # `ModelCheckpoint`, which is a callback function in Keras that is used to save the
        # model after every epoch during training.
        ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.folder = folder
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, ' 'fallback to auto mode.' % (mode), RuntimeWarning)
            mode = 'auto'
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, ' 'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,' ' saving model to %s' % (epoch + 1, self.monitor, self.best, current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                            # Save model.h5 on to google storage
                            with file_io.FileIO(filepath, mode='rb') as input_f:
                                with file_io.FileIO(self.folder + '/checkpoints/' + filepath, mode='wb') as output_f:  # w+ : writing and reading
                                    output_f.write(input_f.read())
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve' %(epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))


# Save the model after every epoch
	# filepath:       String, path to save the model file
	# monitor:        Quantity to monitor {val_loss, val_acc}
	# save_best_only: If save_best_only=True, the latest best model according to the quantity monitored will not be overwritten
	# mode:           One of {auto, min, max}. If save_best_only = True, the decision to overwrite the current save file is made based on either
	#			      the maximization or the minimization of the monitored quantity. For val_acc, this should be max, for val_loss this should
	#			      be min, etc. In auto mode, the direction is automatically inferred from the name of the monitored quantity
	# period:         Interval (number of epochs) between checkpoints
check_point = ModelCheckpoint(
	filepath		= 'ResNet50_{epoch:02d}_{val_loss:.2f}.keras',
	folder 			= folder,
	monitor 		= 'val_loss', # Accuracy is not always a good indicator because of its yes or no nature
	save_best_only	= True,
	mode 			= 'auto',
	period			= 1)

model.fit(
    x = train_generator,
    validation_data=val_data,
    steps_per_epoch=len(train_data_x) // BS,
    shuffle=True,
    epochs=100,
    callbacks=[tensorboard,reduce_lr,early_stop,check_point],
)

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

#SAVING THE TRAINED MODEL #################3


# epoch_str = '-EPOCHS_' + str(EPOCHS)
# test_acc = 'test_acc_%.3f' % results_test[1]
model.save(folder+'RESNET50.keras')
with file_io.FileIO('Inception-v3.keras', mode='rb') as input_f:
    with file_io.FileIO(folder + '/Inception-v3.keras', mode='wb') as output_f:  # w+ : writing and reading
        output_f.write(input_f.read())