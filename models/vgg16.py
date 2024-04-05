
#IMPORTS ###############################

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.python.lib.io import file_io
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping,TensorBoard,ReduceLROnPlateau, Callback
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Flatten, Dense, Activation, Conv2D, MaxPool2D, BatchNormalization, Dropout, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import *

import skimage
from skimage.transform import rescale, resize

import pydot

#PARAMETERS ###############################

EPOCHS = 50
BS = 128
DROPOUT_RATE = 0.5
FROZEN_LAYER_NUM = 19

ADAM_LEARNING_RATE = 0.001
SGD_LEARNING_RATE = 0.01
SGD_DECAY = 0.0001

img_width, img_height = 197,197

#DATASET ##########################

#folder where we will save the logs and the model
folder='C:/Users/Maryem/Desktop/P2M/logs/VGG16'

train_dataset='C:/Users/Maryem/Desktop/fer2013/train.csv'
eval_dataset 	= 'C:/Users/Maryem/Desktop/fer2013/fer2013_eval.csv'


#MODEL #####################################

#create the base pre-trained resnet50 model
    #include_top: include the top fully connected layer --> set to false
    #weights: pre-training on imagenet
    #input_shape: the shape of the input image

vgg16 = tf.keras.applications.VGG16(include_top=False,weights='imagenet',input_shape=(img_width, img_height, 3), pooling='avg')
#we take last_layer the output of our pre-trained model
last_layer = vgg16.get_layer('global_average_pooling2d').output
#add a flatten layer
x = Flatten(name='flatten')(last_layer)
#add a dropout layer 

x = Dropout(DROPOUT_RATE)(x)

#add a fully connected layer with the activation function relu
x = Dense(4096, activation='relu', name='fc6')(x)
x = Dropout(DROPOUT_RATE)(x)
x = Dense(1024, activation='relu', name='fc7')(x)


#freezing layers
for i in range(FROZEN_LAYER_NUM):
    vgg16.layers[i].trainable = False

#defining the output layer
    #activation function: softmax
    #7 units for the 7 classes
out = Dense(7, activation='softmax', name='classifier')(x)

#defining the final model to be trained
model = Model(vgg16.input, out)

#defining the optimizer
optim = tf.keras.optimizers.Adam(learning_rate=ADAM_LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#optim = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
sgd = tf.keras.optimizers.SGD(learning_rate=SGD_LEARNING_RATE, momentum=0.9, nesterov=True)
rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc',mode='max',factor=0.5, patience=10, min_lr=0.00001, verbose=1)

#compiling
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# DATA PREPARATION ###############################

#create an ImageDataGenerator to generate batches of data
# def get_datagen(dataset, aug=False):
#     if aug:
#         datagen = ImageDataGenerator(
#                             rescale=1./255,
#                             featurewise_center=False,
#                             featurewise_std_normalization=False,
#                             rotation_range=10,
#                             width_shift_range=0.1,
#                             height_shift_range=0.1,
#                             zoom_range=0.1,
#                             horizontal_flip=True)
#     else:
#         datagen = ImageDataGenerator(rescale=1./255)

#     return datagen.flow_from_directory(
#             dataset,
#             target_size=(197, 197),
#             color_mode='rgb',
#             shuffle = True,
#             class_mode='categorical',
#             batch_size=BS)
    
# train_generator  = get_datagen('C:/Users/Maryem/Desktop/fer2013/train', True)
# dev_generator    = get_datagen('C:/Users/Maryem/Desktop/fer2013/fer2013_eval.csv')
# test_generator  = get_datagen('C:/Users/Maryem/Desktop/fer2013/test')
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

# from sklearn.utils import class_weight
# file_stream = file_io.FileIO('/content/drive/My Drive/cs230 project/collab/fer2013/dev.csv', mode='r')
# data = pd.read_csv(file_stream)
# data[' pixels'] = data[' pixels'].apply(lambda x: [int(pixel) for pixel in x.split()])
# X, Y = data[' pixels'].tolist(), data['emotion'].values
# class_weights = class_weight.compute_class_weight('balanced',
#                                                     np.unique(Y),
#                   Y)

#TRAINING #######################################

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


model.fit(
    x = train_generator,
    validation_data=val_data, 
    steps_per_epoch=len(train_data_x)// BS,
    shuffle=True,
    epochs=EPOCHS,
    callbacks=[tensorboard,reduce_lr,early_stop],
) 

# print('\n# Evaluate on dev data')
# results_dev = mode.evaluate_generator(dev_generator, 3509 // BS)
# # print('dev loss, dev acc:', results_dev)

# # print('\n# Evaluate on test data')
# results_test = model.evaluate_generator(test_generator, 3509 // BS)l
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

# epoch_str = '-EPOCHS_' + str(EPOCHS)
# test_acc = 'test_acc_%.3f' % results_test[1]

model.save('VGG16.keras')

with file_io.FileIO('VGG16.keras', mode='rb') as input_f:
    with file_io.FileIO(folder + '/VGG16.keras', mode='wb') as output_f:  # w+ : writing and reading
        output_f.write(input_f.read())