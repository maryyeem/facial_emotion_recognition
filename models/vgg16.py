
#IMPORTS ###############################

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.python.lib.io import file_io

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Flatten, Dense, Activation, Conv2D, MaxPool2D, BatchNormalization, Dropout, MaxPooling2D

from sklearn.metrics import *

import skimage
from skimage.transform import rescale, resize

import pydot

#PARAMETERS ##########################

EPOCHS = 50
BS = 128
DROPOUT_RATE = 0.5
FROZEN_LAYER_NUM = 19

ADAM_LEARNING_RATE = 0.001
SGD_LEARNING_RATE = 0.01
SGD_DECAY = 0.0001

Resize_pixelsize = 197


#MODEL ##############################""
#create the base pre-trained resnet50 model
    #include_top: include the top fully connected layer --> set to false
    #weights: pre-training on imagenet
    #input_shape: the shape of the input image

vgg16 = tf.keras.applications.VGG16(include_top=False,weights='imagenet',input_shape=(Resize_pixelsize, Resize_pixelsize, 3), pooling='avg')
#we take last_layer the output of our pre-trained model

last_layer = vgg16.get_layer('pool5').output
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
optim = tf.keras.optimizers.Adam(lr=ADAM_LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#optim = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
sgd = tf.keras.optimizers.SGD(lr=SGD_LEARNING_RATE, momentum=0.9, decay=SGD_DECAY, nesterov=True)
rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc',mode='max',factor=0.5, patience=10, min_lr=0.00001, verbose=1)

#compiling
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# DATA PREPARATION ########################

#create an ImageDataGenerator to generate batches of data
def get_datagen(dataset, aug=False):
    if aug:
        datagen = ImageDataGenerator(
                            rescale=1./255,
                            featurewise_center=False,
                            featurewise_std_normalization=False,
                            rotation_range=10,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.1,
                            horizontal_flip=True)
    else:
        datagen = ImageDataGenerator(rescale=1./255)

    return datagen.flow_from_directory(
            dataset,
            target_size=(197, 197),
            color_mode='rgb',
            shuffle = True,
            class_mode='categorical',
            batch_size=BS)
    
train_generator  = get_datagen('/content/train', True)
dev_generator    = get_datagen('/content/dev')
test_generator  = get_datagen('/content/test')

from sklearn.utils import class_weight
file_stream = file_io.FileIO('/content/drive/My Drive/cs230 project/collab/fer2013/dev.csv', mode='r')
data = pd.read_csv(file_stream)
data[' pixels'] = data[' pixels'].apply(lambda x: [int(pixel) for pixel in x.split()])
X, Y = data[' pixels'].tolist(), data['emotion'].values
class_weights = class_weight.compute_class_weight('balanced',
                                                    np.unique(Y),
                                                    Y)

#TRAINING #############################

history = model.fit(
    x = train_generator,
    validation_data=dev_generator, 
    steps_per_epoch=28709// BS,
    validation_steps=3509 // BS,
    shuffle=True,
    epochs=100,
    callbacks=[rlrop],
    use_multiprocessing=True,
) 

# print('\n# Evaluate on dev data')
results_dev = model.evaluate_generator(dev_generator, 3509 // BS)
# print('dev loss, dev acc:', results_dev)

# print('\n# Evaluate on test data')
results_test = model.evaluate_generator(test_generator, 3509 // BS)
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

epoch_str = '-EPOCHS_' + str(EPOCHS)
test_acc = 'test_acc_%.3f' % results_test[1]
model.save('VGG16' + epoch_str + test_acc + '.keras')