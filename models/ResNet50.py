
#import libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


import tensorflow as tf 
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout

#define parameters

EPOCHS = 50
BS = 128
DROPOUT_RATE = 0.5
FROZEN_LAYER_NUM = 170

ADAM_LEARNING_RATE = 0.001
SGD_LEARNING_RATE = 0.01
SGD_DECAY = 0.0001

from sklearn.metrics import *

import skimage
from skimage.transform import rescale, resize


import pydot

resnet50=tf.keras.applications.resnet50.ResNet50(include_top=False,input_shape=(Resize_pixelsize, Resize_pixelsize, 3),pooling='avg')
last_layer=resnet50.get_layer('avg_pool').output
x=Flatten(name='flatten')(last_layer)
x=Dropout(DROPOUT_RATE)(x)
x=Dense(4096, activation='relu',name='fc6')(x)
x=Dropout(DROPOUT_RATE)(x)
x = Dense(1024, activation='relu', name='fc7')(x)
x = Dropout(DROPOUT_RATE)(x)


#freezing layers
batch_norm_indices = [2, 6, 9, 13, 14, 18, 21, 24, 28, 31, 34, 38, 41, 45, 46, 53, 56, 60, 63, 66, 70, 73, 76, 80, 83, 87, 88, 92, 95, 98, 102, 105, 108, 112, 115, 118, 122, 125, 128, 132, 135, 138, 142, 145, 149, 150, 154, 157, 160, 164, 167, 170]
for i in range(FROZEN_LAYER_NUM):
    if i not in batch_norm_indices:
        resnet50.layers[i].trainable = False

#defining the output layer       
out=Dense(7,activation='softmax',name='classifier')(x)

#defining the final model
model=Model(resnet50.input,out)

#optimizer
optim=tf.keras.optimizers.Adam(learning_rate=ADAM_LEARNING_RATE,beta_1=0.9,beta_2=0.999,epsilon=1e-08)
sgd = tf.keras.optimizers.SGD(learning_rate=SGD_LEARNING_RATE, momentum=0.9, nesterov=True)
rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc',mode='max',factor=0.5, patience=10, min_lr=0.00001, verbose=1)

#compiling
model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])

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

history = model.fit_generator(
    generator = train_generator,
    validation_data=dev_generator,
    steps_per_epoch=28709// BS,
    validation_steps=3509 // BS,
    shuffle=True,
    epochs=100,
    callbacks=[rlrop],
    use_multiprocessing=True,
)

print('\n# Evaluate on dev data')
results_dev = model.evaluate_generator(dev_generator, 3509 // BS)
print('dev loss, dev acc:', results_dev)

print('\n# Evaluate on test data')
results_test = model.evaluate_generator(test_generator, 3509 // BS)
print('test loss, test acc:', results_test)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'dev'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'dev'], loc='upper left')
plt.show()

epoch_str = '-EPOCHS_' + str(EPOCHS)
test_acc = 'test_acc_%.3f' % results_test[1]
model.save('/content/drive/My Drive/cs230 project/models/' + 'RESNET50' + epoch_str + test_acc + '.h5')