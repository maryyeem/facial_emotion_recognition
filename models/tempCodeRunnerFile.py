EPOCHS = 50
# BS = 128
# DROPOUT_RATE = 0.5
# FROZEN_LAYER_NUM = 170

# ADAM_LEARNING_RATE = 0.001
# SGD_LEARNING_RATE = 0.01
# SGD_DECAY = 0.0001

# Resize_pixelsize = 197

# vgg_notop = VGGFace(model='resnet50', include_top=False, input_shape=(Resize_pixelsize, Resize_pixelsize, 3), pooling='avg')
# last_layer = vgg_notop.get_layer('avg_pool').output
# x = Flatten(name='flatten')(last_layer)
# x = Dropout(DROPOUT_RATE)(x)
# x = Dense(4096, activation='relu', name='fc6')(x)
# x = Dropout(DROPOUT_RATE)(x)
# x = Dense(1024, activation='relu', name='fc7')(x)
# x = Dropout(DROPOUT_RATE)(x)

# batch_norm_indices = [2, 6, 9, 13, 14, 18, 21, 24, 28, 31, 34, 38, 41, 45, 46, 53, 56, 60, 63, 66, 70, 73, 76, 80, 83, 87, 88, 92, 95, 98, 102, 105, 108, 112, 115, 118, 122, 125, 128, 132, 135, 138, 142, 145, 149, 150, 154, 157, 160, 164, 167, 170]
# for i in range(FROZEN_LAYER_NUM):
#     if i not in batch_norm_indices:
#         vgg_notop.layers[i].trainable = False

# out = Dense(7, activation='softmax', name='classifier')(x)

# model = Model(vgg_notop.input, out)

# optim = tf.keras.optimizers.Adam(lr=ADAM_LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# sgd = tf.keras.optimizers.SGD(lr=SGD_LEARNING_RATE, momentum=0.9, decay=SGD_DECAY, nesterov=True)
# rlrop = ReduceLROnPlateau(monitor='val_acc', mode='max', factor=0.5, patience=10, min_lr=0.00001, verbose=1)
