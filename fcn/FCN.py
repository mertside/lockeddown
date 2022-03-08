#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
from __future__ import print_function
 
from tensorflow import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import sys

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=23000)])

def readucr(filename):
    data = np.loadtxt(filename, delimiter = ',')
    Y = data[:,0]
    X = data[:,1:]
    return X, Y

def normalizelist(x):
    z = []
    for y in x:
        ymax = max(y)
        ymin = min(y)
        w = []
        for i in range(len(y)):
            w.append((y[i] - ymin) / (ymax - ymin))
        z.append(w)
    return np.array(z)
  
fname = "data/clean/RTX2060S-chromeData/DEMO"
if len(sys.argv) > 1:
  fname = sys.argv[1]

nb_epochs = 1
if len(sys.argv) > 2:
  nb_epochs = int(sys.argv[2])

x_train, y_train = readucr(fname+'_TRAIN')
x_test, y_test = readucr(fname+'_TEST')
nb_classes = len(np.unique(y_test))
batch_size = min(x_train.shape[0]/10, 16)

y_train = (y_train - y_train.min())/(y_train.max()-y_train.min())*(nb_classes-1)
y_test = (y_test - y_test.min())/(y_test.max()-y_test.min())*(nb_classes-1)


Y_train = keras.utils.to_categorical(y_train, nb_classes)
Y_test = keras.utils.to_categorical(y_test, nb_classes)

# Normalization...
# x_train_mean = x_train.mean()
# x_train_std = x_train.std()
# x_train = (x_train - x_train_mean)/(x_train_std)   
# x_test = (x_test - x_train_mean)/(x_train_std)

x_train = normalizelist(x_train)
x_test = normalizelist(x_test)
# Normalization . 

x_train = x_train.reshape(x_train.shape + (1,1,))
x_test = x_test.reshape(x_test.shape + (1,1,))

x = keras.layers.Input(x_train.shape[1:])
#    drop_out = Dropout(0.2)(x)
conv1 = keras.layers.Conv2D(128, 8, 1, padding='same')(x)
conv1 = keras.layers.BatchNormalization()(conv1)
conv1 = keras.layers.Activation('relu')(conv1)

#    drop_out = Dropout(0.2)(conv1)
conv2 = keras.layers.Conv2D(256, 5, 1, padding='same')(conv1)
conv2 = keras.layers.BatchNormalization()(conv2)
conv2 = keras.layers.Activation('relu')(conv2)

#    drop_out = Dropout(0.2)(conv2)
conv3 = keras.layers.Conv2D(128, 3, 1, padding='same')(conv2)
conv3 = keras.layers.BatchNormalization()(conv3)
conv3 = keras.layers.Activation('relu')(conv3)

full = keras.layers.GlobalAveragePooling2D()(conv3)
out = keras.layers.Dense(nb_classes, activation='softmax')(full)


model = keras.models.Model(inputs=x, outputs=out)
    
# New stuff...
callbacks = [
  keras.callbacks.ModelCheckpoint(
    "best_model.h5", save_best_only=True, monitor="val_loss"
  ),
  keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
  ),
  #keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]
model.compile(
  optimizer="adam",
  loss="sparse_categorical_crossentropy",
  metrics=["sparse_categorical_accuracy"],
)
hist = model.fit(
  x_train,
  y_train,
  batch_size=batch_size,
  epochs=nb_epochs,
  callbacks=callbacks,
  validation_split=0.2,
  verbose=1,
)
log = pd.DataFrame(hist.history)
# New stuff .

# optimizer = keras.optimizers.Adam()
# model.compile(loss='categorical_crossentropy',
#                 optimizer=optimizer,
#                 metrics=['accuracy'])
    
# reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'loss', factor=0.5,
#                     patience=50, min_lr=0.0001) 
# hist = model.fit(x_train, Y_train, batch_size=batch_size, epochs=nb_epochs,
#             verbose=1, validation_data=(x_test, Y_test), callbacks = [reduce_lr])
# #Print the testing results which has the lowest training loss.
# log = pd.DataFrame(hist.history)
# # print(log.loc[log['loss'].idxmin]['loss'], log.loc[log['loss'].idxmin]['val_acc'])

modelname = 'MyModel'
if len(sys.argv) > 1:
  modelname = 'Models/' + sys.argv[3] 

tf.saved_model.save(model, modelname + '/')
