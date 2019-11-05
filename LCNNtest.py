from __future__ import absolute_import, division, print_function, unicode_literals, unicode_literals

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

train_images = train_images.reshape(train_images.shape[0], 28,28,1)
test_images = test_images.reshape(test_images.shape[0], 28,28,1)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D, MaxPool2D, Convolution2D, ReLU
from tensorflow.keras.layers import BatchNormalization as BN
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np

in1 = Input(shape = (28,28,1))
c1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(in1)
c2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(in1)
x = keras.layers.maximum([c1, c2])
x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
x = Dropout(.2)(x)

c1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(x)
c2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(x)
x = keras.layers.maximum([c1, c2])
x = BN(axis=3, scale=False)(x)
x = Dropout(.2)(x)

c1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(x)
c2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(x)
x = keras.layers.maximum([c2, c1])
x = Dropout(.2)(x)
x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
x = BN(axis=3, scale=False)(x)
x = Flatten()(x)
f1 = Dense(80, activation = None)(x)
f2 = Dense(80, activation = None)(x)
x = keras.layers.maximum([f1, f2])
x = BN()(x)
x = Dropout(.2)(x)
x = Dense(10, activation = 'sigmoid')(x)
model = keras.models.Model(inputs = in1, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(train_images, train_labels, epochs=5, batch_size = 1000)

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min',patience = 8)

hist = model.fit(train_images, train_labels, epochs=50,  
                 batch_size=1000, validation_data = (test_images, test_labels), 
                 callbacks=[es]) 
