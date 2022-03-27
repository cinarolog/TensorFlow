# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 19:38:53 2022

@author: cinar
"""

#%% import libraries

import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

import numpy as np
import pylab as pl
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.core import Dense,Dropout,Activation,Flatten 
from tensorflow.keras import layers,activations


#%% import data

base_dir=r"C:/Users/cinar/Desktop/tf_armed_forces"

train_datagen=ImageDataGenerator(rescale=1./255,validation_split=0.1)
test_datagen=ImageDataGenerator(rescale=1./255,validation_split=0.1)

train_datagen=train_datagen.flow_from_directory(base_dir,
                                                target_size=(224,224),
                                                subset="training",
                                                batch_size=2)

test_datagen=test_datagen.flow_from_directory(base_dir,
                                                target_size=(224,224),
                                                subset="validation",
                                                batch_size=2)

"""
Found 253 images belonging to 4 classes.
Found 27 images belonging to 4 classes.
"""



 #%% Data augmentation / Veri arttırma

veri_arttırma=Sequential([
    
layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical") 
# layers.experimental.preprocessing.RandomCrop(100, 100),
# layers.experimental.preprocessing.RandomContrast(factor=0.2),
# layers.experimental.preprocessing.RandomRotation(factor=0.2)
    
])

img,_=test_datagen.next()
image=tf.expand_dims(img,0)
plt.figure(figsize=(10,6))

for i in range(6):
    img,_=test_datagen.next()
    aug_image=veri_arttırma(img)
    ax=plt.subplot(3,3,i+1)
    plt.imshow(aug_image[0])
    plt.axis("off")



#%% Model Create Sequential

tf_model=Sequential()



"""
extra olarak modelimize veri_arttırma kısmını ekliyoruz
"""
tf_model.add(veri_arttırma)




tf_model.add(layers.Conv2D(filters=4,
                           activation="elu",
                           kernel_size=(5,5),
                           input_shape=(500,500,3)
                           ))

tf_model.add(layers.MaxPooling2D((2,2)))

tf_model.add(layers.Conv2D(filters=8,activation="elu",kernel_size=(3,3)))

tf_model.add(layers.MaxPooling2D((2,2)))

tf_model.add(layers.Conv2D(filters=16,activation="elu",kernel_size=(2,2)))

tf_model.add(layers.MaxPooling2D((2,2)))

tf_model.add(layers.Conv2D(filters=32,activation="elu",kernel_size=(2,2)))

tf_model.add(layers.Flatten())

tf_model.add(layers.Dense(50,activation="elu"))
tf_model.add(layers.Dense(100,activation="elu"))
tf_model.add(layers.Dense(100,activation="elu"))
tf_model.add(layers.Dense(50,activation="elu"))
tf_model.add(layers.Dense(4,activation="softmax"))



#%% Model Summary

tf_model.summary()



#%% Train Model

optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001)
loss=tf.keras.losses.CategoricalCrossentropy()

tf_model.compile(optimizer=optimizer,loss=loss,metrics=["mse","accuracy"])

result=tf_model.fit(train_datagen,
                    epochs=10,
                    verbose=1,
                    validation_data=test_datagen)


result.history


#%%











#%%










