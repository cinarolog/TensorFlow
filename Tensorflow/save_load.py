# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 16:21:08 2022

@author: cinar
"""


#%% import libraries


import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator

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
                                                target_size=(500,500),
                                                subset="training",
                                                batch_size=2)

test_datagen=test_datagen.flow_from_directory(base_dir,
                                                target_size=(500,500),
                                                subset="validation",
                                                batch_size=2)

"""
Found 253 images belonging to 4 classes.
Found 27 images belonging to 4 classes.
"""



#%% Visualization

import matplotlib.pyplot as plt

for i in range(5):
    img,label=test_datagen.next()
    print(img.shape)
    plt.imshow(img[0])
    print(label[0])
    plt.show()


#%% Model Create Sequential

tf_model=Sequential()

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

"""
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 496, 496, 4)       304       
                                                                 
 max_pooling2d (MaxPooling2D  (None, 248, 248, 4)      0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 246, 246, 8)       296       
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 123, 123, 8)      0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 122, 122, 16)      528       
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 61, 61, 16)       0         
 2D)                                                             
                                                                 
 conv2d_3 (Conv2D)           (None, 60, 60, 32)        2080      
                                                                 
 flatten (Flatten)           (None, 115200)            0         
                                                                 
 dense (Dense)               (None, 50)                5760050   
                                                                 
 dense_1 (Dense)             (None, 100)               5100      
                                                                 
 dense_2 (Dense)             (None, 100)               10100     
                                                                 
 dense_3 (Dense)             (None, 50)                5050      
                                                                 
 dense_4 (Dense)             (None, 4)                 204       
                                                                 
=================================================================
Total params: 5,783,712
Trainable params: 5,783,712
Non-trainable params: 0
_________________________________________________________________

"""


#%% Train Model

optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001)
loss=tf.keras.losses.CategoricalCrossentropy()

tf_model.compile(optimizer=optimizer,loss=loss,metrics=["mse","accuracy"])

result=tf_model.fit(train_datagen,
                    epochs=10,
                    verbose=1,
                    validation_data=test_datagen)


result.history
"""
Out[12]: 
{'loss': [1.0840461254119873,
  0.304612398147583,
  0.08632176369428635,
  0.06418485194444656,
  0.009552608244121075,
  0.005773614160716534,
  0.0036335820332169533,
  0.0025959813501685858,
  0.0019160083029419184,
  0.0015218162443488836],
 'mse': [0.13266979157924652,
  0.038567546755075455,
  0.008875003084540367,
  0.007901025004684925,
  0.00013838018639944494,
  7.286231266334653e-05,
  2.2431830075220205e-05,
  1.1549715054570697e-05,
  5.932252406637417e-06,
  3.983452643296914e-06],
 'accuracy': [0.6324110627174377,
  0.8972331881523132,
  0.9841897487640381,
  0.9802371263504028,
  1.0,
  1.0,
  1.0,
  1.0,
  1.0,
  1.0],
 'val_loss': [0.6574852466583252,
  0.2871999144554138,
  0.26102539896965027,
  0.11931294202804565,
  0.13683131337165833,
  0.1293783038854599,
  0.11917878687381744,
  0.1232861876487732,
  0.14276181161403656,
  0.11026348918676376],
 'val_mse': [0.09392514079809189,
  0.037194155156612396,
  0.036150142550468445,
  0.014324693009257317,
  0.021284984424710274,
  0.0205820482224226,
  0.018948305398225784,
  0.020246338099241257,
  0.02391164004802704,
  0.017982900142669678],
 'val_accuracy': [0.6666666865348816,
  0.9259259104728699,
  0.8888888955116272,
  0.9629629850387573,
  0.9259259104728699,
  0.9259259104728699,
  0.9259259104728699,
  0.9259259104728699,
  0.9259259104728699,
  0.9259259104728699]}

"""


#%% Model Test

tf_model.evaluate(test_datagen)

"""
14/14 [==============================] - 1s 65ms/step - loss: 0.1685 - mse: 0.0228 - accuracy: 0.9630
Out[7]: [0.16848529875278473, 0.022784778848290443, 0.9629629850387573]
              loss                 mse                 accuracy
"""


#%% Model Test from test dataset

print(test_datagen.class_indices)
""" {'aircraft': 0, 'battleship': 1, 'combat tank': 2, 'helicopter': 3} """

for i in range(4):
    img,label=test_datagen.next()
    pred=tf_model.predict(img)
    np.argmax(pred[0])
    plt.imshow(img[0])
    
    if np.argmax(pred[0])==0:
        print("Aircraft")
        
    if np.argmax(pred[0])==1:
        print("Battleship")
        
    if np.argmax(pred[0])==2:
        print("Combat Tank")
        
    if np.argmax(pred[0])==3:
        print("Helicopter")
        
    plt.show()    



#%% Model test on a single image

from PIL import Image
from skimage import transform

random_image=Image.open(r"random_images/random_image4.jpg")

def img(path):
    image=Image.open(path)
    image=np.array(image).astype("float32")/255
    image=transform.resize(image,(500,500,3))
    image=np.expand_dims(image,axis=0)
    print(test_datagen.class_indices)
    return image
    

random_image=img("random_images/random_image4.jpg")#tank

pred=tf_model.predict(random_image)
print(pred)
np.argmax(pred)
""" Out[23]: 2   tank  succesfully"""


#%% Model test 2

test_a=tf_model.predict(test_datagen)

t=[]
print(test_datagen.class_indices)

for i in test_a:
    t.append(np.argmax(i))
    

x=zip(t,test_datagen.labels)

for i,j in x:
    print("Tahmin :{} Gerçek:{}".format(i,j))


#%% Model save and load

#Save_model
tf_model.save("model_save/")#recommended
tf_model.save("model_save/video.h5")

#save_weights
tf_model.save_weights("save_model_weights/")
tf_model.save_weights("save_model_weights/video.h5")

#Load_model
tf_model2=tf_model.load_weights("save_model_weights/video.h5")






