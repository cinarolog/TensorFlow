# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 01:18:51 2022

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


#%% learnin rate

import math
epoch=10
initial_learning_rate=0.001

def lr_step_decay(epoch,lr):
    drop_rate=0.5
    epochs_drop=1.0
    return initial_learning_rate * math.pow(drop_rate,math.floor(epoch/epochs_drop))
    

from tensorflow.keras.callbacks import CSVLogger,EarlyStopping,ModelCheckpoint,TensorBoard,LearningRateScheduler 

callbacks=[
    
    CSVLogger("./Training.csv"),
    TensorBoard(write_graph=True,write_images=1),
    ModelCheckpoint("./best.h5", verbose=1,save_best_only=True,save_weights_only=True,monitor="loss"),
    LearningRateScheduler(lr_step_decay,verbose=1),
    # EarlyStopping(monitor="loss",verbose=1,mode="auto")
    
    ]


callbacks


#%% Train Model

optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001)
loss=tf.keras.losses.CategoricalCrossentropy()

tf_model.compile(optimizer=optimizer,loss=loss,metrics=["mse","accuracy"])

result=tf_model.fit(train_datagen,
                    epochs=epoch,
                    verbose=1,
                    validation_data=test_datagen,
                    callbacks=callbacks)


result.history


#%% Model Test

tf_model.evaluate(test_datagen)

"""
14/14 [==============================] - 1s 61ms/step - loss: 0.2901 - mse: 0.0404 - accuracy: 0.9259
Out[23]: [0.2900674343109131, 0.04035728797316551, 0.9259259104728699]
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


"""
Tahmin :3 Gerçek:0
Tahmin :3 Gerçek:0
Tahmin :1 Gerçek:0
Tahmin :0 Gerçek:0
Tahmin :0 Gerçek:0
Tahmin :0 Gerçek:0
Tahmin :2 Gerçek:0
Tahmin :2 Gerçek:1
Tahmin :0 Gerçek:1
Tahmin :1 Gerçek:1
Tahmin :3 Gerçek:1
Tahmin :3 Gerçek:1
Tahmin :3 Gerçek:2
Tahmin :3 Gerçek:2
Tahmin :3 Gerçek:2
Tahmin :1 Gerçek:2
Tahmin :2 Gerçek:2
Tahmin :0 Gerçek:2
Tahmin :3 Gerçek:2
Tahmin :0 Gerçek:3
Tahmin :1 Gerçek:3
Tahmin :3 Gerçek:3
Tahmin :2 Gerçek:3
Tahmin :2 Gerçek:3
Tahmin :3 Gerçek:3
Tahmin :2 Gerçek:3
Tahmin :2 Gerçek:3
"""

#%%

epoch=range(1,len(result.history["accuracy"])+1)

plt.figure(figsize=(10,6))
plt.plot(epoch,result.history["lr"],color="red")
# plt.plot(epoch,val_acc,label=("Doğrulama Başarısı"),color="green")
plt.title("Eğitim ve Doğrulama Başarısı")
plt.xlabel("Epoch")
plt.ylabel("Learnıng rate")
plt.legend()
plt.show()













