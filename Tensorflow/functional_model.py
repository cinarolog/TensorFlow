# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 17:51:43 2022

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


#%% Functional Model

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model


input_=Input(shape=(500,500,3))

conv11=Conv2D(filters=4,kernel_size=(2,2),activation="elu")(input_)
max11=MaxPooling2D((2,2))(conv11)
conv12=Conv2D(filters=8,kernel_size=(2,2),activation="elu")(max11)
max12=MaxPooling2D((2,2))(conv12)
flat1=Flatten()(max12)

conv21=Conv2D(filters=4,kernel_size=(3,3),activation="elu")(input_)
max21=MaxPooling2D((2,2))(conv21)
conv22=Conv2D(filters=8,kernel_size=(2,2),activation="elu")(max21)
max22=MaxPooling2D((2,2))(conv22)
flat2=Flatten()(max22)

merge=concatenate([flat1,flat2])

#Linear layers
fc1=Dense(50,activation="elu")(merge)
fc2=Dense(100,activation="elu")(fc1)
fc3=Dense(100,activation="elu")(fc2)
fc4=Dense(50,activation="elu")(fc3)
output=Dense(4,activation="softmax")(fc4)


model=Model(inputs=input_,outputs=output)



#%%model summary

model.summary()

"""
_________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_3 (InputLayer)           [(None, 500, 500, 3  0           []                               
                                )]                                                                
                                                                                                  
 conv2d_8 (Conv2D)              (None, 496, 496, 4)  304         ['input_3[0][0]']                
                                                                                                  
 conv2d_10 (Conv2D)             (None, 496, 496, 4)  304         ['input_3[0][0]']                
                                                                                                  
 max_pooling2d_8 (MaxPooling2D)  (None, 248, 248, 4)  0          ['conv2d_8[0][0]']               
                                                                                                  
 max_pooling2d_10 (MaxPooling2D  (None, 248, 248, 4)  0          ['conv2d_10[0][0]']              
 )                                                                                                
                                                                                                  
 conv2d_9 (Conv2D)              (None, 244, 244, 8)  808         ['max_pooling2d_8[0][0]']        
                                                                                                  
 conv2d_11 (Conv2D)             (None, 244, 244, 8)  808         ['max_pooling2d_10[0][0]']       
                                                                                                  
 max_pooling2d_9 (MaxPooling2D)  (None, 122, 122, 8)  0          ['conv2d_9[0][0]']               
                                                                                                  
 max_pooling2d_11 (MaxPooling2D  (None, 122, 122, 8)  0          ['conv2d_11[0][0]']              
 )                                                                                                
                                                                                                  
 flatten_4 (Flatten)            (None, 119072)       0           ['max_pooling2d_9[0][0]']        
                                                                                                  
 flatten_5 (Flatten)            (None, 119072)       0           ['max_pooling2d_11[0][0]']       
                                                                                                  
 concatenate_2 (Concatenate)    (None, 238144)       0           ['flatten_4[0][0]',              
                                                                  'flatten_5[0][0]']              
                                                                                                  
 dense_10 (Dense)               (None, 50)           11907250    ['concatenate_2[0][0]']          
                                                                                                  
 dense_11 (Dense)               (None, 100)          5100        ['dense_10[0][0]']               
                                                                                                  
 dense_12 (Dense)               (None, 100)          10100       ['dense_11[0][0]']               
                                                                                                  
 dense_13 (Dense)               (None, 50)           5050        ['dense_12[0][0]']               
                                                                                                  
 dense_14 (Dense)               (None, 4)            204         ['dense_13[0][0]']               
                                                                                                  
==================================================================================================
Total params: 11,929,928
Trainable params: 11,929,928
Non-trainable params: 0
__________________________________________________________________________________________________


"""

#%% Train Model

optimizer=tf.keras.optimizers.Adamax(learning_rate=0.001)
loss=tf.keras.losses.CategoricalCrossentropy()

model.compile(optimizer=optimizer,loss=loss,metrics=["mse","accuracy"])

result=model.fit(train_datagen,
                    epochs=10,
                    verbose=1,
                    validation_data=test_datagen)


result.history


#%% Model Test

model.evaluate(test_datagen)

"""
14/14 [==============================] - 1s 69ms/step - loss: 0.2108 - mse: 0.0230 - accuracy: 0.9630
Out[27]: [0.21075083315372467, 0.02298165298998356, 0.9629629850387573]
              loss                 mse                 accuracy
"""


#%% Model Test from test dataset

print(test_datagen.class_indices)
""" {'aircraft': 0, 'battleship': 1, 'combat tank': 2, 'helicopter': 3} """

for i in range(4):
    img,label=test_datagen.next()
    pred=model.predict(img)
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

pred=model.predict(random_image)
print(pred)
np.argmax(pred)
""" Out[23]: 2   tank  succesfully"""


#%% Model test 2

test_a=model.predict(test_datagen)

t=[]
print(test_datagen.class_indices)

for i in test_a:
    t.append(np.argmax(i))
    

x=zip(t,test_datagen.labels)

for i,j in x:
    print("Tahmin :{} Gerçek:{}".format(i,j))



"""
Tahmin :3 Gerçek:0
Tahmin :1 Gerçek:0
Tahmin :2 Gerçek:0
Tahmin :3 Gerçek:0
Tahmin :2 Gerçek:0
Tahmin :0 Gerçek:0
Tahmin :2 Gerçek:0
Tahmin :1 Gerçek:1
Tahmin :1 Gerçek:1
Tahmin :2 Gerçek:1
Tahmin :0 Gerçek:1
Tahmin :3 Gerçek:1
Tahmin :0 Gerçek:2
Tahmin :0 Gerçek:2
Tahmin :2 Gerçek:2
Tahmin :0 Gerçek:2
Tahmin :3 Gerçek:2
Tahmin :0 Gerçek:2
Tahmin :1 Gerçek:2
Tahmin :2 Gerçek:3
Tahmin :0 Gerçek:3
Tahmin :0 Gerçek:3
Tahmin :3 Gerçek:3
Tahmin :2 Gerçek:3
Tahmin :1 Gerçek:3
Tahmin :3 Gerçek:3
Tahmin :3 Gerçek:3
"""












#%%











#%%


















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





#%%















#%%











#%%
















