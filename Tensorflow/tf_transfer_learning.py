# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 20:13:01 2022

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

#%% import VGG16


from tensorflow.keras.applications import VGG16

preprocess_input=tf.keras.applications.vgg16.preprocess_input


#%% import data

base_dir=r"C:/Users/cinar/Desktop/tf_armed_forces"

train_datagen=ImageDataGenerator(rescale=1./255,validation_split=0.1,
                                 preprocessing_function=preprocess_input)


test_datagen=ImageDataGenerator(rescale=1./255,validation_split=0.1,
                                preprocessing_function=preprocess_input)

train_datagen=train_datagen.flow_from_directory(base_dir,
                                                target_size=(224,224),
                                                subset="training",
                                                batch_size=2,
                                                class_mode="sparse")

test_datagen=test_datagen.flow_from_directory(base_dir,
                                                target_size=(224,224),
                                                subset="validation",
                                                batch_size=2,
                                                class_mode="sparse")

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


#%% Transfer Learning

vgg_model=VGG16()

vgg_model.summary()
"""
Model: "vgg16"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_1 (InputLayer)        [(None, 224, 224, 3)]     0         
                                                                 
 block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792      
                                                                 
 block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928     
                                                                 
 block1_pool (MaxPooling2D)  (None, 112, 112, 64)      0         
                                                                 
 block2_conv1 (Conv2D)       (None, 112, 112, 128)     73856     
                                                                 
 block2_conv2 (Conv2D)       (None, 112, 112, 128)     147584    
                                                                 
 block2_pool (MaxPooling2D)  (None, 56, 56, 128)       0         
                                                                 
 block3_conv1 (Conv2D)       (None, 56, 56, 256)       295168    
                                                                 
 block3_conv2 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                 
 block3_conv3 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                 
 block3_pool (MaxPooling2D)  (None, 28, 28, 256)       0         
                                                                 
 block4_conv1 (Conv2D)       (None, 28, 28, 512)       1180160   
                                                                 
 block4_conv2 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                 
 block4_conv3 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                 
 block4_pool (MaxPooling2D)  (None, 14, 14, 512)       0         
                                                                 
 block5_conv1 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                 
 block5_conv2 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                 
 block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                 
 block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0         
                                                                 
 flatten (Flatten)           (None, 25088)             0         
                                                                 
 fc1 (Dense)                 (None, 4096)              102764544 
                                                                 
 fc2 (Dense)                 (None, 4096)              16781312  
                                                                 
 predictions (Dense)         (None, 1000)              4097000   
                                                                 
=================================================================
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
_________________________________________________________________

"""

print(type(vgg_model))


#%% Model Create Sequential

transfer_learning_model=Sequential()

for layer in vgg_model.layers[0:-1]:
    transfer_learning_model.add(layer)
    
transfer_learning_model.summary()

"""
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792      
                                                                 
 block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928     
                                                                 
 block1_pool (MaxPooling2D)  (None, 112, 112, 64)      0         
                                                                 
 block2_conv1 (Conv2D)       (None, 112, 112, 128)     73856     
                                                                 
 block2_conv2 (Conv2D)       (None, 112, 112, 128)     147584    
                                                                 
 block2_pool (MaxPooling2D)  (None, 56, 56, 128)       0         
                                                                 
 block3_conv1 (Conv2D)       (None, 56, 56, 256)       295168    
                                                                 
 block3_conv2 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                 
 block3_conv3 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                 
 block3_pool (MaxPooling2D)  (None, 28, 28, 256)       0         
                                                                 
 block4_conv1 (Conv2D)       (None, 28, 28, 512)       1180160   
                                                                 
 block4_conv2 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                 
 block4_conv3 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                 
 block4_pool (MaxPooling2D)  (None, 14, 14, 512)       0         
                                                                 
 block5_conv1 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                 
 block5_conv2 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                 
 block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                 
 block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0         
                                                                 
 flatten (Flatten)           (None, 25088)             0         
                                                                 
 fc1 (Dense)                 (None, 4096)              102764544 
                                                                 
 fc2 (Dense)                 (None, 4096)              16781312  
                                                                 
=================================================================
Total params: 134,260,544
Trainable params: 134,260,544
Non-trainable params: 0
_________________________________________________________________
"""

for layer in transfer_learning_model.layers :
    layer.trainable=False

transfer_learning_model.summary()
"""
Total params: 134,260,544
Trainable params: 0
Non-trainable params: 134,260,544
"""


transfer_learning_model.add(layers.Dense(4))#4 classımız var
transfer_learning_model.summary()

"""
dense (Dense)               (None, 4)                 16388     
"""


#%% Train Model

optimizer=tf.keras.optimizers.Adamax(learning_rate=0.0001)
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


transfer_learning_model.compile(optimizer=optimizer,loss=loss,metrics=["accuracy"])

result=transfer_learning_model.fit(train_datagen,
                    epochs=10,
                    verbose=1,
                    validation_data=test_datagen)


result.history

#%% Model Test

transfer_learning_model.evaluate(test_datagen)

"""
14/14 [==============================] - 3s 224ms/step - loss: 0.6977 - accuracy: 0.8519
Out[18]: [0.6976951956748962, 0.8518518805503845]
              loss                  accuracy
"""


#%% Model Test from test dataset

print(test_datagen.class_indices)
""" {'aircraft': 0, 'battleship': 1, 'combat tank': 2, 'helicopter': 3} """

for i in range(4):
    img,label=test_datagen.next()
    pred=transfer_learning_model.predict(img)
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
    image=transform.resize(image,(224,224,3))
    image=np.expand_dims(image,axis=0)
    print(test_datagen.class_indices)
    return image
    

random_image=img("random_images/random_image4.jpg")#tank

pred=transfer_learning_model.predict(random_image)
print(pred)
np.argmax(pred)
""" Out[23]: 2   tank  succesfully"""


#%% Model test 2

test_a=transfer_learning_model.predict(test_datagen)

t=[]
print(test_datagen.class_indices)

for i in test_a:
    t.append(np.argmax(i))
    

x=zip(t,test_datagen.labels)

for i,j in x:
    print("Tahmin :{} Gerçek:{}".format(i,j))


#%% Model save and load

#Save_model
transfer_learning_model.save("tl_model_save/")#recommended
transfer_learning_model.save("tl_model_save/video.h5")

#save_weights
transfer_learning_model.save_weights("tl_save_model_weights/")
transfer_learning_model.save_weights("tl_save_model_weights/video.h5")

#Load_model
transfer_learning_model2=transfer_learning_model.load_weights("tl_save_model_weights/video.h5")


#%%











