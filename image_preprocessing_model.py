
#import library
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

print(tf.__version__)


#Process the training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
#Preprocessing the Test set
test_datagen=ImageDataGenerator(rescale=1./255)
test_set=test_datagen.flow_from_directory(
    'dataset/test_set',target_size=(64,64),batch_size=32, class_mode='binary')

#making CNN

#initailsing the CNN
cnn=tf.keras.models.Sequential()

#STEP-1: Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3,activation='relu',input_shape=[64,64,3]))

#STEP-2: Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

#Adding a second convolution layer
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

#STEP-3 -FLATTENING
cnn.add(tf.keras.layers.Flatten())

#Step-4 - FULL Connection
cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))

#step -5 -Output Layer
cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

#training the CNN model
cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Trainning the CNN on the Trainging set and evaluating it on the Test set
cnn.fit(x=training_set,validation_data=test_set,epochs=25)

#making a single prediction

from keras.preprocessing import image
test_image=image.load_img('dataset/single_prediction/cat_or_dog_1.jpg',target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=cnn.predict(test_image)
training_set.class_indices
if result[0][0]==1:
    prediction='dog'
else:
    prediction='cat'
    
print(prediction)