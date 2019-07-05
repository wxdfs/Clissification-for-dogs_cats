# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:08:07 2019

@author: HT
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import os
tf.enable_eager_execution()
from tensorflow import keras
import numpy as np
import PIL.Image as Image
#zip_file = tf.keras.utils.get_file(origin="https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip",
#                                   fname="cats_and_dogs_filtered.zip", extract=True)
#base_dir, _ = os.path.splitext("C:\Users\HT\.keras\datasets\cats_and_dogs")
base_dir = 'C:/Users/HT/.keras/datasets/cats_and_dogs'
train_dir = os.path.join(base_dir,'training_set')
validation_dir = os.path.join(base_dir,'test_set')

train_cats_dir = os.path.join(train_dir,'cats')
train_dogs_dir = os.path.join(train_dir,'dogs')
validation_cats_dir = os.path.join(validation_dir,'cats')
validation_dogs_dir = os.path.join(validation_dir,'dogs')
class_names= ['cat','dog']

image_size = 160
batch_size = 128 

train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                             shear_range = 0.2, 
                                                             zoom_range = 0.2,
                                                             horizontal_flip = True)
validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(image_size,image_size),
                                                    batch_size=batch_size,
                                                    class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              target_size=(image_size,image_size),
                                                              batch_size=batch_size,
                                                              class_mode='binary')
image_shape = (image_size,image_size,3)

model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(input_shape=image_shape,
                               filters=64,
                               kernel_size=[5,5],
                               padding='same',
                               activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPool2D(pool_size=[2,2],strides=2),
                
        tf.keras.layers.Conv2D(filters=96,
                               kernel_size=[5,5],
                               padding='same',
                               activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPool2D(pool_size=[2,2],strides=2),
                
        tf.keras.layers.Conv2D(filters=128,
                               kernel_size=[3,3],
                               padding='same',
                               activation=tf.keras.activations.relu),
        tf.keras.layers.Conv2D(filters=128,
                               kernel_size=[3,3],
                               padding='same',
                               activation=tf.keras.activations.relu),
                
        tf.keras.layers.Conv2D(filters=256,
                               kernel_size=[2,2],
                               padding='same',
                               activation=tf.keras.activations.relu),
        tf.keras.layers.MaxPool2D(pool_size=[2,2],strides=2),
       
        tf.keras.layers.Flatten(input_shape=[-1,20*20*256]),
        tf.keras.layers.Dense(units=512,activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(rate=0.3),
        tf.keras.layers.Dense(units=512,activation=tf.keras.activations.relu),
        tf.keras.layers.Dropout(rate=0.3),
        tf.keras.layers.Dense(units=2,activation=tf.keras.activations.softmax)])
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
epochs = 1
train_steps_per_epoch=train_generator.n
validation_steps_per_epoch=validation_generator.n

checkpoint_path = 'E:\checkpiont\cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                  save_weight_only=True,
                                                  verbose=2)
history = model.fit_generator(train_generator,
                              steps_per_epoch=train_steps_per_epoch,
                              epochs=epochs,
                              workers=4,
                              validation_data=validation_generator,
                              validation_steps=validation_steps_per_epoch)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,max(plt.ylim())])
plt.title('Training and Validation Loss')
plt.show()

img = Image.open('C:/Users/HT/.keras/datasets/cats_and_dogs/test_set/dogs/dog.4071.jpg')
new_img = img.resize((160,160))
test_img = np.expand_dims(tf.keras.preprocessing.image.img_to_array(new_img), axis = 0) 
predictions = model.predict(test_img)
pre = np.argmax(predictions)
pre_label = class_names[pre]
plt.figure()
plt.imshow(img)
plt.grid(False)
plt.title(pre_label)
plt.show()