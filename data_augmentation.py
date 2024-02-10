import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
import tensorflow as tf
import pandas as pd
import tensorflow.keras.layers as tlf
from tensorflow import keras

#from tensorflow.keras.preprocessing import image_dataset_from_directory
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation

# directory='/home/enchi/Documentos/PEF/TRAINING/'
# df=pd.read_csv(directory+'labels.csv')

# file_paths=df['image_name'].values
# labels=df['classes'].values
# ds_train=tf.data.Dataset.from_tensor_slices((file_paths,labels))

# def read_image(image_file,label):
#     image=tf.io.read_file(directory+image_file)
#     image=tf.image.decode_image(image,channels=1,dtype=tf.float32)
#     return image,label

# def augment(image,label):
#     #data agumentation here
#     return image,label

# ds_train=ds_train.map(read_image).map(augment).batch(2)
#-----------------------------------------------------------------
img_height=28
img_width=28
batch_size=2

model=keras.Sequential([
    tlf.Input((28,28,1)),
    tlf.Conv2D(16,3,padding='same'),
    tlf.Conv2D(32,3,padding='same'),
    tlf.MaxPooling2D(),
    tlf.Flatten(),
    tlf.Dense(10),
])

ds_train=tf.keras.preprocessing.image_dataset_from_directory(
    '/home/enchi/Documentos/PEF/MOCA_PNG/',
    labels='inferred',
    label_mode="int", #categorical,binary
    #class_names=['0','1','2',...]
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(img_height,img_width),#reshape if not in this size
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="training"
)


df_validation=tf.keras.preprocessing.image_dataset_from_directory(
    '/home/enchi/Documentos/PEF/MOCA_PNG/',
    labels='inferred',
    label_mode="int", #categorical,binary
    #class_names=['0','1','2',...]
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(img_height,img_width),#reshape if not in this size
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset="validation"
)

def augment(x, y):
    image=tf.image.random_brightness(x,max_delta=0.05)
    return image,y

ds_train=ds_train.map(augment)

class_names=ds_train.class_names
print(class_names)
