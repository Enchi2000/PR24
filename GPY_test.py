import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
from tensorflow.keras import layers

batch_size = 12
img_height = 180
img_width = 180


data_dir='/home/enchi/Documentos/PEF/ROI/'

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

num_images_before_augmentation = len(train_ds)
print("Number of images before data augmentation:", num_images_before_augmentation*batch_size)

data_augmentation=tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])

class_names = train_ds.class_names
print(class_names)

#Mostrar imagenes
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    augmented_images = data_augmentation(images)
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

num_images_after_augmentation = len(augmented_images) * batch_size
print("Number of images after data augmentation:", num_images_after_augmentation)