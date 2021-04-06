from model.create_dataset import WINDOW_SIZE
import numpy as np
import tensorflow as tf
import time
import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
print("Version: ",tf.__version__)
ds_size = len(os.listdir("model/dataset_nn/1"))
ds_size += len(os.listdir("model/dataset_nn/0"))
print(ds_size)

from model import settings
BATCH_SIZE = 32
WINDOW_SIZE = settings.WINDOW_SIZE
SEED = 321
TRAIN_DIR = "model/dataset_nn"

#https://keras.io/api/preprocessing/image/
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory=TRAIN_DIR,
    labels="inferred",
    label_mode="binary",
    image_size=(WINDOW_SIZE,WINDOW_SIZE),
    seed=SEED,
    color_mode="grayscale",
    batch_size=BATCH_SIZE
)

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.
    return image, label - 0.0001

dataset = dataset.map(preprocess)

ds_size //= BATCH_SIZE

train_size = int(0.9 * ds_size)
val_size = int(0.05 * ds_size)

train_ds = dataset.take(train_size)
test_ds = dataset.skip(train_size)
val_ds = test_ds.take(val_size)
test_ds = test_ds.skip(val_size)

# def print_dataset(name, dataset):
#     elems = [1 for v in dataset]
#     print("Dataset {} contains {} elements :".format(name, len(elems)))
# print_dataset("train",train_ds)
# print_dataset("val",val_ds)
# print_dataset("test",test_ds)

def get_augmenter():
    augmenter = tf.keras.Sequential()
    augmenter.add(tf.keras.layers.experimental.preprocessing.RandomFlip(input_shape = (WINDOW_SIZE,WINDOW_SIZE,1)))
    return augmenter

def get_model():
    model = tf.keras.Sequential()
    model.add(get_augmenter())
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    # model.compile(optimizer='Adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
    model.compile(optimizer='Adam', loss=tf.keras.losses.MeanSquaredError(), metrics=[])
    return model

model = get_model()

model.fit(train_ds, validation_data=val_ds, epochs=5)
model.summary()

t = str(int(time.time()) % 10000000)
model.save('model/models/model' + t)
print("Saved: model" + t)

acc = model.evaluate(test_ds, verbose=0, return_dict=True)
print(acc)
    
# model = tf.keras.models.load_model('models/model' + time)

