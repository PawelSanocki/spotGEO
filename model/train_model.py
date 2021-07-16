import numpy as np
import settings as settings
import tensorflow as tf
import time
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
import tensorflow_addons as tfa

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
print("Version: ",tf.__version__)

BATCH_SIZE = 32
WINDOW_SIZE = settings.WINDOW_SIZE
SEED = settings.SEED
TRAIN_DIR = "model/dataset_nn"

def get_trained_model():
    ds_size = len(os.listdir("model/dataset_nn/1"))
    ds_size += len(os.listdir("model/dataset_nn/0"))
    print(ds_size)

    #https://keras.io/api/preprocessing/image/
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory=TRAIN_DIR,
        labels="inferred",
        label_mode="binary",
        # label_mode="",
        image_size=(WINDOW_SIZE,WINDOW_SIZE),
        seed=SEED,
        color_mode="grayscale",
        batch_size=BATCH_SIZE
    )

    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.
        return image, label * 0.9999

    dataset = dataset.map(preprocess)

    ds_size //= BATCH_SIZE

    train_size = int(0.9 * ds_size)
    val_size = int(0.05 * ds_size)

    train_ds = dataset.take(train_size)
    test_ds = dataset.skip(train_size)
    val_ds = test_ds.take(val_size)
    test_ds = test_ds.skip(val_size)

    def get_augmenter():
        augmenter = tf.keras.Sequential()
        augmenter.add(tf.keras.layers.experimental.preprocessing.RandomFlip(input_shape = (WINDOW_SIZE,WINDOW_SIZE,1)))
        return augmenter

    def get_model():
        model = tf.keras.Sequential()
        model.add(get_augmenter())
        model.add(tf.keras.layers.Conv2D(128, (5, 5), activation='relu'))
        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(tf.keras.layers.Conv2D(128, (2, 2), activation='relu'))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(512, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss=tf.keras.losses.binary_crossentropy, metrics=[tfa.metrics.F1Score(num_classes=1, average='micro', threshold=0.4)])
        return model

    def get_model_2():
        input = tf.keras.Input(shape=(WINDOW_SIZE,WINDOW_SIZE,1))
        img = get_augmenter()(input)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(input)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        y = tf.keras.layers.Flatten()(img)
        xy = tf.keras.layers.concatenate([x, y])
        xy = tf.keras.layers.Dense(256, activation='relu')(xy)
        xy = tf.keras.layers.Dropout(0.1)(xy)
        xy = tf.keras.layers.Dense(128, activation='relu')(xy)
        xy = tf.keras.layers.Dropout(0.1)(xy)
        xy = tf.keras.layers.Dense(64, activation='relu')(xy)
        xy = tf.keras.layers.Dropout(0.1)(xy)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(xy)
        model = tf.keras.Model(input, output)
        # model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tf.keras.losses.MeanSquaredError())
        model.compile(optimizer='Adam', loss="binary_crossentropy", 
            metrics=[tfa.metrics.F1Score(num_classes=1, average='micro', threshold=0.4)]
            )
        return model

    model = get_model_2()

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_f1_score', patience=20, restore_best_weights=True, mode="max")

    hist = model.fit(train_ds, validation_data=val_ds, epochs=200, callbacks=[callback])
    model.summary()

    t = str(int(time.time()) % 1000000000)
    model.save('model/models/model' + t)
    model.save('model/models/model' + t + '_' + str(int(1000*hist.history['val_f1_score'][-1])))
    print("Saved: model" + t + '_' + str(int(1000*hist.history['val_f1_score'][-1])))

    acc = model.evaluate(test_ds, verbose=0, return_dict=True)
    print(acc)

    return model
# model = tf.keras.models.load_model('models/model' + time)

if __name__ == "__main__":
    get_trained_model()