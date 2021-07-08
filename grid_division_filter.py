from matplotlib.pyplot import axis
import numpy as np
import tensorflow as tf
import time
import json
import cv2
import random


class GridSegmentatingModel(tf.keras.Model):
    def __init__(self, downsampling = 4, name="GridSegmentatingModel", **kwargs):
        super(GridSegmentatingModel, self).__init__(name=name, **kwargs)

        self.conv_1 = tf.keras.layers.Conv2D(
            256, (5, 5), strides=(1, 1), padding='same', activation="relu")
        self.conv_2 = tf.keras.layers.Conv2D(
            128, (7, 7), strides=(2, 2), padding='same', activation="relu")
        self.conv_3 = tf.keras.layers.Conv2D(
            64, (3, 3), strides=(2, 2), padding='same', activation="relu")
        self.max_pool_1 = tf.keras.layers.MaxPool2D((2, 2), strides = (1,1), padding='same')
        self.conv_4 = tf.keras.layers.Conv2D(
            64, (3, 3), strides=(1, 1), padding='same', activation="relu")

        self.flatten = tf.keras.layers.Flatten()
        self.dense_01 = tf.keras.layers.Dense(256, activation='relu')
        self.dense_02 = tf.keras.layers.Dense(256, activation='relu')
        self.dense_03 = tf.keras.layers.Dense((480 // downsampling) * (640 // downsampling), activation='sigmoid')
        self.reshape = tf.keras.layers.Reshape((480 // downsampling, 640 // downsampling, 1))

        self.multi = tf.keras.layers.Multiply()
        self.conv_6 = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation="relu")
        self.conv_out = tf.keras.layers.Conv2D(1, (1, 1), strides=(1, 1), activation="sigmoid")

    def call(self, input):
        x = self.conv_1(input)
        x = self.conv_2(x)
        y = self.conv_3(x)
        x = self.max_pool_1(y)
        x = self.conv_4(x)

        y = self.flatten(y)
        y = self.dense_01(y)
        y = self.dense_02(y)
        y = self.dense_03(y)
        y = self.reshape(y)
        
        x = self.multi([x,y])
        x = self.conv_6(x)
        x = self.conv_out(x)
        return x


class Data_generator(tf.keras.utils.Sequence):
    def __init__(self, downsampling=4, batch_size = 1, shuffle=True, val=False, cv = 10, current_fold = 0):
        with open("train_anno.json", 'r') as f:
            self.annotations = json.load(f)[100 * 5:]
        val_size = len(self.annotations) // cv
        start_val = current_fold * val_size // 5 * 5
        if val:
            self.annotations = self.annotations[start_val:val_size + start_val]
        else:
            self.annotations = self.annotations[0:start_val] + self.annotations[start_val + val_size:]
        self.downsampling = downsampling
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return np.math.ceil(len(self.annotations) / self.batch_size)
    def __getitem__(self, index):
        imgs = []
        gts = []
        for i in range(self.batch_size):
            if index * self.batch_size + i >= len(self.annotations): break
            anno = self.annotations[index * self.batch_size + i]
            seq = int(anno['sequence_id'])
            frm = int(anno['frame'])
            obj_coords = anno['object_coords']
            img = cv2.imread("train/" + str(seq) + "/" + str(frm) + ".png", 0)
            gt = np.zeros((np.math.ceil(img.shape[0] / self.downsampling),np.math.ceil(img.shape[1] / self.downsampling)))
            for coord in obj_coords:
                gt[int(coord[1]) // self.downsampling, int(coord[0]) // self.downsampling] = 255
            imgs.append(np.reshape(img, (img.shape[0], img.shape[1], 1)))
            gts.append(np.reshape(gt, (gt.shape[0], gt.shape[1], 1)))
        imgs = np.array(imgs)
        gts = np.array(gts)
        return imgs / 255., gts / 255.

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.annotations)

def F1_metric(y_true, y_pred):
    smoothing = 0.01
    intersection = tf.keras.backend.sum(y_true * y_pred, axis=[1,2,3])
    union = tf.keras.backend.sum(y_true, axis=[1,2,3]) + tf.keras.backend.sum(y_pred, axis=[1,2,3])
    dice = tf.keras.backend.mean((2. * intersection + smoothing)/(union + smoothing), axis=0)
    return dice

def train_model(batch_size = 2, cv = 5, kfold = False, epochs=10, model_number = None):
    if kfold:
      best_score = 0
      for i in range(cv):
        model = GridSegmentatingModel()

        train_gen = Data_generator(downsampling=4, batch_size=batch_size, cv = cv, current_fold = i)
        val_gen = Data_generator(downsampling=4, val=True, cv = cv, current_fold = i)

        model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                      loss=tf.keras.losses.binary_crossentropy, metrics=[F1_metric])

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_F1_metric', patience=2, restore_best_weights=True, mode="max")

        history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=[callback])

        if best_score < history.history['val_F1_metric'][-1]:
          best_score = history.history['val_F1_metric'][-1]
          best_model = model
          t = str(int(time.time()) % 1000000000)
          model.save('model/models/grid_filter' + t + '_' + str(best_score))
          print("Saved: model" + t)

      t = str(int(time.time()) % 1000000000)
      best_model.save('model/models/grid_filter' + t + '_' + str(1000*best_score//1))
      print("Saved: model" + t + '_' + str(1000*best_score//1))
      return best_model

    else:
      if model_number is None:
        model = GridSegmentatingModel()
      else:
        model = tf.keras.models.load_model('model/models/grid_filter' + str(model_number), compile=False)
      train_gen = Data_generator(downsampling=4, batch_size=batch_size)
      val_gen = Data_generator(downsampling=4, val=True)

      model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                    loss=tf.keras.losses.binary_crossentropy, metrics=[F1_metric])
      
      callback = tf.keras.callbacks.EarlyStopping(monitor='val_F1_metric', patience=3, restore_best_weights=True, mode="max")
      
      history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=[callback])
      
      # saving the trained model / best model
      t = str(int(time.time()) % 1000000000)
      model.save('model/models/grid_filter' + t + '_' + str(int(1000*history.history['val_F1_metric'][-1])))
      print("Saved: model" + t + '_' + str(int(1000*history.history['val_F1_metric'][-1])))
      model.summary()
      return model


def filter(imgs, model=None, model_number = None):
    if model is None:
        if model_number is None:
            model = tf.keras.models.load_model(
                'model/models/grid_filter' + str('625428212_433'), compile=False)
        else:
            model = tf.keras.models.load_model(
                'model/models/grid_filter' + str(model_number), compile=False)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                  loss=tf.keras.losses.binary_crossentropy, metrics=[F1_metric])

    imgs = np.expand_dims(imgs.copy(), axis=-1)
    imgs = imgs / 255.

    r = model.predict(imgs)
    new_imgs = []
    for im in r:
        im = cv2.resize(im, (640, 480), interpolation=cv2.INTER_NEAREST)
        new_imgs.append(im)
    new_imgs = np.array(new_imgs)
    return new_imgs, model

if __name__ == "__main__":
    model = None
    # model = train_model(batch_size = 1, cv = 6, kfold = True)
    # model = train_model(batch_size = 4, epochs=10, model_number = "624801661_10")
    # model = train_model(batch_size = 1, epochs=30)
    img = cv2.imread("train/1/1.png", 0)
    imgs = np.array([img])
    r, model = filter(imgs, model, '625428212_433')
    import matplotlib.pyplot as plt
    plt.imshow(r[0 ].astype(float))
    plt.show()

