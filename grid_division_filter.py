import numpy as np
import tensorflow as tf
import time
import json
import cv2
import random


class GridSegmentatingModel(tf.keras.Model):
    def __init__(self, name="GridSegmentatingModel", **kwargs):
        super(GridSegmentatingModel, self).__init__(name=name, **kwargs)

        self.conv_1 = tf.keras.layers.Conv2D(
            16, (3, 3), strides=(1, 1), padding='same', activation="relu")
        self.conv_2 = tf.keras.layers.Conv2D(
            16, (7, 7), strides=(4, 4), padding='same', activation="relu")
        self.conv_3 = tf.keras.layers.Conv2D(
            16, (3, 3), strides=(1, 1), padding='same', activation="relu")
        self.conv_4 = tf.keras.layers.Conv2D(1, (1, 1), strides=(1, 1), activation="sigmoid")

    def call(self, input):
        x = self.conv_1(input)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        return x


class Data_generator(tf.keras.utils.Sequence):
    def __init__(self, downsampling=4, val_size=100, batch_size = 2, shuffle=True, val=False):
        with open("train_anno.json", 'r') as f:
            if val:
                self.annotations = json.load(f)[:val_size*5]
            else:
                self.annotations = json.load(f)[val_size*5:]
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
            anno = self.annotations[index * self.batch_size + i]
            seq = int(anno['sequence_id'])
            frm = int(anno['frame'])
            obj_coords = anno['object_coords']
            img = cv2.imread("train\\" + str(seq) + "\\" + str(frm) + ".png", 0)
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


def train_model():
    model = GridSegmentatingModel()

    train_gen = Data_generator(downsampling=4)
    val_gen = Data_generator(downsampling=4, val=True)


    metrics = [tf.keras.metrics.Recall(name="Recall"),
               tf.keras.metrics.Precision(name="Precision")]

    model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                  loss=tf.keras.losses.binary_crossentropy)

    model.fit(train_gen, epochs=1, validation_data=val_gen)

    t = str(int(time.time()) % 1000000000)
    model.save('model/models/grid_filter' + t)
    print("Saved: model" + t)
    return model


def filter(imgs, model=None):
    if model is None:
        model = tf.keras.models.load_model(
            'model/models/grid_filter' + str(623327572), compile=True)
    new_imgs = model.predict(imgs)
    return new_imgs


if __name__ == "__main__":
    model = None
    # model = train_model()
    img = cv2.imread("train/1/2.png", 0)
    img = np.reshape(img, (1, 480, 640, 1)) / 255.
    r = filter(img, model)[0]
    print(r.max())
    r[r > r.max() / 10. * 3] = 1
    r = cv2.resize(r, (640, 480), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("", r)
    cv2.waitKey()
