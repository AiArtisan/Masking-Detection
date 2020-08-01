

import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import random
import time
from tensorflow.keras import layers, Model, Sequential



# the image size for netwokr-input
img_size_net = 160
# the train batch
batch_size = 32
# the path of dataset
train_path = './train/'
test_path = './val/'

# path of each kind in the dataset_path
sorts_list = ['mask', 'nomask']
# name for lables, maybe Chinese
wordlist = ['mask', 'nomask']
# path for result
run_path = './run/'
if not os.path.exists(run_path):
    os.mkdir(run_path)


####################################################
# prepare the train data

# 1. read data_set
def load_valid_data(data_path):
    # num of each kind of samples
    cnt_each = np.zeros(2, dtype=np.int)
    # num of all samples
    img_cnt = 0
    # counter for lable
    label_cnt = 0
    test_images = []
    test_lables = []
    for sort_path in sorts_list:
        flower_list = os.listdir(data_path + sort_path)
        for img_name in flower_list:
            img_path = data_path + sort_path + "/" + img_name
            img = cv2.imread(img_path)
            img_scale = cv2.resize(img, (img_size_net, img_size_net), interpolation=cv2.INTER_CUBIC)
            if not img is None:
                test_images.append(img_scale)
                test_lables.append(label_cnt)

                # static the num of different lable
                cnt_each[label_cnt] += 1



        label_cnt += 1
    print('The samples in the data contain: mask-', cnt_each[0], ', nomask-', cnt_each[1])
    return test_images, test_lables

#load train_data, val_data
(validSet_images, validSet_lables) = load_valid_data(train_path)
dataSet_img = np.array(validSet_images)
dataSet_lable = np.array(validSet_lables)

(validSet_images_test, validSet_lables_test) = load_valid_data(test_path)
dataSet_img_test = np.array(validSet_images_test)
dataSet_lable_test = np.array(validSet_lables_test)

from sklearn.utils import shuffle

dataSet_img, dataSet_lable = shuffle(dataSet_img, dataSet_lable)
dataSet_img = np.array(dataSet_img, dtype=np.float32)
dataSet_lable = np.array(dataSet_lable)
dataSet_img = dataSet_img / 255.

dataSet_img_test, dataSet_lable_test = shuffle(dataSet_img_test, dataSet_lable_test)
dataSet_img_test = np.array(dataSet_img_test, dtype=np.float32)
dataSet_lable_test = np.array(dataSet_lable_test)
dataSet_img_test = dataSet_img_test / 255.



trainSet_img = dataSet_img
testSet_img = dataSet_img_test
trainSet_label = dataSet_lable
testSet_label = dataSet_lable_test



#model_design
input_image = layers.Input(shape=(img_size_net, img_size_net, 3), name='x_input', dtype='float32')
x = keras.layers.Conv2D(32, (3, 3), padding="same")(input_image)
x = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
x = keras.layers.ReLU()(x)
x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

x = keras.layers.Conv2D(64, (3, 3), padding="same")(x)
x = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
x = keras.layers.ReLU()(x)
x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

x = keras.layers.Conv2D(128, (3, 3), padding="same")(x)
x = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
x = keras.layers.ReLU()(x)
x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

x = keras.layers.Conv2D(128, (3, 3), padding="same")(x)
x = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
x = keras.layers.ReLU()(x)
x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

x = keras.layers.Conv2D(256, (3, 3), padding="same")(x)
x = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
x = keras.layers.ReLU()(x)
x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

x = keras.layers.Conv2D(512, (3, 3), padding="same")(x)
x = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
x = keras.layers.ReLU()(x)
x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

x = keras.layers.Conv2D(512, (3, 3), padding="same")(x)
x = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
x = keras.layers.ReLU()(x)
x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)

x = keras.layers.Flatten()(x)
x = keras.layers.Dense(200, activation=tf.nn.relu)(x)
x = keras.layers.Dropout(0.5)(x)

output = keras.layers.Dense(2, activation=tf.nn.softmax, name='y_out')(x)

model = Model(inputs=input_image, outputs=output)
# model = keras.models.load_model(run_path + 'img_norm_deeper_new_model.h5')



model.compile(optimizer=keras.optimizers.Adam(lr=0.00001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
print('First train: ')
history = model.fit(trainSet_img, trainSet_label,
                    batch_size=batch_size,
                    epochs=10,
                    validation_data=(testSet_img, testSet_label)
                    )
# import matplotlib.pyplot as plt
# plt.plot(history.history['accuracy'][1:])
# plt.plot(history.history['val_accuracy'][1:])
# plt.legend(['training', 'valivation'], loc='upper left')
# plt.show()

#######################################################################
# save trained model

model_path = run_path + "img_norm_deeper_new_model.h5"
model.save(model_path)
print('The trained result is saved on ', os.path.join(os.getcwd(), model_path))