# import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2 as cv
from glob import glob

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.applications import mobilenet_v2
from keras.models import Sequential
from keras.models import Model
from keras.preprocessing.image import load_img
from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.applications import InceptionV3 as IV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as v3_preprocess

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.keras.optimizers import Adam

import warnings

warnings.filterwarnings('ignore')

# import data
train = './train'
test = './test'
label = 'labels.csv'
train_size = len(os.listdir(train))
test_size = len(os.listdir(test))
# ensure that import was successful
print("Training set : " + str(train_size))
print("Test set : " + str(test_size))
# check labels
labels_df = pd.read_csv(label)
num_rows = 0
for row in open(label):
    num_rows += 1
print(labels_df.head(100))

# extract classes for classification
breeds = labels_df["breed"].unique()  # look for strings that reoccur within file
classes = breeds.size
breeds.sort()
print(breeds)

# get id numbers for classes
class_to_num = dict(zip(breeds, range(classes)))
print(class_to_num)


# create function to load images to array
def img_to_arr(data_dir, df, image_size):
    image_names = df['id']
    image_labels = df['breed']
    data_size = len(image_names)
    x = np.zeros([data_size, image_size[0], image_size[1], image_size[2]], dtype=np.uint8)
    y = np.zeros([data_size, 1], dtype=np.uint8)

    for i in range(data_size):
        img_name = image_names[i]
        img_dir = os.path.join(data_dir, img_name + '.jpg')
        img_pixels = load_img(img_dir, target_size=image_size)
        x[i] = img_pixels
        y[i] = class_to_num[image_labels[i]]

    y = to_categorical(y)

    ind = np.random.permutation(data_size)
    x = x[ind]
    y = y[ind]
    print('Output Data Size: ', x.shape)
    print('Output Label Size: ', y.shape)
    return x, y


# Image Processing
# set image size
img_size = (255, 255, 3)  # this is solely based on choice of architecture
x, y = img_to_arr(train, labels_df, img_size)


# feature extraction
# function to get features using different models
def get_features(model, data_preprocessor, input_size, data):
    input_layer = Input(input_size)
    preprocessor = Lambda(data_preprocessor)(input_layer)
    base_model = model(weights='imagenet', include_top=False, input_shape=input_size)(preprocessor)
    avg = GlobalAveragePooling2D()(base_model)
    feature_extractor = Model(inputs=input_layer, outputs=avg)

    # Extract feature.
    feature_maps = feature_extractor.predict(data, batch_size=32, verbose=1)
    print('Feature maps shape: ', feature_maps.shape)
    return feature_maps


# Model 1: Inception V3
inception_features = get_features(IV3, v3_preprocess, img_size, x)
# this ensures the network stops learning when optimal parameter values are achieved
EarlyStop_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
my_callback = [EarlyStop_callback]

# build classification model
model = Sequential()
model.add(InputLayer(inception_features.shape[1:]))
model.add(Dropout(0.7))
model.add(Dense(120, activation='softmax'))
# Compiling Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# Training Model
history = model.fit(inception_features, y, batch_size=10, epochs=50, validation_split=0.1, callbacks=my_callback)
