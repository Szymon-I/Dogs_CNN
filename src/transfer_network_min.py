import datetime
import os
import numpy as np
import tensorflow as tf
from Dog_dict import Dogs
from keras.applications import MobileNet
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# config session and gpu along with tensorflow
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
MAIN_DIR = os.path.dirname(os.getcwd())
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
log_dir = os.path.join(MAIN_DIR, 'logs', 'fit', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)

# config training
epochs = 50
batch_size = 64

data_dir = os.path.join(MAIN_DIR, 'np_data')
num_classes = len(Dogs)

# load matrix data
x_data = np.load(os.path.join(data_dir, 'x_min.npy'))
y_data = np.load(os.path.join(data_dir, 'y_min.npy'))
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)

# imports the mobilenet model and discards the last 1000 neuron layer
base_model = MobileNet(weights='imagenet',
                       include_top=False)

# add cutom output layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.6)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
preds = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=preds)

# compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

# train model
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          verbose=1,
          callbacks=[tensorboard_callback])
