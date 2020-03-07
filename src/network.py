import os
import keras
import numpy as np
import tensorflow as tf
from Dog_dict import Dogs, Dogs_all
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation
from keras.layers import Dense, Dropout, Flatten
from sklearn.model_selection import train_test_split

# gpu config
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# training config
epochs = 5
batch_size = 32
data_augmentation = True
data_dir = os.path.join(os.path.dirname(os.getcwd()), 'np_data')

# load matrix data
num_classes = len(Dogs)
x_data = np.load(os.path.join(data_dir, 'x_min.npy'))
y_data = np.load(os.path.join(data_dir, 'y_min.npy'))

# split data to train and test
input_shape = x_data[0].shape
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=42)

# create sequential cnn
model = Sequential()
model.add(Conv2D(32, (2, 2), padding='same',
                 input_shape=input_shape))
model.add(Activation('relu'))

model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.15))

model.add(Conv2D(64, (2, 2), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.15))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# compile model
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# train model
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          verbose=1)

