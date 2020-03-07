import datetime
import os

import tensorflow as tf
from DogImageGenerator import DogImageGenerator
from keras.applications import MobileNet
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from Dog_dict import Dogs_all

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
all_data = True
epochs = 1
batch_size = 6
data_augmentation = True
save_dir = os.path.join(os.path.dirname(os.getcwd()), 'models')
data_dir = os.path.join(MAIN_DIR, 'np_data')
num_classes = len(Dogs_all)
DogGenerator = DogImageGenerator(batch_size)

input_shape = DogGenerator.WIDTH, DogGenerator.HEIGHT

# imports the mobilenet model and discards the last 1000 neuron layer
base_model = MobileNet(weights='imagenet',
                       include_top=False)

# add cutom output layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
preds = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=preds)

# compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# train with custom generator
steps = DogGenerator.images_number // batch_size
model.fit_generator(generator=DogGenerator.generate_train_batch(),
                    validation_data=DogGenerator.generate_test_batch(),
                    steps_per_epoch=steps,
                    validation_steps=steps // 2,
                    epochs=epochs,
                    callbacks=[tensorboard_callback])
