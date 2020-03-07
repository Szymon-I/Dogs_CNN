import datetime
import os
import numpy as np
import tensorflow as tf
from DogImageGeneratorImgaug import DogImageGeneratorImgaug
from keras.engine.saving import load_model
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# config session, gpu and tensorboard callbacks
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
MAIN_DIR = os.path.dirname(os.getcwd())
log_dir = os.path.join(MAIN_DIR, 'logs', 'fit', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)

# load pretrained model
model_path = os.path.join(MAIN_DIR, 'models', 'best_model.h5')
model = load_model(model_path)

# config for training
epochs = 2
batch_size = 64
DogGenerator = DogImageGeneratorImgaug(batch_size)

try:
    # train using generator
    steps = DogGenerator.images_number // batch_size
    model.fit_generator(generator=DogGenerator.generate_train_batch(),
                        validation_data=DogGenerator.generate_test_batch(),
                        steps_per_epoch=steps,
                        validation_steps=steps // 2,
                        epochs=epochs,
                        callbacks=[tensorboard_callback])
except:
    # evaluate with np array in ram (if encountered error
    data_dir = os.path.join(MAIN_DIR, 'np_data')
    x_data = np.load(os.path.join(data_dir, 'x_min.npy'))
    y_data = np.load(os.path.join(data_dir, 'y_min.npy'))
    result = model.evaluate(x_data, y_data)
    print(f'loss: {result[0]}  acc: {result[1]}')
    # val_loss: 2.886842060420248  val_acc: 0.37083333333333335
