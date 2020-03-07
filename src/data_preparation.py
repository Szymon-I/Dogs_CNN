import os

import cv2
import numpy as np
from Dog_dict import Dogs, Dogs_all
from keras.utils import to_categorical, normalize

# all data -> prepare all classes instead of 10
all_data = False
MAIN_DIR = os.path.dirname(os.getcwd())

# set constants
if all_data:
    IMG_DIR = os.path.join(MAIN_DIR, 'images')
    categories_number = len(Dogs_all)
else:
    IMG_DIR = os.path.join(MAIN_DIR, 'images_slice')
    categories_number = len(Dogs)
DATA_DIR = os.path.join(MAIN_DIR, 'np_data')

X = []
Y = []
WIDTH, HEIGHT = 200, 200
counter = 0

# iterate every image in subfolders, change to plain matrix, resize and append to list
for dog_folder in os.listdir(os.path.join(os.getcwd(), IMG_DIR)):
    if dog_folder not in Dogs_all:
        continue
    for image_path in os.listdir(os.path.join(os.getcwd(), IMG_DIR, dog_folder)):
        counter += 1
        if counter % 200 == 0:
            print(f'Images converted: {counter}')
        try:
            image = os.path.join(os.getcwd(), IMG_DIR, dog_folder, image_path)
            img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            (b, g, r) = cv2.split(img)
            img = cv2.merge([r, g, b])
            X.append(img)
        except ValueError:
            pass
        if all_data:
            Y.append(Dogs_all[dog_folder])
        else:
            Y.append(Dogs[dog_folder])

# normalize data for better gradient descent optimalisation and create hot encoder of outputs
x = normalize(np.array(X))
y = to_categorical(np.array(Y), categories_number, int)

# save prepared matrix into bytestream
if all_data:
    np.save(f'{DATA_DIR + os.sep}x_all.npy', x)
    np.save(f'{DATA_DIR + os.sep}y_all.npy', y)
else:
    np.save(f'{DATA_DIR + os.sep}x_min.npy', x)
    np.save(f'{DATA_DIR + os.sep}y_min.npy', y)
