import os
import cv2
import numpy as np
from Dog_dict import Dogs
from keras.utils import normalize, to_categorical
from sklearn.model_selection import train_test_split

# Batch generator modified for imgaug training

class DogImageGeneratorImgaug:
    DOG_DIR = 'imgaug'
    WIDTH, HEIGHT = 200, 200

    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.MAIN_DIR = os.path.dirname(os.getcwd())
        self.data_train = None
        self.data_test = None
        self.data_list = self.get_data_list()
        self.images_number = len(self.data_list)
        self.test_counter = 0
        self.train_counter = 0

    def get_data_list(self):
        data_list = []
        for dog_folder in os.listdir(os.path.join(self.MAIN_DIR, DogImageGeneratorImgaug.DOG_DIR)):
            if dog_folder not in Dogs:
                continue
            for image_path in os.listdir(os.path.join(self.MAIN_DIR, DogImageGeneratorImgaug.DOG_DIR, dog_folder)):
                full_path = os.path.join(self.MAIN_DIR, DogImageGeneratorImgaug.DOG_DIR, dog_folder, image_path)
                data_list.append((full_path, Dogs[dog_folder]))
        slice_n = len(data_list) % self.batch_size
        data_list = data_list[slice_n:]
        feautures = [x[0] for x in data_list]
        labels = [x[1] for x in data_list]
        x_train, _x_test, y_train, _y_test = train_test_split(feautures, labels,
                                                              test_size=0.25,
                                                              random_state=42)
        self.data_train = x_train, y_train
        MAIN_DIR = os.path.dirname(os.getcwd())
        DATA_DIR = os.path.join(MAIN_DIR, 'np_data')
        x_loaded = np.load(f'{DATA_DIR + os.sep}x_min.npy')
        y_loaded = np.load(f'{DATA_DIR + os.sep}y_min.npy')
        _blank, x_test, blank_2, y_test = train_test_split(x_loaded, y_loaded,
                                                           test_size=0.25,
                                                           random_state=42)
        self.data_test = x_test, y_test
        return data_list

    def reset_generator(self):
        self.test_counter = 0
        self.train_counter = 0

    def get_default(self):
        batch_x = []
        batch_y = []
        img = cv2.imread(self.data_train[0][0], cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (DogImageGeneratorImgaug.WIDTH, DogImageGeneratorImgaug.HEIGHT),
                         interpolation=cv2.INTER_AREA)
        batch_x.append(img)
        batch_y.append(self.data_train[1][0])
        x = normalize(np.array(batch_x))
        y = to_categorical(np.array(batch_y), len(Dogs), int)
        return x, y

    def generate_train_batch(self):
        while True:
            batch_x = []
            batch_y = []
            for i in range(self.batch_size):
                try:
                    img = cv2.imread(self.data_train[0][self.train_counter + i], cv2.IMREAD_UNCHANGED)
                    img = cv2.resize(img, (DogImageGeneratorImgaug.WIDTH, DogImageGeneratorImgaug.HEIGHT),
                                     interpolation=cv2.INTER_AREA)
                    batch_x.append(img)
                    batch_y.append(self.data_train[1][self.train_counter + i])
                except ValueError:
                    break
                except IndexError:
                    break
                except TypeError:
                    break
            self.train_counter += self.batch_size
            if len(batch_x) == 0 or len(batch_y) == 0:
                yield self.get_default()
            x = normalize(np.array(batch_x))
            y = to_categorical(np.array(batch_y), len(Dogs), int)
            yield x, y

    def generate_test_batch(self):
        while True:
            batch_x = []
            batch_y = []
            for i in range(self.batch_size):
                try:
                    img = cv2.imread(self.data_test[0][self.test_counter + i], cv2.IMREAD_UNCHANGED)

                    img = cv2.resize(img, (DogImageGeneratorImgaug.WIDTH, DogImageGeneratorImgaug.HEIGHT),
                                     interpolation=cv2.INTER_AREA)
                    batch_x.append(img)
                    batch_y.append(self.data_test[1][self.test_counter + i])
                except ValueError:
                    break
                except IndexError:
                    break
                except TypeError:
                    break
            self.test_counter += self.batch_size
            if len(batch_x) == 0 or len(batch_y) == 0:
                yield self.get_default()
            #x = normalize(np.array(batch_x))
            #y = to_categorical(np.array(batch_y), len(Dogs), int)
            yield batch_x,batch_y
