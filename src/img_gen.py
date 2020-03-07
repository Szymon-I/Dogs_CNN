import os
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
from Dog_dict import Dogs, Dogs_all
from PIL import Image

# set generator function for distorting images
ia.seed(1)
seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Crop(percent=(0, 0.1)),

    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
    iaa.ContrastNormalization((0.75, 1.5)),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True)  # apply augmenters in random order

# set constants
MAIN_DIR = os.path.dirname(os.getcwd())
IMG_DIR = os.path.join(MAIN_DIR, 'images_slice')
categories_number = len(Dogs)
DATA_DIR = os.path.join(MAIN_DIR, 'np_data')

X = []
Y = []
WIDTH, HEIGHT = 200, 200
counter = 0
create_n = 30 # numbers of augumented images to create of 1

# iterate every image in subfolders and create augumented images of them and save to imagaug folder
for dog_folder in os.listdir(os.path.join(os.getcwd(), IMG_DIR)):
    if dog_folder not in Dogs_all:
        continue
    for image_path in os.listdir(os.path.join(os.getcwd(), IMG_DIR, dog_folder)):
        counter += 1
        if counter % 200 == 0:
            print(f'Images converted: {counter}')
        try:
            image = os.path.join(os.getcwd(), IMG_DIR, dog_folder, image_path)
            img_arr = cv2.imread(image, cv2.IMREAD_UNCHANGED)
            img_arr = cv2.resize(img_arr, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            img = [img_arr for _ in range(create_n)]
            images_aug = seq(images=img)
            pic_name = ''.join(image.split('/')[-1])
            for i in range(create_n):
                im = Image.fromarray(np.uint8(images_aug[i]))
                new_name = pic_name[:-4] + f'_gen_{i}.jpg'
                folder_path = os.path.join(MAIN_DIR, 'imgaug', dog_folder)
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                new_path = os.path.join(MAIN_DIR, 'imgaug', dog_folder, new_name)
                im.save(new_path)
        except:
            pass
