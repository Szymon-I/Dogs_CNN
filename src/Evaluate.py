import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from Dog_dict import Dogs
from keras.engine.saving import load_model
from keras.utils import normalize, to_categorical
from sklearn.metrics import confusion_matrix

WIDTH, HEIGHT = 200, 200

MAIN_DIR = os.path.dirname(os.getcwd())
model_path = os.path.join(MAIN_DIR, 'models', 'best_model.h5')
model = load_model(model_path)


# return key of given dog name
def get_dog_key(pic_name):
    check = pic_name.split('_')[0].lower()
    for k, v in Dogs.items():
        if check in k.lower():
            return v
    print('Dog unknown')
    return -1


x = []
y = []
images = []
# iterate all images in EvalImages folder and append to lists
for image in os.listdir(os.path.join(MAIN_DIR, 'EvalImages')):
    try:
        img = cv2.imread(os.path.join(MAIN_DIR, 'EvalImages', image), cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        (b, g, r) = cv2.split(img)
        img = cv2.merge([r, g, b])
        images.append(image)
        x.append(img)
        y.append(get_dog_key(image))
    except:
        print("Encountered and error while predicting....")

# normalize arrays and make predictions based on model
x = normalize(np.array(x))
y = to_categorical(np.array(y), len(Dogs), int)

predictions = model.predict(x)
conf_matrix = confusion_matrix(y.argmax(axis=1), predictions.argmax(axis=1))

loss, acc = model.evaluate(x, y)

# print prdictions within filenames
for i, image in enumerate(images):
    print(f'{image} -> {list(Dogs.keys())[predictions[i].argmax()]}')

# print evaluation stats
print(f'loss: {loss} acc: {acc}')

# show confusion matrix
plt.imshow(conf_matrix, cmap='binary', interpolation='None')
plt.show()
