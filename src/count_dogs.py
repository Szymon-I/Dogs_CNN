import os

dog_counter = []

# set directories constants
MAIN_DIR = os.path.dirname(os.getcwd())
IMG_DIR = os.path.join(MAIN_DIR, 'images')

# for every folder in path count images inside and add counter to list
for dog_folder in os.listdir(IMG_DIR):
    counter = 0
    for image in os.listdir(os.path.join(os.getcwd(), IMG_DIR, dog_folder)):
        counter += 1
    dog_counter.append((dog_folder, counter))

# sort according to quantity of images
dog_counter.sort(key=lambda x: x[1], reverse=True)
# print only n entries
n = 50
print(f'Sum: {sum([x[1] for x in dog_counter])}\n')
for x in dog_counter[:n]:
    print(f'{x[0]}: {x[1]}')
