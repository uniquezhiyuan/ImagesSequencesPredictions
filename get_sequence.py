import numpy as np
from PIL import Image
import os
import time

IMAGE_PATH = 'd:/radar/image/sample/12/'
#IMAGE_PATH = 'c:/data/radar/sample/12/'

WIDTH = 100
HEIGHT = 100
SEQUENCE = np.array([])
BASIC_SEQUENCE = np.array([])
NEXT_SEQUENCE = np.array([])
NUMBER = 0

def image_initialize(image):
    picture = Image.open(image)
    picture = picture.crop((243, 176, 1428, 1280))
    picture = picture.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
    picture = picture.convert('L')
    picture.save('c:/temp/1.png')  # 非保留
    data = np.array(picture.getdata()).reshape(WIDTH, HEIGHT, 1)
    return data

for file in os.listdir(IMAGE_PATH):
    # print(os.path.join(IMAGE_PATH, directories))
    # print(os.path.join(os.path.join(IMAGE_PATH, directories), file))
    image_array = image_initialize(os.path.join(IMAGE_PATH, file))
    SEQUENCE = np.append(SEQUENCE, image_array)
    NUMBER += 1
    print(NUMBER)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

SEQUENCE = SEQUENCE.reshape(NUMBER, WIDTH * HEIGHT)
for i in SEQUENCE:
    for j in range(int(len(i))):
        if i[j] < 50:
            i[j] = 0


np.savez('sequence_array.npz', sequence_array=SEQUENCE)
print('Data saved.')
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))



'''
for directories in os.listdir(IMAGE_PATH):
    # print(os.path.join(IMAGE_PATH, directories))
    for file in os.listdir(os.path.join(IMAGE_PATH, directories)):
        # print(os.path.join(os.path.join(IMAGE_PATH, directories), file))
        image_array = image_initialize(os.path.join(os.path.join(IMAGE_PATH, directories), file))
        SEQUENCE = np.append(SEQUENCE, image_array)
        NUMBER += 1
        print(NUMBER)
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
'''