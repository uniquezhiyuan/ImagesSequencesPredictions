import numpy as np
from PIL import Image
import os
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
import time
from keras.utils import multi_gpu_model
from keras import optimizers

WIDTH = 100
HEIGHT = 100
FRAMES = 16

SEQUENCE = np.load('sequence_array.npz')['sequence_array']  # load array
print(SEQUENCE[0])
print('Data loaded.')
print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

NUMBER = len(SEQUENCE)

'''
i = 0
while i < NUMBER:
    if (i + 1) % 11 != 0:
        BASIC_SEQUENCE = np.append(BASIC_SEQUENCE, SEQUENCE[i])
        NEXT_SEQUENCE = np.append(NEXT_SEQUENCE, SEQUENCE[i+1])
    i += 1
    print(i)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
'''

# step =1
SEQUENCE = SEQUENCE.reshape(NUMBER, WIDTH, HEIGHT, 1)
# step =2
SEQUENCE_2 = []
for i in range(int(NUMBER / 2)):
    SEQUENCE_2.append(SEQUENCE[2 * i])

# step = 3
SEQUENCE_3 = []
for i in range(int(NUMBER / 3)):
    SEQUENCE_3.append(SEQUENCE[3 * i])

def get_sequence()


SEQUENCE = SEQUENCE.reshape(NUMBER, WIDTH, HEIGHT, 1)
BASIC_SEQUENCE = np.zeros((NUMBER-FRAMES, FRAMES, WIDTH, HEIGHT, 1))
NEXT_SEQUENCE = np.zeros((NUMBER-FRAMES, FRAMES, WIDTH, HEIGHT, 1))


for i in range(FRAMES):
    print(i)
    BASIC_SEQUENCE[:, i, :, :, :] = SEQUENCE[i:i+NUMBER-FRAMES]
    NEXT_SEQUENCE[:, i, :, :, :] = SEQUENCE[i+1:i+NUMBER-FRAMES+1]



#plt.imshow(BASIC_SEQUENCE[0][0].reshape(100, 100))
#plt.show()
# build model


seq = Sequential()

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),input_shape=(None, WIDTH, HEIGHT, 1), padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=60, kernel_size=(3, 3), padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=60, kernel_size=(3, 3), padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3), padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3), activation='sigmoid', padding='same', data_format='channels_last'))

# sgd = optimizers.SGD(lr=0.01, clipvalue=0.5)

# seq.compile(loss='binary_crossentropy', optimizer='adadelta')
'''
seq.compile(loss='mean_squared_error', optimizer='adadelta')

seq.fit(BASIC_SEQUENCE[:10], NEXT_SEQUENCE[:10], batch_size=32,
        epochs=2, validation_split=0.05)


'''

parallel_model = multi_gpu_model(seq, gpus=4)
sgd = optimizers.SGD(lr=0.01, clipnorm=1)
#rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
#adadelta_ = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
parallel_model.compile(loss='mean_squared_error', optimizer='adadelta')


parallel_model.fit(BASIC_SEQUENCE[:1000], NEXT_SEQUENCE[:1000], batch_size=10, epochs=10, validation_split=0.05)


seq.save('nice_model.h5')

which = 600
track = BASIC_SEQUENCE[which][:12, ::, ::, ::]

for j in range(FRAMES+1):
    new_pos = seq.predict(track[np.newaxis, ::, ::, ::, ::])
    new = new_pos[::, -1, ::, ::, ::]
    track = np.concatenate((track, new), axis=0)

# And then compare the predictions
# to the ground truth
track2 = BASIC_SEQUENCE[which][::, ::, ::, ::]
for i in range(FRAMES):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)
    if i >= 8:
        ax.text(1, 3, 'Predictions !', fontsize=20, color='w')
    else:
        ax.text(1, 3, 'Inital trajectory', fontsize=20)
    toplot = track[i, ::, ::, 0]
    plt.imshow(toplot, cmap='binary')
    ax = fig.add_subplot(122)
    plt.text(1, 3, 'Ground truth', fontsize=20)
    toplot = track2[i, ::, ::, 0]
    if i >= 8:
        toplot = NEXT_SEQUENCE[which][i - 1, ::, ::, 0]
    plt.imshow(toplot, cmap='binary')
    plt.savefig('%i_animate.png' % (i + 1))


# 1201.23
# 1132.25 385.68

