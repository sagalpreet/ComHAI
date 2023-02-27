from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD

import numpy as np
import pandas as pd

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

model = None

baseline = True
if (not baseline):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')) #added
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    opt = SGD(lr=1e-5, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
else:
    model = load_model('model/3_conv_layers/cnn_206')

for _ in range(1):
	for i in range(5):
		data = unpickle(f'raw/data_batch_{i%5 + 1}')
		X = data[b'data'].reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
		Y = np.array(list(map(lambda k: [(i==k)*1 for i in range(10)], data[b'labels'])))

		model.fit(X, Y, epochs = 1)
	model.save(f'model/3_conv_layers/cnn_{_}')
