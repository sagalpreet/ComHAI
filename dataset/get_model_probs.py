from keras.models import load_model

import numpy as np
import pandas as pd

MODEL_NAME = '2_conv_layers/cnn_9'

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def to_string(x):
    return str(x, 'utf-8')

model = load_model(f'model/{MODEL_NAME}')
predictions = model(unpickle(f'raw/test_batch')[b'data'].reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1))
df = pd.DataFrame(predictions)
df = df.applymap(lambda x: float(x))

true_labels = unpickle(f'raw/test_batch')[b'labels']
df['true_labels'] = true_labels
df['filenames'] = list(map(to_string, unpickle(f'raw/test_batch')[b'filenames']))

print(f'Accuracy: {np.mean(1*(np.argmax(predictions, axis = 1) == true_labels))}')

df.to_csv('raw/model_probs.csv', index=False)
