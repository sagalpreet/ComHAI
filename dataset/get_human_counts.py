import pandas as pd
import numpy as np

from collections import defaultdict
import random

random.seed(1253)

df = pd.read_csv('raw/cifar10h-raw.csv', index_col=False)

encodings = {
    'airplane': 0,
    'automobile': 1,
    'bird': 2,
    'cat': 3,
    'deer': 4,
    'dog': 5,
    'frog': 6,
    'horse': 7,
    'ship': 8,
    'truck': 9
}

NUM_CLASSES = 10

human_counts = defaultdict(lambda: [0 for i in range(NUM_CLASSES)])

def preprocess():
    global df
    global human_counts
    n = len(df)
    image_filename, human_label = df['image_filename'], df['chosen_label']

    for i in range(n):
        human_counts[image_filename[i]][human_label[i]] += 1

    return


def get_human_counts(filename):
    return human_counts[filename]
    # global df
    # temp = list(df[df['image_filename'] == image_path]['chosen_label'])
    # l = [0 for _ in range(10)]
    # for i in temp:
    #     l[i] += 1
    # return l

def process():
    global df
    n = len(df)
    image_filename, true_category = df['image_filename'], df['true_category']

    data = list(set([(df['image_filename'][i], encodings[true_category[i]]) for i in range(n)]))
    n = len(data)

    true_labels = []
    filenames = []
    human_counts = [[] for _ in range(10)]

    for i in range(n):
        print(f"{i}/{n}\r", end='\r')
        image_filename, true_category = data[i]
        
        counts = get_human_counts(image_filename)
        for i in range(10):
            human_counts[i].append(counts[i])
        
        true_labels.append(true_category)
        filenames.append(image_filename)
    
    df = pd.DataFrame(np.array(human_counts).T)
    df['true_labels'] = true_labels
    df['filenames'] = filenames

    df.to_csv('raw/human_counts.csv', index=False)

    # print(df)

preprocess()
process()