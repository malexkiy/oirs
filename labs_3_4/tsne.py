import os
import h5py
import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from random import random

h5_data = 'output/data.h5'
h5_labels = 'output/labels.h5'

# import the feature vector and trained labels
h5f_data = h5py.File(h5_data, 'r')
h5f_label = h5py.File(h5_labels, 'r')

global_features = np.array(h5f_data['dataset_1'])
global_labels = np.array(h5f_label['dataset_1'])

h5f_data.close()
h5f_label.close()

# class_names = ["daffodil", "snowdrop", "lilyvalley", "bluebell", "crocus",
#                "iris", "tigerlily", "tulip", "fritillary", "sunflower",
#                "daisy", "coltsfoot", "dandelion", "cowslip", "buttercup",
#                "windflower", "pansy"]

class_names = os.listdir('dataset')

ss = StandardScaler()
scaled_features = ss.fit_transform(global_features)

tsne = TSNE(n_components=2, perplexity=22)
tsne_data = tsne.fit_transform(scaled_features)

plt.figure(figsize=(7, 6))

for index, label in enumerate(class_names):
    color = (random(), random(), random(), 0.95)
    plt.scatter(tsne_data[global_labels == index, 0], tsne_data[global_labels == index, 1], color=color, label=label)

plt.legend()
plt.show()
