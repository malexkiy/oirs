# -----------------------------------
# GLOBAL FEATURE EXTRACTION
# -----------------------------------
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import cv2
import os
import h5py
from features import fd_hu_moments, fd_haralick, fd_histogram, fd_canny, fd_laplacian, fd_sobelx, fd_sobely, fd_kaze, fd_sift

# --------------------
# tunable-parameters
# --------------------
images_per_class = 80
fixed_size = tuple((500, 500))
train_path = os.path.join('train')
output_dir = 'output'
h5_data = os.path.join(output_dir, 'data.h5')
h5_labels = os.path.join(output_dir, 'labels.h5')

# get the training labels
train_labels = os.listdir(train_path)

# sort the training labels
train_labels.sort()
print(train_labels)

# empty lists to hold feature vectors and labels
global_features = []
labels = []

# loop over the training data sub-folders
for training_name in train_labels:
    # join the training data path and each species training folder
    current_dir = os.path.join(train_path, training_name)
    images = [f for f in os.listdir(current_dir) if os.path.isfile(os.path.join(current_dir, f)) and
              (f.endswith(".jpg") or f.endswith(".png"))]

    # get the current training label
    current_label = training_name

    # loop over the images in each sub-folder
    for file in images:
        # get the image file name
        file_path = os.path.join(current_dir, file)

        # read the image and resize it to a fixed-size
        image = cv2.imread(file_path)
        image = cv2.resize(image, fixed_size)

        ####################################
        # Global Feature extraction
        ####################################
        # fv_hu_moments = fd_hu_moments(image)
        fv_haralick = fd_haralick(image)
        fv_histogram = fd_histogram(image)
        # fv_kaze = fd_kaze(image)
        # fv_fast = fd_fast(image)
        fv_sift = fd_sift(image)
        # fv_canny = fd_canny(image)
        # fv_sobelx = fd_sobelx(image)
        # fv_sobely = fd_sobely(image)
        # fv_laplacian = fd_laplacian(image)

        ###################################
        # Concatenate global features
        ###################################
        # global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
        global_feature = np.hstack([fv_sift, fv_histogram, fv_haralick])
        # global_feature = np.hstack([fv_canny, fv_sobelx, fv_sobely, fv_laplacian])

        # update the list of labels and feature vectors
        labels.append(current_label)
        global_features.append(global_feature)

    print("[STATUS] processed folder: {}".format(current_label))

print("[STATUS] completed Global Feature Extraction...")

# get the overall feature vector size
# print("[STATUS] feature vector size {}".format(np.array(global_features).shape))

# get the overall training label size
# print("[STATUS] training Labels {}".format(np.array(labels).shape))

# encode the target labels
targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)
print("[STATUS] training labels encoded...")

# scale features in the range (0-1)
# scaler = MinMaxScaler(feature_range=(0, 1))
# rescaled_features = scaler.fit_transform(global_features)
rescaled_features = global_features
print("[STATUS] feature vector normalized...")

# print("[STATUS] target labels: {}".format(target))
# print("[STATUS] target labels shape: {}".format(target.shape))

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# save the feature vector using HDF5
h5f_data = h5py.File(h5_data, 'w')
h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

h5f_label = h5py.File(h5_labels, 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))

h5f_data.close()
h5f_label.close()

print("[STATUS] end of training..")

