#importing stuff
import cv2
import numpy as np
import os
#path to the directries
positive_dir = "path/to/positive/images"
negative_dir = "path/to/negative/images"
#positive samples
positive_files = os.listdir(positive_dir)
negative_files = os.listdir(negative_dir)
#training images in list
positive_images = [cv2.imread(os.path.join(positive_dir, f)) for f in positive_files]
negative_images = [cv2.imread(os.path.join(negative_dir, f)) for f in negative_files]
#labelling data
positive_samples = np.array([cv2.resize(img, (24, 24)) for img in positive_images])
negative_samples = np.array([cv2.resize(img, (24, 24)) for img in negative_images])
#
positive_labels = np.ones((len(positive_samples), 1))
negative_labels = np.zeros((len(negative_samples), 1))

samples = np.concatenate((positive_samples, negative_samples))
labels = np.concatenate((positive_labels, negative_labels))

haarcascade = cv2.CascadeClassifier()

haarcascade.train(samples, labels, cv2.ml.EM_TRAIN_AUTO, 15, 10)

haarcascade.save("haarcascade_my_detector.xml")