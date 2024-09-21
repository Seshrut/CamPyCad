import numpy as np
import cv2 as cv
import os
test_images = np.array([cv.imread("photos/dataset_final/battery_r0/1.png"), cv.imread("photos/dataset_final/cap_r1/1.png")])

test_images = test_images.astype(np.float32)
test_images = test_images.reshape(len(test_images), -1)
print('loading model')
svm = cv.ml.SVM.create().load('svm_model.xml')
print('starting test')
predictions = svm.predict(test_images)[1]
print('test ended successfully')
print(predictions)