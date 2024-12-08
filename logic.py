#get images from imgsticher
from imgsticher import get_image
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import cv2 as cv
import numpy as np
import os
import random
import time

# Example usage:
x,y = get_image()
# print(y)
print(np.array([[(1,2),(2,3),("abc","a")],[(3,4),(4,5),("a",'a')]]))