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
num_img_in_training = 1
sum = 0
X = []
Y = []
for i in range(1,num_img_in_training+1):
    print(f'{int(i*100/num_img_in_training)}%||{sum/i}',end='\r')
    s = time.time()
    x,y = get_image() # x=cv.typing.matlike | y = [(x:int,y:int),(x+120,y+120),label:str]
    e = time.time()
    sum += (e-s)
    X.append(x)
    Y.append(y)
print()
print(sum/num_img_in_training)
print(Y)
for i in range(len(Y)):
    print(f'---- {len(Y[i])}')
    for j in range(len(Y[i])):
        print(len(Y[i][j]))
s = time.time()
X = np.array(X,dtype=np.float32)
Y = np.array(Y)
X = X.reshape(X.shape[0], -1)
print('splitting data')
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print('setting Model')
clf = MLPClassifier(solver='adam', alpha=0.0001, hidden_layer_sizes=(300 ,250 ,200, 150, 100, 50), random_state=42)# Experiment with different kernels (e.g., 'rbf', 'poly')
print('started training')
clf.fit(X_train, y_train)
e = time.time()
print('end training')
joblib.dump(clf,"neural_model_scikit_300(1)_nuerons.pkl")

print('testing')
y_pred = clf.predict(X_test)
print('calculating acuracy')
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(e-s)