# import numpy as np
# import cv2 as cv
# import os
# test_images = np.array([cv.imread("photos/dataset_final/battery_r0/1.png"), cv.imread("photos/dataset_final/cap_r1/1.png")])

# test_images = test_images.astype(np.float32)
# test_images = test_images.reshape(len(test_images), -1)
# print('loading model')
# svm = cv.ml.SVM.create().load('svm_model.xml')
# print('starting test')
# predictions = svm.predict(test_images)[1]
# print('test ended successfully')
# print(predictions)

import torch
from imgsticher import get_image
import cv2 as cv
# from finalneural import lable_Dict

x,y = get_image()
for k in range(len(y)):
    cv.rectangle(x,(y[k][0][0],y[k][0][1]),(y[k][0][2],y[k][0][3]),color=(255,0,0))
    print(y[k][1])


j = x
x = torch.from_numpy(x.transpose(2, 0, 1)).to(dtype=torch.float)

model = torch.load('mod.pth')
model.eval()

with torch.no_grad():
    prediction = model([x])

print(prediction)

for i in range(len(prediction[0]['boxes'])):
    num=int(prediction[0]['labels'][i])
    print(lable_Dict[num] if num<=44 else num)
    a = (round(prediction[0]['boxes'][i][0].item()),round(prediction[0]['boxes'][i][1].item()))
    b = (round(prediction[0]['boxes'][i][2].item()), round(prediction[0]['boxes'][i][3].item()))
    if prediction[0]['scores'][i]>=0.7:
        img = cv.rectangle(j,a,b,color=(255,0,0))

cv.imshow('image', img)