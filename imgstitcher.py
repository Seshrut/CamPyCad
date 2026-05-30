import cv2 as cv
import numpy as np
import os
import random
def get_image():

    data_dir = 'photos\\dataset_final'
    labels = os.listdir(data_dir)#battery, resistor, etc
    row = np.zeros((120,1,3))
    picture = np.zeros((1,700,3))
    final=[]#--->[(x,y),(x+120,y+120),label]
    y = 0
    #loop
    for i in range(5):
        x = 0
        for j in range(5):
            # one of labels (eg. battery)
            label = labels[random.randint(0,len(labels)-1)]
            labelpos = os.path.join(data_dir,label)
            # one of image of label
            img = os.listdir(labelpos)[random.randint(0,len(os.listdir(labelpos))-1)]
            # converting to cv image
            img = cv.imread(os.path.join(labelpos,img))
            # random gap to insert in left
            rand_x_gap = random.randint(10,20)
            L_img = np.concatenate((np.zeros((120,rand_x_gap,3)),img),axis=1)#add to left
            x += rand_x_gap
            # cv.rectangle(L_img,(x,y),(x+120,y+120),(0,0,255),3)
            final.append([[x,y,x+120,y+120],label])
            x+=120
            if str(row) == str(np.zeros((120,1,3))):
                row = L_img
            else:
                row = np.concatenate((row,L_img),axis=1)
        if str(picture) == str(np.zeros((120,1,3))):
            picture = row
        else:
            remaining = 700-(x)
            if remaining != 0:
                row = np.concatenate((row,np.zeros((120,remaining,3))),axis=1)
            picture = np.concatenate((picture,row),axis=0)
        row = np.zeros((120,1,3))
        rand_y_gap = random.randint(10,20)
        picture = np.concatenate((picture, np.zeros((rand_y_gap,700,3))), axis=0)
        y+=rand_y_gap+700
    remaining_y = 700 - picture.shape[0]
    picture = np.concatenate((picture,np.zeros((remaining_y,700,3))),axis=0)
    return picture,final
    # imgdir, x0y0 x1y1 component gawar grameen