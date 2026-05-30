import cv2 as cv
import numpy as np
from collections import deque
def rescale(frame:cv.typing.MatLike,scale:float=0.75)->cv.typing.MatLike:
    width = int(frame.shape[1] * scale)  
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)#resises frame to 

#increase brightness
def change_brightness(img, value=30):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    v = cv.add(v,value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv.merge((h, s, v))
    img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
    return img


i = 0
j=0
l = deque(maxlen=5)
capture = cv.VideoCapture(2,cv.CAP_V4L2) # replace 0 with video location
# change height and width of image
cap_w = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
cap_h = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
print(cap_w,cap_h)

# capture.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
# grid dimensions
grid_w = cap_w//5
grid_h = cap_h//5
# corners list
corners_list = grid_corners = [(x,y) for x in range(0,cap_w,grid_w) for y in range(0,cap_h,grid_h)]
while True:
    isTrue, frame = capture.read()
    # frame = rescale(frame,0.5)
    # frame = change_brightness(frame,50)
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(
    gray,
    (5,5),
    0
    )
    # add 120x120 px box to frame in center
    box_size = 120
    center_x = cap_w // 2
    center_y = cap_h // 2
    top_left = (center_x - box_size // 2, center_y - box_size // 2)
    bottom_right = (center_x + box_size // 2, center_y + box_size // 2)
    thresh = cv.adaptiveThreshold(
        blur,
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY_INV,
        11,
        2
    )
    thresh = cv.rectangle(thresh, top_left, bottom_right, (0, 0, 0), 5)
    # canny = cv.Canny(gray,20,100)# lower and upper threshold for canny edge detection
    # l.append(canny)
    l.append(thresh)
    cv.imshow('orig',frame)
    avgframe = np.mean(l,axis=0).astype(np.uint8)
    cv.imshow('eh',avgframe)
    # cv.imshow('Video', canny)
    # cv.imshow('Video_resized', frame_resized)
    j += 1
    # addframe = 0
    # if j%5==1:
    #     avgframe = np.mean(l,axis=0).astype(np.uint8)
    #     cv.imshow('eh',avgframe)
    #     l = []
    key=cv.waitKey(20) & 0xFF
    if key==ord('d'):
        break
    if key==ord(' '):
        cv.imwrite('frame'+str(i)+'.jpg',avgframe)
        i+=1
capture.release()
cv.destroyAllWindows()