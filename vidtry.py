import cv2 as cv
import numpy as np
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
capture = cv.VideoCapture(0) # replace 0 with video location
while True:
    i += 1
    isTrue, frame = capture.read()
    # frame_resized = rescale(frame)
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    canny = cv.Canny(gray,150,175)
    cv.imshow('orig',frame)
    cv.imshow('Video', canny)
    # cv.imshow('Video_resized', frame_resized)
    if i%60==1:
        pass
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows()