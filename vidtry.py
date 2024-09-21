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
l = []
capture = cv.VideoCapture(0) # replace 0 with video location
# change height and width of image
cap_w = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
cap_h = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))
# grid dimensions
grid_w = cap_w//5
grid_h = cap_h//5
# corners list
corners_list = grid_corners = [(x,y) for x in range(0,cap_w,grid_w) for y in range(0,cap_h,grid_h)]
while True:
    isTrue, frame = capture.read()
    # frame_resized = rescale(frame)
    frame = change_brightness(frame,100)
    #show grid
    i = 0
    for t_lft_corners in corners_list:
        btm_rt_corners = (t_lft_corners[0]+grid_w,t_lft_corners[1]+grid_h)
        cv.rectangle(frame,t_lft_corners,btm_rt_corners,color=(255,0,0),thickness=5)
        i+=1
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    canny = cv.Canny(gray,150,175)
    l.append(canny)
    cv.imshow('orig',frame)
    cv.imshow('Video', canny)
    # cv.imshow('Video_resized', frame_resized)
    i += 1
    addframe = 0
    if i%15==1:
        for frame in l:
            addframe+=frame
        avgframe = addframe/15
        cv.imshow('eh',avgframe)
        l = []
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows()
