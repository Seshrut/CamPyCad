import cv2 as cv
import numpy as np
#image
'''
img = cv.imread('./photos/cat (Small).jpg')# returns matrix of pixels

cv.imshow('cat',mat=img)#display the image

cv.waitKey(0)#wait infinitely long
'''
'''
#video
capture = cv.VideoCapture(0) # replace 0 with video location
# replacing 0 with file will throw (-215 assertion failed) at the end of the file
while True:
    isTrue, frame = capture.read()
    cv.imshow('Video', frame)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows()
cv.waitKey(0)
'''
#rescale a frame
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
'''
capture = cv.VideoCapture("E:\\HDD Backup\\E\\dad pendrive\\MP4\\VID-20180916-WA0003.mp4") # replace 0 with video location
while True:
    isTrue, frame = capture.read()
    frame_resized = rescale(frame)
    cv.imshow('Video', frame)
    cv.imshow('Video_resized', frame_resized)
    if cv.waitKey(20) & 0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows()'''

#For edge detection 
img = rescale(cv.imread('photos/pgcir.jpg'),0.5)
cv.imshow('orig',img)
## BGR2GRAY
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

# laplacion
lap = cv.Laplacian(gray,cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('laplacian',lap)
#sobel
sobelx = cv.Sobel(gray,cv.CV_64F,1,0)
sobely = cv.Sobel(gray,cv.CV_64F,0,1)
sobelcomb = cv.bitwise_or(sobelx,sobely)
cv.imshow('x',sobelx)
cv.imshow('y',sobely)
cv.imshow('comb',sobelcomb)
#canny 
canny = cv.Canny(change_brightness(img,150),150,175)
cv.imshow('canny',canny)
cv.waitKey(0)