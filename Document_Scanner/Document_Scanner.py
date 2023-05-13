import cv2 as c
import numpy as np
width=480
height=690
video=c.VideoCapture(1)
# (1088, 608)
video.set(3, width)
video.set(4,height)
# video.set(10,150)
def order(point):
    point=point.reshape((4,2))
    newpoint=np.zeros((4,1,2),np.int32)
    add=point.sum(1)
    newpoint[0]=point[np.argmin(add)]
    newpoint[3]=point[np.argmax(add)]
    diff=np.diff(point,axis=1)
    newpoint[1]=point[np.argmin(diff)]
    newpoint[2] = point[np.argmax(diff)]
    return newpoint

def getWarp(im,big):
    # print(big.shape)
    big=order(big)
    point1=np.float32(big)
    point2=np.float32([[0,0], [width,0], [0,height], [width, height]])
    merge=c.getPerspectiveTransform(point1,point2)
    output=c.warpPerspective(im,merge,(480,640))
    return output
def contours(im):
    big = np.array([])
    max = 0
    contour, h = c.findContours(dilation, c.RETR_EXTERNAL, c.CHAIN_APPROX_NONE)
    for con in contour:
        area = c.contourArea(con)
        if area > 5000:
            lenght = c.arcLength(con, True)
            approx = c.approxPolyDP(con, 0.05 * lenght, True)
            max = len(approx)
            if area > max and len(approx) == 4:
                big = approx
                max = area
        c.drawContours(image, big, -1, (0, 0, 255), 20)
        c.imshow('document',image)
    return big

while True:
    s,image=video.read()
    imagegray = c.cvtColor(image, c.COLOR_BGR2GRAY)
    blur=c.GaussianBlur(imagegray,(5,5),1)
    canny = c.Canny(blur,200,250)
    kernal=np.ones((5,5))
    dilation=c.dilate(canny,kernal,iterations=2)
    erode=c.erode(dilation,kernal,iterations=1)
    big=contours(erode)
    if len(big)!=0:
        wa= getWarp(image, big)
        c.imshow('Output', wa)
    if(c.waitKey(1) & 0xFF==ord('z')):
        break