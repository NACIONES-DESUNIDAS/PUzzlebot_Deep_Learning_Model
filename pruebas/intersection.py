# programa para captar video en opencv

"""
Modelo en octavio 2
"""

from pickle import TRUE
import cv2 as cv2


capture = cv2.VideoCapture("puzzlebot_record10.avi")

import numpy as np


def imagePreprocessing(img):
    scale = 50
    width = int(img.shape[0]*scale/100)
    height = int(img.shape[1]*scale/100)
    img = cv2.resize(img,(height,width))
    rot = cv2.rotate(img,cv2.ROTATE_180)

    gray = cv2.cvtColor(rot,cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray =cv2.GaussianBlur(gray,(11,11),0)
    return gray,rot


def sliceImage(img):
    h = img.shape[0]
    return img[int(h*0.4):,:]



def edgeDetection(img):


    filterC = list()
    minY = 35
    maxY = 65
    minArea = 250
    maxArea = 600
    contours,hierchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  



    #contours = [contour for contour in contours if cv2.contourArea(contour) < 600 and cv2.contourArea(contour) > 250] 
    #  (216, 640, 3)
    for contour in contours:

        try:
            M = cv2.moments(contour)
            y = int(M["m01"]/M["m00"])
        except:
            continue


        app = False
        app2 = False


        area = cv2.contourArea(contour)
        if area < maxArea and area > minArea:
            app = True
        
        
        if y > minY and y < maxY:
            app2 = True

        if app and app2:
            filterC.append(contour)

    
    return filterC
        


        


def createImageMask(image,lowerBound,upperBound):
    return  cv2.inRange(image, lowerBound, upperBound)


def thresholdImg(img):
    retval, binary = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    return binary



THRESHOLD = 3

while True: # process videos frame by frame

    isTrue, img = capture.read()

    gray,frameP = imagePreprocessing(img)


    slicedGray = sliceImage(gray)
    oriSliced = sliceImage(frameP)

    threshold = thresholdImg(slicedGray)

    edges = edgeDetection(threshold)

    contourImage = cv2.drawContours(oriSliced,edges,-1,(0,0,255),2)

    flag = True if len(edges) > THRESHOLD else False

    print(flag)

    cv2.imshow("grayscale",threshold)
    cv2.imshow("color",contourImage)

    if cv2.waitKey(20) & 0xFF == ord('d'): # stop the video if the d key is pŕessed
        break
capture.release()
cv2.destroyAllWindows()

