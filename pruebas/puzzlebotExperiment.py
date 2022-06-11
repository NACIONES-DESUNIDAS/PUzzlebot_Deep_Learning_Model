# programa para captar video en opencv

import cv2 as cv2
from regex import D

capture = cv2.VideoCapture("puzzlebot_record.avi")
import numpy as np

def preprocessImage(img):
    # Method used to preprocess the node input image, the image processing must be divided in:
    # 1 - Resize the input image to a specified image scale factor.
    # 2 - Rotate the image if required.
    # 3 - Apply an adequate Gaussian Blur to the image, modify the filter kernel as required.
    # 4 - Return the final processed image

    ##########################################################################################################
    # TODO: Complete the class method definition using the previous description
    ##########################################################################################################

    # Your code here...
    width = int(img.shape[0]*50/100)
    height = int(img.shape[1]*50/100)
    img = cv2.resize(img,(height,width))
    img = cv2.rotate(img,cv2.ROTATE_180)
    resized = img.copy()
    img =cv2.GaussianBlur(img,(7,7),0)
    ##########################################################################################################
    return img,resized


def equalizeSV(hsv):
    H,S,V = cv2.split(hsv)
    S = cv2.equalizeHist(S)
    V = cv2.equalizeHist(S)
    equalized = cv2.merge([H,S,V])
    return equalized




def extractWhitePixels(img):

    #upperWhite = np.array([0,50,255])
    #lowerWhite = np.array([0,0,0])
    upperWhite = np.array([180,255,255])
    lowerWhite = np.array([50,0,0])         
    hsv = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2RGB)
    hsv = equalizeSV(hsv)

    filtered = cv2.inRange(hsv,lowerb=lowerWhite,upperb=upperWhite)
    masked = cv2.bitwise_and(img,img,mask=filtered)


    
    return masked

dims = 3
d = 1
e = 5

while True: # process videos frame by frame
    isTrue, img = capture.read()
    width = int(img.shape[0]*50/100)
    height = int(img.shape[1]*50/100)
    img = cv2.resize(img,(height,width))
    img = cv2.rotate(img,cv2.ROTATE_180)
    resized = img.copy()
    img =cv2.GaussianBlur(img,(7,7),0)


    #upperWhite = np.array([147,255,255])
    #lowerWhite = np.array([130,100,200])
    upperWhite = np.array([150,245,255])
    lowerWhite = np.array([120,200,200])         
    hsv = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2RGB)
    hsv = equalizeSV(hsv)

    filtered = cv2.inRange(hsv,lowerb=lowerWhite,upperb=upperWhite)
    masked = cv2.bitwise_and(img,img,mask=filtered)


    gray = cv2.cvtColor(masked,cv2.COLOR_BGR2GRAY)
    threshold,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    dilation = cv2.dilate(src=thresh,kernel=np.ones((dims,dims)) ,iterations=d)
    erotion = cv2.erode(src=dilation,kernel=np.ones((dims,dims)) ,iterations=e)

    cv2.imshow('Video',erotion)
    cv2.imshow("masked",masked)
    if cv2.waitKey(20) & 0xFF == ord('d'): # stop the video if the d key is pŕessed
        break
capture.release()
cv2.destroyAllWindows()