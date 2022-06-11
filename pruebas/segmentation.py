# programa para captar video en opencv

import cv2 as cv2

capture = cv2.VideoCapture("puzzlebot_record2.avi")
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
    #img =cv2.GaussianBlur(img,(7,7),0)
    ##########################################################################################################
    return img,resized


def getHoughCircles(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    circles_img = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,100,param1=50,param2=30,minRadius=15,maxRadius=30)
    #circles_img = np.uint16(np.around(circles_img))
    return circles_img
    """
    for i in circles_img[0,:]:
        cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
        cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
    return img, circles_img
    """





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


def getSegementedImages(circles,img):
    images = list()
    outputImg = img.copy()
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for circle in circles:
            x = circle[0]
            y = circle[1]
            rad = circle[2] + 10
            topLeftCorner = (x-(rad),y-(rad))
            bottomRightCorner =  (x+(rad),y+(rad))
            outputImg = cv2.rectangle(outputImg,topLeftCorner,bottomRightCorner,(255,0,0),5)
            segemented = img[y-rad:y+rad,x-rad:x+rad,:]
            #print(segemented.shape)
            try:
                cv2.imshow("video",segemented)
            except:
                continue

    return outputImg




dims = 3
d = 1
e = 5

while True: # process videos frame by frame
    isTrue, img = capture.read()

    # preprocesamiento
    width = int(img.shape[0]*50/100)
    height = int(img.shape[1]*50/100)
    img = cv2.resize(img,(height,width))
    img = cv2.rotate(img,cv2.ROTATE_180)

    circles = getHoughCircles(img)
    images = getSegementedImages(circles=circles,img=img)
    #cv2.imshow("video",images)



    if cv2.waitKey(20) & 0xFF == ord('d'): # stop the video if the d key is pŕessed
        break
capture.release()
cv2.destroyAllWindows()