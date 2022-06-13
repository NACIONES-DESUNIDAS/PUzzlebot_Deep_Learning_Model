# programa para captar video en opencv

import cv2 as cv2
from tensorflow.keras.models import load_model


capture = cv2.VideoCapture("no_speed.avi")

model = load_model("puzzlebot_model.h5")
model.summary()
import numpy as np


IMG_SHAPE = (150,150,1)
IMG_TUPPLE_SHAPE = (150,150)
BATCH_SIZE= 128
dims = 150

def imageProcessing(image):
    image = image.astype(np.uint8)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.resize(gray,IMG_TUPPLE_SHAPE)
    gray = gray.reshape(1,dims,dims,1)
    return gray.astype(np.float64)/255


def getCalssName(classNo):
    labels = {0: 'stop_signal', 1: 'aplastame', 2: 'right_signal', 3: 'left_signal', 4: 'up_signal', 5: 'around_signal'}
    return labels[classNo]





def getHoughCircles(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    circles_img = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,100,param1=50,param2=30,minRadius=20,maxRadius=30)
    #circles_img = np.uint16(np.around(circles_img))
    return circles_img


# Logica de loslabelsque se mandan a nagegacion


class SignalDetector:
    def __init__(self):
        # classatributes
        self.lastLabel = None
        self.labelCounter = 0
        self.labelThreshold = 4


    def pubLabelFlag(self,label):
        if self.lastLabel == label:
            self.labelCounter += 1
        if self.labelCounter > self.labelThreshold:
            self.labelCounter = 0
            flag = True if self.lastLabel != "not_found" else False
            print(self.lastLabel,flag)

        self.lastLabel = label


def getSegementedImages(circles,img,d):


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
            try:
            
                image = segemented.astype(np.uint8)
                gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                gray = cv2.resize(gray,IMG_TUPPLE_SHAPE)
                cv2.imshow("video",gray)
                gray = gray.reshape(1,150,150,1)
                gray = gray.astype(np.float64)/255
                prediction = model.predict(gray)
                index = np.argmax(prediction)
                prob = prediction[0][index]
                label = getCalssName(index) if prob > 0.9 else "not_found"
                #d.pubLabelFlag(label)
                print(label)
            except:
                continue
            #cv2.imshow("video",outputImg)

    return outputImg







dims = 3
d = 1
e = 5


d = SignalDetector()


while True: # process videos frame by frame

    isTrue, img = capture.read()
    # private atributes
    # preprocesamiento
    width = int(img.shape[0]*50/100)
    height = int(img.shape[1]*50/100)
    img = cv2.resize(img,(height,width))
    img = cv2.rotate(img,cv2.ROTATE_180)

    circles = getHoughCircles(img)
    images = getSegementedImages(circles=circles,img=img,d=d)
    #cv2.imshow("video",images)



    if cv2.waitKey(20) & 0xFF == ord('d'): # stop the video if the d key is pŕessed
        break
capture.release()
cv2.destroyAllWindows()