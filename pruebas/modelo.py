from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np






tf.config.set_visible_devices([], 'GPU')

#############################################

frameWidth= 640         # CAMERA RESOLUTION
frameHeight = 480
brightness = 180
threshold = 0.75         # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################

# SETUP THE VIDEO CAMERA
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

model = load_model('puzzlebot_model.h5')

model.summary()


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


while True:

    # READ IMAGE
    success, imgOrignal = cap.read()
    
    # PROCESS IMAGE
    img = np.asarray(imgOrignal)
    img = imageProcessing(img)



    cv2.putText(imgOrignal, "CLASS: " , (20, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    # PREDICT IMAGE
    

    preds = model.predict(img)

    index = np.argmax(preds)
    probabilityValue = preds[0][index]
    label = getCalssName(index)

    




    if probabilityValue > threshold:
        cv2.putText(imgOrignal,str(index)+" "+str(label), (120, 35), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(imgOrignal, str(round(probabilityValue*100,2) )+"%", (180, 75), font, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Result", imgOrignal)
    else:
        cv2.putText(imgOrignal, 'No Traffic Sign', (120, 35), font, 0.75,
                    (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Result", imgOrignal)
        
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break

    