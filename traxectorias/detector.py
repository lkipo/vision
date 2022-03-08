import cv2 as cv
from cv2 import VideoCapture
import numpy as np

kernel = np.ones((3,3),np.uint8)

cap = VideoCapture(0)

# Substractor de fondo para camara estable
object_detector = cv.createBackgroundSubtractorMOG2(history=50, varThreshold=10)

while True:
    
    ret, frame = cap.read()
    #frame=cv.resize(frame,(1366,768))

    mascara = object_detector.apply(frame)

    bgs = cv.erode(mascara,kernel,iterations = 1)
    bgs = cv.medianBlur(bgs,3)
    bgs = cv.dilate(bgs,kernel,iterations=2)


    # Sacamos sombras da máscara
    #_, mascara = cv.threshold(mascara, 254, 255, cv.THRESH_BINARY)
    # difuminacion = cv.medianBlur(mascara, 7)
    contornos, _ = cv.findContours(bgs, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contornos: 
        area = cv.contourArea(cnt)
        if area > 200:
            x, y, w, h = cv.boundingRect(cnt)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            #cv.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

    cv.imshow("Frame", frame)
    cv.imshow("MOG2", bgs)

    key = cv.waitKey(100)

    if key == 115:
        break

cap.release()
cv.destroyAllWindows() 