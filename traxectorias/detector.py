import cv2
from cv2 import VideoCapture
import numpy as np

kernel = np.ones((3,3),np.uint8)

cap = VideoCapture(0)

# Substractor de fondo para camara estable
object_detector = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=10)

while True:
    
    ret, frame = cap.read()
    #frame=cv2.resize(frame,(1366,768))

    mascara = object_detector.apply(frame)

    bgs = cv2.erode(mascara,kernel,iterations = 1)
    bgs = cv2.medianBlur(bgs,3)
    bgs = cv2.dilate(bgs,kernel,iterations=2)


    # Sacamos sombras da máscara
    #_, mascara = cv2.threshold(mascara, 254, 255, cv2.THRESH_BINARY)
    # difuminacion = cv2.medianBlur(mascara, 7)
    contornos, _ = cv2.findContours(bgs, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contornos: 
        area = cv2.contourArea(cnt)
        if area > 200:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            #cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("MOG2", bgs)

    key = cv2.waitKey(100)

    if key == 115:
        break

cap.release()
cv2.destroyAllWindows() 