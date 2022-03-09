import cv2 as cv
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="C")
parser.add_argument('-l', '--initial_location', action='store')
parser.set_defaults(initial_location = '')
options = parser.parse_args()

global location

if not(options.initial_location == ''):
    print(options)
    x, y, w, h = eval(str(options.initial_location))
    print(x)
    print(y)
    print(w)
    print(h)
    location = True

# Cambiar esto ↓
track_window = (x, y, w, h)

cap = cv.VideoCapture(0)

ret, frame = cap.read()

# Desde aqui ata a 33 está copiado xd
roi = frame[y:y+h, x:x+w]
hsv_roi =  cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )

while True:
    # Lemos a info da camara
    ret, frame = cap.read()

    if ret == False:
        break

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1) # nin idea de que fai esto

    ret, track_window = cv.CamShift(dst, track_window, term_crit) # A saber que fai esto

    # Mostramos a imaxe
    pts = cv.boxPoints(ret)
    pts = np.int0(pts)
    img2 = cv.polylines(frame,[pts],True, 255,2)
    cv.imshow('img2',img2)
    key = cv.waitKey(1)

    # si presionamos s paramos a execucion do progrma
    if key == 115:
        break