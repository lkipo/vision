import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description="C")
parser.add_argument('-l', '--initial_location', action='store')
parser.set_defaults(initial_location = '')
options = parser.parse_args()

global location

if not(options.initial_location == ''):
    print(options)
    x, y, w, h = eval(str(options.initial_location))
    print(x, y, w, h)
    location = True

cap = cv.VideoCapture(0)

ret, frame = cap.read()


while True:
    # Lemos a info da camara
    ret, frame = cap.read()

    if ret == False:
        break



    # Mostramos a imaxe
    cv.imshow("camara", frame)
    key = cv.waitKey(1)

    # si presionamos s paramos a execucion do progrma
    if key == 115:
        break