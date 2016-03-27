import cv2
import dlib
import numpy as np
import os, sys, argparse

idx = 0
data = {}
input = None

def markPoint(event, x, y, flags, param):
    global idx
    global data
    global input

    if event == cv2.EVENT_LBUTTONUP:
        data[str(idx)] = [x, y]
        cv2.circle(input, (x, y), 3, (0,0,255), 2)
        cv2.putText(input, str(idx), (x, y+4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("Mark points", input)
        idx = idx + 1

        if idx == 68:
            print data
            cv2.destroyAllWindows()
            sys.exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_image', help='Input image ( Features are extracted )', required=True)
    parser.add_argument('-p', '--predictor', help='Predictor', required=True)
    args = parser.parse_args()

    input = cv2.imread(args.input_image)
    shape_predictor = dlib.shape_predictor(args.predictor)
    face_detector = dlib.get_frontal_face_detector()

    cv2.imshow("Mark points", input)
    cv2.setMouseCallback("Mark points", markPoint)
    cv2.waitKey(0)
    print data
    cv2.destroyAllWindows()
