import cv2
import dlib
import numpy as np
import os, sys, argparse

idx = 0
data = []
input = None

def markPoint(event, x, y, flags, param):
    global idx
    global data
    global input

    if event == cv2.EVENT_LBUTTONUP:
        data.append((x, y))
        cv2.circle(input, (x, y), 3, (0,0,255), 2)
        cv2.putText(input, str(idx), (x, y+4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("Mark points", input)
        idx = idx + 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_mask_image', help='Input mask image ( Features are extracted )', required=True)
    parser.add_argument('-o', '--output_mask_data', help='Output mask data path ( *.npy )', required=True)
    args = parser.parse_args()

    input = cv2.imread(args.input_mask_image)

    cv2.imshow("Mark points", input)
    cv2.setMouseCallback("Mark points", markPoint)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if idx < 68:
        print "Insufficient points marked. Aborting.."
        sys.exit()
    else:
        np.save(args.output_mask_data, np.array(data))
        print "Mask data successfully saved to: " + args.output_mask_data
