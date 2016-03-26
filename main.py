import cv2
import dlib
import os, sys, argparse
import numpy as np

BLUR_FRACTION = 0.6
BLUR_AMOUNT = 11

JAW_IDX = list(np.arange(0, 17))
FACE_IDX = list(np.arange(17, 68))
MOUTH_IDX = list(np.arange(48, 61))

RIGHT_EYE_IDX = list(np.arange(36, 42))
LEFT_EYE_IDX = list(np.arange(42, 48))

NOSE_IDX = list(np.arange(27, 35))
LEFT_EYE_BROW_IDX = list(np.arange(22, 27))
RIGHT_EYE_BROW_IDX = list(np.arange(17, 22))

MATCH_POINTS_IDX = LEFT_EYE_BROW_IDX + RIGHT_EYE_BROW_IDX + LEFT_EYE_IDX + RIGHT_EYE_IDX + NOSE_IDX + MOUTH_IDX
OVERLAY_POINTS_IDX = [
    LEFT_EYE_IDX + RIGHT_EYE_IDX + LEFT_EYE_BROW_IDX + RIGHT_EYE_BROW_IDX,
    NOSE_IDX + MOUTH_IDX,
]

face_detector = dlib.get_frontal_face_detector()
shape_predictor = None

def get_facial_landmarks(img):
    rects = face_detector(img, 1)

    if len(rects) == 0:
        print "No faces"
        return None

    rect = rects[0]
    shape = shape_predictor(img, rect)
    return np.matrix([[pt.x, pt.y] for pt in shape.parts()]), rect, shape

def get_face_mask(img, img_l):
    img = np.zeros(img.shape[:2], dtype = np.float64)

    for idx in OVERLAY_POINTS_IDX:
        cv2.fillConvexPoly(img, cv2.convexHull(img_l[idx]), color = 1)

    img = np.array([img, img, img]).transpose((1, 2, 0))
    img = (cv2.GaussianBlur(img, (BLUR_AMOUNT, BLUR_AMOUNT), 0) > 0) * 1.0
    img = cv2.GaussianBlur(img, (BLUR_AMOUNT, BLUR_AMOUNT), 0)

    return img

def smooth_colors(src, dst, src_l):
    blur_amount = BLUR_FRACTION * np.linalg.norm(np.mean(src_l[LEFT_EYE_IDX], axis = 0) - np.mean(src_l[RIGHT_EYE_IDX], axis = 0))
    blur_amount = (int)(blur_amount)

    if blur_amount % 2 == 0:
        blur_amount += 1

    src_blur = cv2.GaussianBlur(src, (blur_amount, blur_amount), 0)
    dst_blur = cv2.GaussianBlur(dst, (blur_amount, blur_amount), 0)

    dst_blur += (128 * ( dst_blur <= 1.0 )).astype(dst_blur.dtype)

    return (np.float64(dst) * np.float64(src_blur)/np.float64(dst_blur))

def get_tm_opp(pts1, pts2):
    pts1 = np.float64(pts1)
    pts2 = np.float64(pts2)

    m1 = np.mean(pts1, axis = 0)
    m2 = np.mean(pts2, axis = 0)

    pts1 -= m1
    pts2 -= m2

    std1 = np.std(pts1)
    std2 = np.std(pts2)
    std_r = std2/std1

    pts1 /= std1
    pts2 /= std2

    U, S, V = np.linalg.svd(np.transpose(pts1) * pts2)

    R = np.transpose(U * V)

    return np.vstack([np.hstack((std_r * R,
        np.transpose(m2) - std_r * R * np.transpose(m1))), np.matrix([0.0, 0.0, 1.0])])

def getRectShape(rect):
    return (rect.bottom() - rect.top(), rect.right() - rect.left())

def toRoi(rect):
    return dlib.rectangle(0, 0, rect.right() - rect.left(), rect.bottom() - rect.top())

def warp_image(img, tM, shape):
    out = np.zeros(shape, dtype=img.dtype)
    cv2.warpAffine(img,
                   tM[:2],
                   (shape[1], shape[0]),
                   dst=out,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return out

def swap_faces(tmpl_, input_):
    tmpl_fl = get_facial_landmarks(tmpl_)
    input_fl = get_facial_landmarks(input_)

    if tmpl_fl is None or input_fl is None:
        return None

    tmpl_fl = list(tmpl_fl)
    input_fl = list(input_fl)

    # {tmpl, input}_fl : [ landmarks, rect, shape ]

    tmpl = tmpl_

    input = input_[input_fl[1].top():input_fl[1].bottom(), input_fl[1].left():input_fl[1].right()]
    input_fl[0] -= [input_fl[1].left(), input_fl[1].top()]

    tM = get_tm_opp(tmpl_fl[0][MATCH_POINTS_IDX], input_fl[0][MATCH_POINTS_IDX])
    mask = get_face_mask(input, input_fl[0])
    mask_w = warp_image(mask, tM, tmpl.shape)

    mask_t = np.max([get_face_mask(tmpl, tmpl_fl[0]), mask_w], axis = 0)

    input_warp = warp_image(input, tM, tmpl.shape)

    t1 = tmpl*(1.0 - mask_t)
    t2 = smooth_colors(tmpl, input_warp, tmpl_fl[0])
    t2 = t2*mask_t

    return (t1+t2)

def generateOutput(img):
    cv2.imwrite('temp.jpg', img)
    out = cv2.imread('temp.jpg')
    os.remove('temp.jpg')
    return out

# TODO: Modify this method to get a better face contour mask
def get_face_contour_mask(rect_shape, pt_tl, shape):
    mask = np.zeros(rect_shape)
    lm = np.matrix([[pt.x - pt_tl[0], pt.y - pt_tl[1]] for pt in shape.parts()])
    temp1 = RIGHT_EYE_BROW_IDX
    temp1.reverse()
    temp2 = LEFT_EYE_BROW_IDX
    temp2.reverse()
    hull = lm[JAW_IDX + temp2 + temp1]
    cv2.drawContours(mask, [hull], 0, 255 , -1)
    return np.uint8(mask)

def swap_faces_wrap(frame, args):
    input = args[0]
    out_ = swap_faces(frame, input)
    if out_ is None:
        return None
    out = generateOutput(out_)
    return out

def videoize(func, args, src = 0, win_name = "Cam", delim_wait = 1, delim_key = 27):
    cap = cv2.VideoCapture(src)
    while(1):
        ret, frame = cap.read()
        out = func(frame, args)
        if out is None:
            continue
        cv2.imshow(win_name, out)
        k = cv2.waitKey(delim_wait)

        if k == delim_key:
            cv2.destroyAllWindows()
            cap.release()
            return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_image', help='Input image ( Features are extracted )', required=True)
    parser.add_argument('-p', '--predictor', help='Predictor', required=True)
    parser.add_argument('-v', '--video', help='Mode', action='store_true')
    parser.add_argument('-t', '--template_image', help='Template image ( Face template )')
    parser.add_argument('-o', '--output_image', help='Output image path')
    args = parser.parse_args()

    input = cv2.imread(args.input_image)
    shape_predictor = dlib.shape_predictor(args.predictor)

    if args.video:
        videoize(swap_faces_wrap, [input])
    else:
        if args.template_image is None:
            print "Template image required."
            sys.exit()
        tmpl = cv2.imread(args.template_image)
        out_ = swap_faces(tmpl, input)
        if out_ is None:
            print "No faces detected."
            sys.exit()
        out = generateOutput(out_)
        cv2.imshow("out", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
