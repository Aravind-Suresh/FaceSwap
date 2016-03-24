import cv2
import dlib
import sys, argparse
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

    cv2.imwrite("img.jpg", img)

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

def swap_faces(src, dst):
    src_l = get_facial_landmarks(src)
    dst_l = get_facial_landmarks(dst)

    if src_l is None or dst_l is None:
        return None, None, None

    src_l, _, _ = src_l
    dst_l, rect, shape = dst_l

    tM = get_tm_opp(src_l[MATCH_POINTS_IDX], dst_l[MATCH_POINTS_IDX])

    mask = get_face_mask(dst, dst_l)

    mask_w = cv2.warpAffine(mask, tM[:2],
                   (src.shape[1], src.shape[0]),
                   dst = None,
                   borderMode = cv2.BORDER_TRANSPARENT,
                   flags = cv2.WARP_INVERSE_MAP)

    mask_t = np.max([get_face_mask(src, src_l), mask_w], axis = 0)

    dst_warp = cv2.warpAffine(dst, tM[:2],
                   (src.shape[1], src.shape[0]),
                   dst = None,
                   borderMode = cv2.BORDER_TRANSPARENT,
                   flags = cv2.WARP_INVERSE_MAP)

    t1 = src*(1.0 - mask_t)
    t2 = smooth_colors(src, dst_warp, src_l)
    cv2.imwrite("img.jpg", t1)
    t2 = t2*mask_t
    return (t1 + t2), rect, shape

def generateOutput(img):
    cv2.imwrite('temp.jpg', img)
    return cv2.imread('temp.jpg')

def get_face_contour_mask(rect_shape, pt_tl, shape):
    mask = np.zeros(rect_shape)
    lm = np.matrix([[pt.x - pt_tl[0], pt.y - pt_tl[1]] for pt in shape.parts()])
    hull = cv2.convexHull(lm)
    cv2.fillConvexPoly(mask, hull, color = 1)
    return np.uint8(mask)
    # return np.array([np.uint8(mask), np.uint8(mask), np.uint8(mask)]).transpose(1, 2, 0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--source_image', help='Source image ( Features are extracted )', required=True)
    parser.add_argument('-p', '--predictor', help='Predictor', required=True)
    parser.add_argument('-t', '--target_image', help='Target image ( Configuration is extracted )')
    parser.add_argument('-v', '--video', help='Mode', action='store_true')
    args = parser.parse_args()

    data_path = args.source_image
    shape_predictor_path = args.predictor

    shape_predictor = dlib.shape_predictor(shape_predictor_path)

    src = cv2.imread(data_path)

    if args.video:
        cap = cv2.VideoCapture(0)
        while(1):
            ret, frame = cap.read()
            out, rect, shape = swap_faces(src, frame)

            if out is None:
                continue
            else:
                #TODO: Take the face contour from frame and replace with swapped face
                out = generateOutput(out)
                rect_shape = (rect.bottom() - rect.top(), rect.right() - rect.left())
                scale = (1.0*np.array(rect_shape)/np.array(src.shape[:2]))
                out_r = cv2.resize(out, None, fx=scale[1], fy=scale[0])

                face_contour_mask = get_face_contour_mask(rect_shape, [rect.left(), rect.top()], shape)

                output = np.zeros(frame.shape, dtype=frame.dtype)

                cv2.imshow("out_r", out_r)
                cv2.imshow("mask", 255*face_contour_mask)
                face_contour_mask = 255*face_contour_mask

                frame_fg = cv2.bitwise_and(out_r, out_r, mask=face_contour_mask)
                frame_bg = cv2.bitwise_and(
                    frame[rect.top():rect.bottom(), rect.left():rect.right()],
                    frame[rect.top():rect.bottom(), rect.left():rect.right()],
                    mask=(255 - face_contour_mask)
                    )
                output[rect.top():rect.bottom(), rect.left():rect.right()] = frame_fg + frame_bg

                cv2.imshow("Cam", output)
                k = cv2.waitKey(1)

                if k == 27:
                    cv2.destroyAllWindows()
                    cap.release()
                    break
    else:
        if args.target_image is None:
            print "Target image path missing."
            sys.exit()
        target_path = args.target_image
        dst = cv2.imread(target_path)
        out, _, _ = swap_faces(src, dst)
        cv2.imshow("Output", generateOutput(out))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
