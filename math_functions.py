import cv2
import numpy as np
from numba import njit

import mediapipe as mp

#njit pentru procesare mai C-like (imi pare rau ca am folosit sarpele...)

#pentru detalii, check documentatie. memoization, pe scurt
@njit(cache=True)
def compute_integral_image(image):
    rows, cols = image.shape
    integral_image = np.zeros((rows + 1, cols + 1), dtype=np.float64)

    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            integral_image[r, c] = (
                image[r-1, c-1]
                + integral_image[r-1, c]
                + integral_image[r, c-1]
                - integral_image[r-1, c-1]
            )

    return integral_image

#profitam de memoiz
@njit(cache=True)
def compute_sum_of_region(integral_image,row1,col1,row2,col2):
    whole_region = integral_image[row2+1,col2+1]
    subtract_region1 = integral_image[row2+1,col1]
    subtract_region2 = integral_image[row1,col2+1]
    add_region = integral_image[row1,col1]

    return whole_region - subtract_region1 - subtract_region2 + add_region

def rgb_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64)

#mi am definit eu desenator:)
@njit(cache=True)
def place_rectangles_in_image(image, row1, col1, row2, col2):

    img_rows, img_cols, _ = image.shape

    if row1 < 0: row1 = 0
    if col1 < 0: col1 = 0

    if row2 >= img_rows: row2 = img_rows - 1
    if col2 >= img_cols: col2 = img_cols - 1


    for r in range(row1, row2 + 1):
        image[r, col1] = [255, 0, 0]
        image[r, col2] = [255, 0, 0]

    for c in range(col1, col2 + 1):
        image[row1, c] = [255, 0, 0]
        image[row2, c] = [255, 0, 0]

#am incercat, in prima faza, sa dau resize la imagine... ups
@njit(cache=True)
def resize_bilinear(image, scale_factor):
    inv_scale = 1/scale_factor
    h, w = image.shape
    new_h, new_w = int(h * inv_scale), int(w * inv_scale)
    result = np.zeros((new_h, new_w), dtype=image.dtype)

    for y in range(new_h):
        src_y = y * scale_factor
        src_x = 0.0
        for x in range(new_w):

            x0 = int(src_x)
            y0 = int(src_y)

            x1 = min(x0 + 1, w - 1)
            y1 = min(y0 + 1, h - 1)

            dx = src_x - x0 #src_x si src_y o sa fie float. impartim la 1.25
            dy = src_y - y0

            result[y, x] = (
                (1 - dx) * (1 - dy) * image[y0, x0] +
                dx * (1 - dy) * image[y0, x1] +
                (1 - dx) * dy * image[y1, x0] +
                dx * dy * image[y1, x1]
            )
            src_x += scale_factor

    return result


@njit(cache=True)
def build_gamma_lut(gamma=2.2):
    lut = np.empty(256, dtype=np.uint8)

    inv = 1.0 / 255.0

    for i in range(256):
        lut[i] = np.uint8(
            255.0 * ((i * inv) ** gamma)
        )

    return lut

@njit(cache=True)
def gamma_correction_lut(image, lut):
    #lut = build_gamma_lut(gamma)

    out = np.empty_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            out[i, j] = lut[image[i, j]]

    return out
#un feature, pozitia de unde incepe window ul
@njit(cache=True)
def evaluate_feature(integral_image, features, numb_features, i, j, scale):
    value = 0.0
    total_area = 0.0

    for k in range(numb_features):
        x = features[k, 0]
        y = features[k, 1]
        w = features[k, 2]
        h = features[k, 3]

        # se scaleaza dimensiunea. mutam si coordonatele si inmultim si w si h
        x1 = int(x * scale)
        y1 = int(y * scale)

        w_scaled = max(1, int(w * scale))
        h_scaled = max(1, int(h * scale))
        weight = features[k, 4]
        #poz de unde incepe + mutare fata de unde ar fi feature ul in window
        row1 = i + y1
        col1 = j + x1

        row2 = row1 + h_scaled - 1
        col2 = col1 + w_scaled - 1

        region_sum = compute_sum_of_region(integral_image, row1, col1, row2, col2)
        value += weight * region_sum

        total_area += w_scaled * h_scaled

    return value / total_area

thresholds = np.array([
    5,
    5,
    5,
    5,
    5
])

@njit(cache=True)
def check_haar_features(integral_image, i, j,haar_features, haar_counts, scale):
    #daca nu trec toate feature urile de 0.3 (threshold) returnam -1. altfel: suma scorurilor
    f1 = evaluate_feature(integral_image, haar_features[0], haar_counts[0], i, j, scale)
    if f1 < thresholds[0]:
        return -1

    f2 = evaluate_feature(integral_image, haar_features[1], haar_counts[1], i, j, scale)
    if f2 < thresholds[1]:
        return -1

    f3 = evaluate_feature(integral_image, haar_features[2], haar_counts[2], i, j, scale)
    if f3 < thresholds[2]:
        return -1

    f4 = evaluate_feature(integral_image, haar_features[3], haar_counts[3], i, j, scale)
    if f4 < thresholds[3]:
        return -1

    f5 = evaluate_feature(integral_image, haar_features[4], haar_counts[4], i, j, scale)
    if f5 < thresholds[4]:
        return -1
    score = f1 + f2 + f3 + f4 + f5
    return score

def non_maxima_suppression(pq,iou_threshold=0.3,containment_threshold=0.3):#intersection over union
    boxes = []

    while not pq.empty():
        score, (x, y, w, h) = pq.get()
        area = w * h
        boxes.append((area, [x, y, w, h]))#TODO: type of search to perform

    boxes.sort(key=lambda x: x[0], reverse=True)#highest area - score

    keep = []

    while boxes:

        best_score, best_box = boxes.pop(0)

        bx, by, bw, bh = best_box
        # if bw < 100 or by < 100:
        #     continue
        keep.append(best_box)

        remaining = []

        best_area = bw * bh

        for score, box in boxes:

            ox, oy, ow, oh = box

            other_area = ow * oh

            #colturi la cutii
            x1 = max(bx, ox)
            y1 = max(by, oy)
            x2 = min(bx + bw, ox + ow)
            y2 = min(by + bh, oy + oh)

            inter_w = max(0, x2 - x1)
            inter_h = max(0, y2 - y1)

            intersection = inter_w * inter_h

            if intersection == 0:#disjuncte
                remaining.append((score, box))
                continue
            #calculam iou
            union = best_area + other_area - intersection
            iou = intersection / union
            #cat din cutia mai mica e acoperita
            contained_ratio = intersection / min(best_area, other_area)
            #nu desenam daca IOU e prea mic sau contain prea mare
            suppress = (
                iou > iou_threshold or
                contained_ratio > containment_threshold
            )
            #altfel, punem in lista
            if not suppress:
                remaining.append((score, box))

        boxes = remaining

    return keep


#rotate fata

import math



def align_face(face_roi, eye_detector):

    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)#detector needs grayscale

    eyes = eye_detector.detectMultiScale(gray)

    if len(eyes) < 2:
        return face_roi

    eyes = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[:2]#selectam cele 2 cele mai sigure locatii - cele mai mari W*H (0-x,1-y,2-w,3-h)

    eye1, eye2 = eyes

    # comparam x-i
    if eye1[0] < eye2[0]:
        left_eye, right_eye = eye1, eye2
    else:
        left_eye, right_eye = eye2, eye1

    #matematic. centrul efectiv ar depinde si de tip de ochi...
    left_center = (
        left_eye[0] + left_eye[2] // 2,
        left_eye[1] + left_eye[3] // 2
    )

    right_center = (
        right_eye[0] + right_eye[2] // 2,
        right_eye[1] + right_eye[3] // 2
    )

    dy = right_center[1] - left_center[1]
    dx = right_center[0] - left_center[0]

    angle = np.degrees(np.arctan2(dy, dx))

    # rotate
    h, w = face_roi.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    aligned = cv2.warpAffine(
        face_roi,
        M,
        (w, h),
        flags=cv2.INTER_LANCZOS4 #INTER_LANCZOS4 INTER_CUBIC
    )

    return aligned

