
from queue import PriorityQueue

import cv2

import math_functions
from haar_masks import haar_features, haar_counts

import numpy as np

def naive_viola_and_jones(image, scale_factor):
    pq = PriorityQueue()

    base_window = 24
    step        = 4
    height, width = image.shape

    # computed integral image at start. stop changing image, change window
    integral_image = math_functions.compute_integral_image(image)
    gray_bgr = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    scale = 1.0
    while True:
        scaled_window = int(base_window * scale)
        scaled_step   = max(4, scaled_window // 6)  #scale step
        #scaled_step = step

        if scaled_window >= height or scaled_window >= width: # stop if window gets too big
            break

        #slide the window across the image
        for i in range(0, height - scaled_window, scaled_step):
            for j in range(0, width - scaled_window, scaled_step):

                #compute the score for this position and this scale. it needs to be passed the features and counts to make njit work
                score = math_functions.check_haar_features(
                    integral_image, i, j, haar_features, haar_counts, scale
                )
                if score == -1:#ca sa se afiseze -1 :)
                    score = -5

                x, y, w, h = j, i, scaled_window, scaled_window
                if height / scaled_window < 2: #to only show the last stage of the pipeline. 1.25^13*24 => 436.55

                    base_visualize = gray_bgr.copy() #copy the gray image
                    #show: current window, fail / pass + score, current overall score, size,
                    # JUST ONE FEATURE (place the feature once => removal...)
                    cv2.rectangle(base_visualize, (x, y), (x + w, y + h), (255, 255, 0), 1) #window cu cyan. punctul stanga sus si dreapta jos
                    cv2.putText(base_visualize,
                                f"score={score:.2f}  scale={scale:.2f}  win={scaled_window}px",
                                (5, height - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

                    for feat_idx in range(5):#5 features. process each of them

                        feat_val = math_functions.evaluate_feature(
                            integral_image,
                            haar_features[feat_idx],
                            haar_counts[feat_idx],
                            i, j, scale
                        )
                        passed = feat_val > math_functions.thresholds[feat_idx]#compare with the thresholds

                        # bGr si bgR
                        label_color = (0, 255, 0) if passed else (0, 0, 255)

                        #scor per feature
                        cv2.putText(base_visualize,
                                    f"f{feat_idx + 1}: {feat_val:+.3f} {'PASS' if passed else 'FAIL'}",
                                    (5, 15 + feat_idx * 16),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, label_color, 1)
                        #ce e common la fiecare: features + scor
                        vis = base_visualize.copy()
                        for k in range(haar_counts[feat_idx]):#display la feature
                            fx = haar_features[feat_idx][k, 0]
                            fy = haar_features[feat_idx][k, 1]
                            fw = haar_features[feat_idx][k, 2]
                            fh = haar_features[feat_idx][k, 3]
                            fw8 = haar_features[feat_idx][k, 4]

                            r1 = int(i + fy * scale)
                            c1 = int(j + fx * scale)
                            r2 = int(i + (fy + fh) * scale)
                            c2 = int(j + (fx + fw) * scale)

                            rect_color = (0, 255, 0) if fw8 > 0 else (0, 0, 255)  # pos part (+) is green. - red

                            overlay = vis.copy()
                            cv2.rectangle(overlay, (c1, r1), (c2, r2), rect_color, -1)
                            cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
                            cv2.rectangle(vis, (c1, r1), (c2, r2), rect_color, 1)

                        cv2.imshow("haar debug", vis)

                        cv2.waitKey(0)

                if score > 25:
                    pq.put((-score, (x, y, w, h)))

        scale *= scale_factor

    return pq