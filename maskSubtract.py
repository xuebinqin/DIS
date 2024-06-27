import cv2
import numpy as np
# base = cv2.imread("obj1_mask.png")
# next_frame = cv2.imread("obj3_mask.png")
# sub = cv2.subtract(next_frame, base)
# cv2.imshow("sub", sub)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def maskSubtract(base_frame, current_frame):
    sub = cv2.subtract(current_frame, base_frame)
    return sub