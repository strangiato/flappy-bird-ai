import mss
import cv2
import numpy as np


def capture_screenshot(monitor):
    with mss.mss() as sct:
        # capture screen and store as an numpy array
        img = np.array(sct.grab(monitor))
        return img
