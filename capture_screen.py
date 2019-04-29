import mss
import cv2
import numpy as np


def capture_screenshot(monitor):
    with mss.mss() as sct:
        # Part of the screen to capture

        img = np.array(sct.grab(monitor))
        return img
