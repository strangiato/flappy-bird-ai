import numpy as np
import cv2


def get_bird_location(img):
    # downsize image to 1/2
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    # crop it to the only area the bird can be
    img = img[0:600, 170:250]
    # convert to rgb
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lower_red = np.array([200, 70, 60])
    upper_red = np.array([255, 110, 100])
    # find all of the beak color pixels in the img
    mask = cv2.inRange(img, lower_red, upper_red)
    points = cv2.findNonZero(mask)
    # asume its centered if you cant find it
    avg = (50, 0)

    if points is not None:
        # avg points to get the center
        avg = np.mean(points, axis=0)
        avg = avg[0]

    return (int(avg[0]) + 170, int(avg[1]))
