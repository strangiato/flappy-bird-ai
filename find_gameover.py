import numpy as np
import cv2


def is_gameover(img, template):

    # resize image
    # img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    # crop imgage
    img = img[340:400, 90:240]
    # make image grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # find the bottom pipes in the image
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    threshold = .84
    loc = np.where(res >= threshold)
    # returns true if it locations the gameover
    return True if len(loc[0]) > 0 else False


def init_gameover_template(path):
    template = cv2.imread(path)
    template = cv2.resize(template, (0, 0), fx=0.5, fy=0.5)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    return template
