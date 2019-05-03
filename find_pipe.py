import numpy as np
import cv2


def get_pipe_locations(img, template):
    # resize image
    # img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    # crop imgage
    img = img[220:520, 100:475]
    # make image grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # find the bottom pipes in the image
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    threshold = .87
    loc = np.where(res >= threshold)
    if len(loc[0]) > 0:
        # find closest pipe to the bird
        closestpipe = np.argmin(loc[1])
        return (loc[1][closestpipe] + 100, loc[0][closestpipe] + 220)
    else:
        return (450, 350)


def init_pipe_template(path):
    template = cv2.imread(path)
    template = cv2.resize(template, (0, 0), fx=0.5, fy=0.5)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    return template
