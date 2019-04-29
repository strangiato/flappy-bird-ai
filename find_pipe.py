import numpy as np
import cv2


def get_pipe_locations(img, template):

    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    img = img[220:520, 150:475]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    threshold = .84
    loc = np.where(res >= threshold)
    if len(loc[0]) > 0:
        # print(loc)
        closestpipe = np.argmin(loc[1])
        return (loc[1][closestpipe] + 150, loc[0][closestpipe] + 220)
    else:
        return (450, 350)


def init_pipe_template(path):
    template = cv2.imread(path)
    w, h = template.shape[:-1]
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    return template, w, h
