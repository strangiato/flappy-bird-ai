import mss
import cv2
import numpy as np
import find_gameover

monitor = {"top": 135, "left": 480, "width": 475, "height": 600}
template = find_gameover.init_gameover_template(
    'gameover_template.png')

with mss.mss() as sct:
    while True:
        # capture screen and store as an numpy array
        img = np.array(sct.grab(monitor))
        # img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        # crop imgage
        img = img[0:600, 200:210]
        # img = img[340:400, 90:240]
        # make image grayscale

        cv2.imshow("OpenCV/Numpy normal", img)
        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
