import cv2
import time


def show_debug(img, bird_location, pipe_location, last_time):
    # img = img[0:600, 170:250]  # where to look for bird
    # img = img[220:520, 100:475]  # where to look for pipes

    bird_height = bird_location[1]
    pipe_height = pipe_location[1]
    pipe_distance = pipe_location[0] - bird_location[0]

    # draw what the cv finds
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    cv2.circle(img, bird_location, 25, (0, 0, 255), 2)
    if pipe_location != (450, 350):
        cv2.rectangle(img, pipe_location, (pipe_location[0] + 32,
                                           pipe_location[1] + 32), (0, 0, 255), 2)
        cv2.putText(img, str(pipe_distance), (pipe_location[0] - 8, pipe_location[1] + 64),
                    cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img, str(round(1/(time.time() - last_time))), (400, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, str(bird_height), (bird_location[0] - 24, bird_location[1] + 48),
                cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2, cv2.LINE_AA)
    # show window
    cv2.imshow("OpenCV/Numpy normal", img)
    # Press "q" to quit
    if cv2.waitKey(25) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
    return time.time()
