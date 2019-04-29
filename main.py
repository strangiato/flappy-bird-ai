import numpy
import cv2
import time
import find_pipe
import capture_screen
import find_bird
import opencv_debug
from concurrent.futures import ThreadPoolExecutor, wait
# import keras
# from keras.models import Sequential
# from keras.layers import Activation, Dense
# from keras.optimizers import SGD

# https://flappybird.io for game site


# def get_current_state():
#     bird_thread = threading.Thread(target=find_bird.get_bird_location, (img))
#     pipe_thread = threading.Thread(target=find_pipe.get_pipe_locations, (img))

# define where on your screeen the game is
monitor = {"top": 135, "left": 480, "width": 475, "height": 600}
# import pipe template image
pipe_template, pipe_template_w, pipe_template_h = find_pipe.init_pipe_template(
    'pipe_template_gray.png')

pool = ThreadPoolExecutor(2)
last_time = time.time()

while "Screen capturing":
    # get screen
    img = capture_screen.capture_screenshot(monitor)

    futures = []
    # find bird
    bird_location_future = pool.submit(find_bird.get_bird_location, img)
    futures.append(bird_location_future)
    # find pipe
    pipe_location_future = pool.submit(
        find_pipe.get_pipe_locations, img, pipe_template)
    futures.append(pipe_location_future)

    # wait for async to both finish then get values
    futures, _ = wait(futures)
    bird_location = bird_location_future.result()
    pipe_location = pipe_location_future.result()

    bird_height = bird_location[1]
    pipe_height = pipe_location[1]
    pipe_distance = pipe_location[0] - bird_location[0]
    #print(bird_height, pipe_height, pipe_distance)
    #print("Bird Location : ", bird_location)
    #print("Pipe locartion: ",  pipe_location)

    # display what opencv is seeing, passes time back in to determin fps
    last_time = opencv_debug.show_debug(
        img, bird_location, pipe_location, last_time)
