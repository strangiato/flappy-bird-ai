import time
import find_pipe
import capture_screen
import find_bird
import opencv_debug
from concurrent.futures import ThreadPoolExecutor, wait

# https://flappybird.io for game site

# define where on your screeen the game is
monitor = {"top": 135, "left": 480, "width": 475, "height": 600}
# import pipe template image
pipe_template, pipe_template_w, pipe_template_h = find_pipe.init_pipe_template(
    'pipe_template_gray.png')
# create thread pool to execute cv
pool = ThreadPoolExecutor(2)
last_time = time.time()

while "Screen capturing":
    # get screen
    img = capture_screen.capture_screenshot(monitor)

    # find bird
    bird_location_future = pool.submit(find_bird.get_bird_location, img)
    # find pipe
    pipe_location_future = pool.submit(
        find_pipe.get_pipe_locations, img, pipe_template)

    bird_location = bird_location_future.result()
    pipe_location = pipe_location_future.result()

    bird_height = bird_location[1]
    pipe_height = pipe_location[1]
    pipe_distance = pipe_location[0] - bird_location[0]

    # display what opencv is seeing, passes time back in to determin fps
    last_time = opencv_debug.show_debug(
        img, bird_location, pipe_location, last_time)
