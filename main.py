import time
import find_pipe
import capture_screen
import find_bird
import find_gameover
import opencv_debug
import input_control
import training
import numpy as np
from statistics import mean
from concurrent.futures import ThreadPoolExecutor, wait

# https://flappybird.io for game site

# define where on your screeen the game is
monitor = {"top": 135, "left": 480, "width": 475, "height": 600}
# import pipe template image
pipe_template = find_pipe.init_pipe_template(
    'pipe_template_gray.png')
# import gameover template
gameover_template = find_gameover.init_gameover_template(
    'gameover_template.png')
# create thread pool to execute cv
pool = ThreadPoolExecutor(6)
last_time = time.time()

generations = 0


def test(bird):
    gameOver = False
    last_time = time.time()
    height_history = []
    time.sleep(.25)
    start_time = time.time()
    input_control.press_space()
    while not gameOver:
        # get screen
        img = capture_screen.capture_screenshot(monitor)
        # find if the gameover screen exists
        gameover_future = pool.submit(
            find_gameover.is_gamover, img, gameover_template)
        # find bird
        bird_location_future = pool.submit(find_bird.get_bird_location, img)
        # find pipe
        pipe_location_future = pool.submit(
            find_pipe.get_pipe_locations, img, pipe_template)

        if gameover_future.result():
            # print("Game Over!!!!")
            gameOver = True
            time.sleep(.25)
            input_control.press_space()
            if len(height_history) > 0:
                # print(np.argmax(counts))
                if mean(height_history) > 550:
                    # print(start_time)
                    start_time = start_time + 4
                    # print(start_time)

        bird_location = bird_location_future.result()
        pipe_location = pipe_location_future.result()

        # get the values we want to feed forward
        bird_height = 600 - bird_location[1]
        # print(bird_height)
        height_history.append(bird_height)
        # print(bird_height)
        pipe_height = 600 - pipe_location[1]
        pipe_distance = pipe_location[0] - bird_location[0]

        if bird.jump(bird_height, pipe_height, pipe_distance):
            input_control.press_space()

        # display what opencv is seeing, passes time back in to determin fps
        # last_time = opencv_debug.show_debug(
          #  img, bird_location, pipe_location, last_time)
    print('fps: ', len(height_history)/(time.time() - start_time))
    return time.time() - start_time


    # training stuff
population = training.init_models(52)
time.sleep(3)
for g in range(25):
    print("GENERATION: ", g)
    time.sleep(1)
    img = capture_screen.capture_screenshot(monitor)
    if find_gameover.is_gamover(img, gameover_template):
        input_control.press_space()
    time.sleep(1)
    for i in range(len(population)):
        time_survived = test(population[i])
        print(i, " survived ", time_survived)
        population[i].fitness = time_survived
    population = training.breed(population)
    topfuckingpercentage = sorted(
        population, key=lambda b: b.fitness, reverse=True)
    print(topfuckingpercentage[0].model.get_weights())
