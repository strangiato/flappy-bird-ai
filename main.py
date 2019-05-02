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
# from concurrent.futures import ThreadPoolExecutor, wait
from threading import Thread


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

last_time = time.time()
pool = None
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
        # gameover_future = pool.submit(
        #     find_gameover.is_gamover, img, gameover_template)
        # # find bird
        # bird_location_future = pool.submit(find_bird.get_bird_location, img)
        # # find pipe
        # pipe_location_future = pool.submit(
        #     find_pipe.get_pipe_locations, img, pipe_template)

        gameover = find_gameover.is_gamover(img, gameover_template)
        bird_location = find_bird.get_bird_location(img)
        pipe_location = find_pipe.get_pipe_locations(img, pipe_template)
        # if gameover_future.result():
        if gameover:
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

        # bird_location = bird_location_future.result()
        # pipe_location = pipe_location_future.result()

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

# seed a decent one
old_w = population[1].model.get_weights()
population[1].model.set_weights(np.asarray([[[-0.71838874,  0.10175963, -0.53971887,  0.53295004, -0.65514034,
                                              0.92242104,  0.524729],
                                             [0.98804337,  0.13244502,  0.02498293,  1., -0.7232261,
                                              -0.408572,  0.6222308],
                                             [0.17909239, -0.42393923, -0.12373942, -0.28093052, -0.36233485,
                                              0.50228304,  0.5213122]], old_w[1], [[0.72656137],
                                                                                   [-0.27863267],
                                                                                   [0.40443367],
                                                                                   [-0.11759067],
                                                                                   [-0.38491687],
                                                                                   [0.50011617],
                                                                                   [-0.40900514]], old_w[3]]))

old_w = population[2].model.get_weights()
population[2].model.set_weights(np.asarray([[[-0.6568911, -0.518004,  1., -0.62241995, -0.81495273,
                                              -0.33501074,  0.13660248],
                                             [0.7628037,  0.3569571,  0.0974381,  0.13337,  0.6942532,
                                              -0.9749049, -0.0245974],
                                             [0.39551434, -0.35870144, -1., -0.34638736, -0.7140113,
                                              0.35973847, -0.4925347]], old_w[1], [[1.],
                                                                                   [0.13644716],
                                                                                   [-0.95934737],
                                                                                   [-0.0036537],
                                                                                   [-0.5608243],
                                                                                   [0.53900385],
                                                                                   [0.41420707]], old_w[3]]))

old_w = population[3].model.get_weights()
population[3].model.set_weights(np.asarray([[[-1., -0.85244834,  1., -0.62241995, -0.62880415,
                                              -0.40136376,  1.],
                                             [1., -0.10167147, -0.36847225,  0.04832859,  0.9995372,
                                              -1.,  0.36090007],
                                             [1.,  0.05351942, -0.6417299, -0.00200751, -0.7140113,
                                              0.1387828, -0.19265364]], old_w[1], [[0.63712656],
                                                                                   [1.],
                                                                                   [-0.95934737],
                                                                                   [-1.],
                                                                                   [-0.07639154],
                                                                                   [0.5512512],
                                                                                   [0.8318679]], old_w[3]]))


time.sleep(3)

for g in range(100):
    # pool = ThreadPoolExecutor(6)
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

    topfuckingpercentage = sorted(
        population, key=lambda b: b.fitness, reverse=True)
    print(topfuckingpercentage[0].model.get_weights())

    population = training.breed(population)


# [array([[ 0.39952922, -0.706868  , -0.6584651 , -0.25239456,  0.49512887,
#          0.31369007, -0.49652296],
#        [-0.29712865, -0.52023065, -0.6602587 , -0.28444788, -0.00169295,
#          0.76802385,  0.56427824],
#        [-0.46744463,  0.17240137, -0.5094087 ,  0.19417512,  0.664835  ,
#         -0.2867403 ,  0.10904253]], dtype=float32), array([0., 0., 0., 0., 0., 0., 0.], dtype=float32), array([[ 0.19508272],
#        [-0.76031375],
#        [-0.5153545 ],
#        [ 0.4044487 ],
#        [-0.71976113],
#        [ 0.38121003],
#        [ 0.5945998 ]], dtype=float32), array([0.], dtype=float32)]
