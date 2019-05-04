import time
import find_pipe
import capture_screen
import find_bird
import find_gameover
import opencv_debug
import input_control
import training
import bird_statistics
import numpy as np
from statistics import mean
import pickle
# from concurrent.futures import ThreadPoolExecutor, wait


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
# pool = ThreadPoolExecutor(6)

best = 0
best_current = 0
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
        #     find_gameover.is_gameover, img, gameover_template)
        # # find bird
        # bird_location_future = pool.submit(find_bird.get_bird_location, img)
        # # find pipe
        # pipe_location_future = pool.submit(
        #     find_pipe.get_pipe_locations, img, pipe_template)

        gameover = find_gameover.is_gameover(img, gameover_template)
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
                if mean(height_history) > 500:
                    # print(start_time)
                    start_time = start_time + 4
                    # print(start_time)

        # bird_location = bird_location_future.result()
        # pipe_location = pipe_location_future.result()

        # get the values we want to feed forward
        bird_height = 550 - bird_location[1]
        # print(bird_height)
        height_history.append(bird_height)
        # print(bird_height)
        pipe_height = 550 - pipe_location[1]
        pipe_distance = pipe_location[0] - bird_location[0]

        if bird.jump(bird_height, pipe_height, pipe_distance):
            input_control.press_space()

        # display what opencv is seeing, passes time back in to determin fps
        last_time = opencv_debug.show_debug(
            img, bird_location, pipe_location, last_time, generations, species, bird.name, best, best_current)

    # print('fps: ', len(height_history)/(time.time() - start_time))
    return time.time() - start_time

                                                                      
stat_tracker = bird_statistics.StatTracker()
# training stuff
population = training.init_models(100)
time.sleep(3)

for g in range(10000):
    # pool = ThreadPoolExecutor(6)
    best_current = 0
    stat_tracker.plot_progress()
    generations = g
    species = 0
    print("GENERATION: ", g)
    time.sleep(1)
    img = capture_screen.capture_screenshot(monitor)
    if find_gameover.is_gameover(img, gameover_template):
        input_control.press_space()
    time.sleep(1)
    for i in range(len(population)):
        time_survived = test(population[i])
        if time_survived > best:
            best = time_survived
        if time_survived > best_current:
            best_current = time_survived
        species += 1
        print("gen ", g, "species ", i, " survived ", time_survived)
        population[i].fitness = time_survived
        population[i].hiscore = population[i].hiscore - 1
        if time_survived > population[i].hiscore:
            population[i].hiscore = time_survived

    # topfuckingpercentage = sorted(
    #     population, key=lambda b: b.fitness, reverse=True)
    # print(topfuckingpercentage[0].model.get_weights())
    # fitness_list = map(lambda b: b.fitness, population)
    fitness_list=[bird.fitness for bird in population]

    print("saving weights")
    with open('models/' + str(generations), "wb") as fp:
        pickle.dump(population, fp)
    stat_tracker.add_generation_data(fitness_list)
    population = training.breed(population)

topfuckingpercentage = sorted(
    population, key=lambda b: b.fitness, reverse=True)
for g in topfuckingpercentage:
    print(g.model.get_weights())





# seed a decent one
# old_w = population[1].model.get_weights()
# population[1].model.set_weights(np.asarray([[[-0.71838874,  0.10175963, -0.53971887,  0.53295004, -0.65514034,
#                                               0.92242104,  0.524729],
#                                              [0.98804337,  0.13244502,  0.02498293,  1., -0.7232261,
#                                               -0.408572,  0.6222308],
#                                              [0.17909239, -0.42393923, -0.12373942, -0.28093052, -0.36233485,
#                                               0.50228304,  0.5213122]], old_w[1], [[0.72656137],
#                                                                                    [-0.27863267],
#                                                                                    [0.40443367],
#                                                                                    [-0.11759067],
#                                                                                    [-0.38491687],
#                                                                                    [0.50011617],
#                                                                                    [-0.40900514]], old_w[3]]))


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



#        [[-0.39231586,  0.25174683, -1.        ,  0.25457144, -0.69422036,
#          0.20906056,  0.4758531 ],
#        [-0.60764855, -0.74695957, -0.7193313 ,  0.8227564 ,  0.80594134,
#          0.767703  ,  1.        ],
#        [-0.01255755, -1.        , -1.        ,  1.        , -0.12812884,
#          0.23673773,  0.03819155]]
# [[ 1.        ],
#        [-0.68139595],
#        [ 0.12158702],
#        [-0.5474268 ],
#        [ 0.925998  ],
#        [ 0.04053107],
#        [ 0.12087795]]