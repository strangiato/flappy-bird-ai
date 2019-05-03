import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras import backend
from bird import Bird
import numpy as np
import random


def init_models(generation_size):
    population = []
    for i in range(generation_size):
        model = Sequential()
        model.add(Dense(5, activation='sigmoid', input_dim=3))
        model.add(Dense(1, activation='sigmoid'))
        population.append(Bird(model, 0))
    return population


def crossover_models(model1, model2):
    rng = random.uniform(0, 1)
    if rng <= .33:
        return np.asarray([[model2.get_weights()[0][0], model1.get_weights()[0][1], model1.get_weights()[0][2]], model1.get_weights()[1], model1.get_weights()[2], model1.get_weights()[3]]), np.asarray([[model1.get_weights()[0][0], model2.get_weights()[0][1], model2.get_weights()[0][2]], model2.get_weights()[1], model2.get_weights()[2], model2.get_weights()[3]])
    elif rng > .33 and rng < .67:
        return np.asarray([[model1.get_weights()[0][0], model2.get_weights()[0][1], model1.get_weights()[0][2]], model1.get_weights()[1], model1.get_weights()[2], model1.get_weights()[3]]), np.asarray([[model2.get_weights()[0][0], model1.get_weights()[0][1], model2.get_weights()[0][2]], model2.get_weights()[1], model2.get_weights()[2], model2.get_weights()[3]])
    else:
        return np.asarray([[model1.get_weights()[0][0], model1.get_weights()[0][1], model2.get_weights()[0][2]], model1.get_weights()[1], model1.get_weights()[2], model1.get_weights()[3]]), np.asarray([[model2.get_weights()[0][0], model2.get_weights()[0][1], model1.get_weights()[0][2]], model2.get_weights()[1], model2.get_weights()[2], model2.get_weights()[3]])

# hardcoded to # neurons in hidden layer
def crossover_models_inner(model1, model2):
    rng = round(random.uniform(0, 4))

    
    if rng == 0:
        child1 =  np.asarray([np.asarray(
                [np.concatenate((model2.get_weights()[0][0][:1], model1.get_weights()[0][0][1:]), axis=None), np.concatenate((model2.get_weights()[0][1][:1], model1.get_weights()[0][1][1:]), axis=None), np.concatenate((model2.get_weights()[0][2][:1] + model1.get_weights()[0][2][1:]), axis=None)]), 
                model1.get_weights()[1], np.asarray([np.concatenate((model2.get_weights()[2][:1] + model1.get_weights()[2][1:]), axis=None)]), model1.get_weights()[3]])

        child2 = np.asarray([ np.asarray(
                    [np.concatenate((model1.get_weights()[0][0][:1], model2.get_weights()[0][0][1:]), axis=None), np.concatenate((model1.get_weights()[0][1][:1], model2.get_weights()[0][1][1:]), axis=None), np.concatenate((model1.get_weights()[0][2][:1] + model2.get_weights()[0][2][1:]), axis=None)]), 
                model2.get_weights()[1], np.asarray([np.concatenate((model1.get_weights()[2][:1], model2.get_weights()[2][1:]), axis=None)]), model2.get_weights()[3]])
    elif rng > 0 and rng < 4:
        child1 = np.asarray(
                [[np.concatenate((model1.get_weights()[0][0][:rng], model2.get_weights()[0][0][rng:rng+1], model1.get_weights()[0][0][rng + 1:]), axis=None), np.concatenate((model1.get_weights()[0][1][:rng], model2.get_weights()[0][1][rng:rng+1], model1.get_weights()[0][1][rng + 1:]), axis=None), np.concatenate((model1.get_weights()[0][2][:rng], model2.get_weights()[0][2][rng:rng+1], model1.get_weights()[0][2][rng + 1:]), axis=None)], 
                model1.get_weights()[1], np.concatenate((model1.get_weights()[2][:rng], model2.get_weights()[2][rng:rng+1], model1.get_weights()[2][rng + 1:]), axis=None), model1.get_weights()[3]])
        child2 = np.asarray(
                [[np.concatenate((model2.get_weights()[0][0][:rng], model1.get_weights()[0][0][rng:rng+1], model2.get_weights()[0][0][rng + 1:]), axis=None), np.concatenate((model2.get_weights()[0][1][:rng], model1.get_weights()[0][1][rng:rng+1], model2.get_weights()[0][1][rng + 1:]), axis=None), np.concatenate((model2.get_weights()[0][2][:rng], model1.get_weights()[0][2][rng:rng+1], model2.get_weights()[0][2][rng + 1:]), axis=None)], 
                model2.get_weights()[1], np.concatenate((model2.get_weights()[2][:rng], model1.get_weights()[2][rng:rng+1], model2.get_weights()[2][rng + 1:]), axis=None), model2.get_weights()[3]])
    else:
        child1 = np.asarray(
                [[np.concatenate((model1.get_weights()[0][0][:rng], model2.get_weights()[0][0][-1:]), axis=None), np.concatenate((model1.get_weights()[0][1][:rng], model1.get_weights()[0][1][-1:]), axis=None),np.concatenate((model2.get_weights()[0][2][:rng], model1.get_weights()[0][2][-1:]), axis=None)], 
                model1.get_weights()[1], np.concatenate((model2.get_weights()[2][:rng], model1.get_weights()[2][-1:]), axis=None), model1.get_weights()[3]])
        child2 = np.asarray(
                [[np.concatenate((model2.get_weights()[0][0][:rng], model1.get_weights()[0][0][-1:]), axis=None), np.concatenate((model2.get_weights()[0][1][:rng], model2.get_weights()[0][1][-1:]), axis=None),np.concatenate((model1.get_weights()[0][2][:rng], model2.get_weights()[0][2][-1:]), axis=None)], 
                model2.get_weights()[1], np.concatenate((model1.get_weights()[2][:rng], model2.get_weights()[2][-1:]), axis=None), model2.get_weights()[3]])

    return child1, child2

def mutate(weights):
    # weights = model.get_weights()
    mutated_weights = weights[0]
    for x in range(len(mutated_weights)):
        for y in range(len(mutated_weights[x])):
            if random.uniform(0, 1) > .85:
                change = random.uniform(-.5, .5) + mutated_weights[x][y]
                if change > 1:
                    change = 1
                if change < -1:
                    change = -1
                mutated_weights[x][y] = change
    weights[0] = mutated_weights
    mutated_weights = weights[2]
    for x in range(len(mutated_weights)):
        for y in range(len(mutated_weights[x])):
            if random.uniform(0, 1) > .85:
                change = random.uniform(-.5, .5) + mutated_weights[x][y]
                if change > 1:
                    change = 1
                if change < -1:
                    change = -1
                mutated_weights[x][y] = change
    weights[2] = mutated_weights
    return weights


def breed(population):
    # find 50 best birds
    old_population = sorted(
        population, key=lambda b: b.hiscore, reverse=True)
    # old_population = old_population[:24]  # hardcoded todo fix
    print('top 10 birds')
    for bird in old_population[:10]:
        print(bird.fitness)
    # reset population to add children to it
    population = []
    raw_weights = []
    for chad in range(10):
        for stacy in range(9):
            child1, child2 = crossover_models_inner(
                old_population[chad].model, old_population[round(random.uniform(0, 9))].model)
            raw_weights.append(mutate(child1))
            raw_weights.append(mutate(child2))

    # the first run messes up sometimes, so we do this not to sacrifice the best bird
    population.append(old_population[-1])

    for bird in old_population[:10]:
        population.append(bird)

    for index, bird in enumerate(old_population[10:-1]):
        bird.model.set_weights(raw_weights[index])
        population.append(bird)

    # print(population)
    return population
