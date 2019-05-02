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
        model.add(Dense(7, activation='sigmoid', input_dim=3))
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
        population, key=lambda b: b.fitness, reverse=True)
    # old_population = old_population[:24]  # hardcoded todo fix
    for bird in old_population:
        print(bird.fitness)
    # reset population to add children to it
    population = []
    raw_weights = []
    for chad in range(4):
        for stacy in range(6):
            child1, child2 = crossover_models(
                old_population[chad].model, old_population[chad + stacy].model)
            raw_weights.append(mutate(child1))
            raw_weights.append(mutate(child2))

    for bird in old_population[:4]:
        population.append(bird)

    for index, bird in enumerate(old_population[4:]):
        bird.model.set_weights(raw_weights[index])
        population.append(bird)
    return population
