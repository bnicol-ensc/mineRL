#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
try for neat implementation

Created : 04/12/2019 by Guilhem Le Moigne
Updated : 05/12/2019 by Guilhem Le Moigne

"""


import os
import neat
import random

generation = 0

def fitness_function(population, config):
    global generation
    print(generation)
    for genome_id, genome in population:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        print(genome_id, net, sep='\n')
        if genome.fitness == None : genome.fitness = random.random()
        else : genome.fitness += random.random()
    generation += 1


def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
    p = neat.Population(config)
    winner = p.run(fitness_function, 5)
    print(winner)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-file.txt')
    run(config_path)
