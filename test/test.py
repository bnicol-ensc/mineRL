#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
try for neat implementation

Created : 04/12/2019 by Guilhem Le Moigne
Updated : 05/12/2019 by Guilhem Le Moigne
Updated : 10/12/2019 by Guilhem Le Moigne
"""

import os
import neat
import train


def fitness_function(population, config):
    trainings = []
    for genome_id, genome in population:
        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        trainings.append(train.Trainer(net))
        trainings[-1].start()
    for trainer, genome in zip(trainings, [individual[1] for individual in population]) :
        trainer.join()
        genome.fitness = trainer.fitness

def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
    p = neat.Population(config)
    winner = p.run(fitness_function, 5)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-file.txt')
    run(config_path)
