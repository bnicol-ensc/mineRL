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
import random
import gym
import minerl
import time
import threading

a=0

class train(threading.Thread) :
    def __init__(self, net) :
        threading.Thread.__init__(self)
        

def fitness_function(population, config):
    global a
    for genome_id, genome in population:
        data = minerl.data.make("MineRLNavigateDense-v0")
        total_rew = 0
        print(genome_id)
        for obs, action, reward, next_state, done in data.sarsd_iter(num_epochs=1, max_sequence_len=1):#taille : 7686
            if a == 0 :
                print(obs, action, reward, next_state, done)
                a=1
            break
        genome.fitness = total_rew


def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
    p = neat.Population(config)
    winner = p.run(fitness_function, 5)
    print(winner)


if __name__ == '__main__':
    os.environ['MINERL_DATA_ROOT'] = '/Users/guilhem/Documents/MineRL/neat/data'
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-file.txt')
    run(config_path)
