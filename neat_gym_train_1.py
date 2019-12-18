#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
trainer for neat

Created : 13/12/2019 by Guilhem Le Moigne
Updated : 16/12/2019 by Guilhem Le Moigne
Updated : 17/12/2019 by Guilhem Le Moigne
Updated : 18/12/2019 by Guilhem Le Moigne
"""

import os
import time
import threading
import configparser
import NEAT_recurrent_network_file
import math
import datetime
import minerl
import gym
import neat


NB_GENERATIONS = 1
NB_ENVS = 1
TRAINING_TIME = lambda generation : int(5 + 55/(1+math.exp(-(generation-15))))#sigmoid centered on 15 generations that converge towards 1 min


def standardize(v, scope, offset=0) :
	try :
		l = list()
		for i in range(len(v)) :
			l.append((v[i]+offset)/scope)
		return l
	except TypeError : return (v+offset)/scope

class Fitness:
	def __init__(self, envs):
		self.envs = envs
		
	def fitness(self, population, config) :
		print('\n\rGénération : ', self.envs[0].generation, '.', sep=' ', end='\n\r\n\r')
		for genome_id, genome in population :
			net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
			i = 0
			while self.envs[i].used : i = (i+1)%len(self.envs)
			if self.envs[i].fitness != None :
				list(filter(lambda x : x[0] == self.envs[i].genome_id, population))[0][1].fitness = self.envs[i].fitness
			self.envs[i].net_activate = net.activate
			self.envs[i].genome_id = genome_id
			self.envs[i].training_time = TRAINING_TIME(self.envs[i].generation)
			self.envs[i].used=True
			print('Environnement ', self.envs[i].env_id,' utilisé par le génome ', self.envs[i].genome_id, '.', sep='')
		for i in self.envs :
			while i.used : time.sleep(0.1)
			if i.genome_id != None :
				list(filter(lambda x : x[0] == i.genome_id, population))[0][1].fitness = i.fitness
			i.fitness = None
			i.generation += 1



class MinerlEnv(threading.Thread):
	def __init__(self, env_id):
		self.env_id = env_id
		self.ready = False
		self.net_activate = None
		self.genome_id = None
		self.used = False
		self.generation = 1
		self.fitness = None
		self.training_time = 1
		super(MinerlEnv, self).__init__()

	def run(self):
		with gym.make('MineRLNavigateDense-v0') as env :
			obs = env.reset()
			self.ready = True
			done = False
			reward = 0
			total_rew = 0
			t = time.time()
			while self.generation <= NB_GENERATIONS :
				if not self.used :
					time.sleep(0.1)
					t = time.time()
				else :
					if done : obs = env.reset()
					inputs = [standardize(obs['compassAngle'], 360, 180)]
					inputs.extend([standardize(reward, 100)])
					inputs.extend(standardize(obs['pov'].flatten(), 255))
					output = self.net_activate(inputs)
					action = env.action_space.noop()
					action['camera'] = output[0]*360-180, output[1]*360-180
					action['jump'] = 0 if output[2]<=0.5 else 1
					action['forward'] = 0 if output[3]<=0.5 else 1
					action['back'] = 0 if output[4]<=0.5 else 1
					action['left'] = 0 if output[5]<=0.5 else 1
					action['right'] = 0 if output[6]<=0.5 else 1
					action['sprint'] = 0 if output[7]<=0.5 else 1
					action['sneak'] = 0 if output[8]<=0.5 else 1
					obs, reward, done, _ = env.step(action)
					total_rew += reward
					if time.time() - t > self.training_time :
						print('Le génome ', self.genome_id, ' a libére l\'environnement ', self.env_id,'.', sep='')
						self.used = False
						self.fitness = total_rew
						total_rew = 0


def main() :
	local_dir = os.path.dirname(__file__)
	config_file = os.path.join(local_dir, 'config-file.txt')
	neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)

	print('Lancement des environnements...', end='')
	envs = [MinerlEnv(env_id) for env_id in range(NB_ENVS)]
	for env in envs : env.start()
	for env in envs : 
		while not env.ready : time.sleep(0.1)

	print('\rTous les environnements sont prêts.\n\r')
	print('Demarage de l\'entraînement.')
	p = neat.Population(neat_config)
	winner = p.run(Fitness(envs).fitness, NB_GENERATIONS)

	date = datetime.datetime.now()
	net_file_path = 'train/Gym_trained_NEAT_network_{0}-{1}-{2}_{3}h{4}.json'.format(date.day, date.month, date.year, date.hour, date.minute)
	net = neat.nn.recurrent.RecurrentNetwork.create(winner, neat_config)
	NEAT_recurrent_network_file.save(net_file_path, net)

	print('\n\r\n\rEntraînement terminé.\n\r')
	print('Réseau final enregistré dans ', net_file_path, '.', sep='')

if __name__ == '__main__':
	main()
