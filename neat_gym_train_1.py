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
import math
import datetime
import minerl
import gym
import neat
import neat_recurrent_network_file


TRAINING_FUNC = lambda generation : int(5 + 55/(1+math.exp(-(generation-15))))#sigmoid centered on 15 generations that converge towards 1 min


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
		generation = self.envs[0].generation
		print('\n\rGénération : ', generation, '.', sep='', end='\n\r\n\r')
		for genome_id, genome in population :
			net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
			i = 0
			while self.envs[i].used : i = (i+1)%len(self.envs)
			if self.envs[i].fitness != None :
				list(filter(lambda x : x[0] == self.envs[i].genome_id, population))[0][1].fitness = self.envs[i].fitness
			self.envs[i].net_activate = net.activate
			self.envs[i].genome_id = genome_id
			self.envs[i].training_time = TRAINING_FUNC(self.envs[i].generation)
			self.envs[i].used=True
			print('Environnement ', self.envs[i].env_id,' utilisé par le génome ', self.envs[i].genome_id, '.', sep='')
		for env in self.envs :
			while env.used : time.sleep(0.1)
			if env.genome_id != None :
				list(filter(lambda x : x[0] == env.genome_id, population))[0][1].fitness = env.fitness
			env.fitness = None
			env.generation += 1
		if generation%1000 == 0 :
			print('Sauvegarde...', end='')
			best_gen = population[0][1]
			for genome_id, genome in population :
				if genome.fitness > best_gen.fitness : best_gen = genome
			net_file_path = 'saved/Gym_trained_NEAT_network_{0}_generations.json'.format(generation)
			net = neat.nn.recurrent.RecurrentNetwork.create(winner, neat_config)
			neat_recurrent_network_file.save(net_file_path, net)
			print('\rRéseau enregistré dans ', net_file_path, '.', sep='')


class MinerlEnv(threading.Thread):
	def __init__(self, env_id, num_generations):
		self.env_id = env_id
		self.num_generations = num_generations
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
			while self.generation <= self.num_generations :
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
	config_path = os.path.join(local_dir, 'config-file.txt')
	neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

	config = configparser.ConfigParser()
	config.read(config_path)
	num_generations = int(config['PERSONAL']['num_generations'])
	num_environments = int(config['PERSONAL']['num_environments'])

	print('Lancement des environnements...', end='')
	envs = [MinerlEnv(env_id, num_generations) for env_id in range(num_environments)]
	for env in envs : env.start()
	for env in envs : 
		while not env.ready : time.sleep(0.1)

	print('\rTous les environnements sont prêts.\n\r')
	print('Demarage de l\'entraînement.')
	p = neat.Population(neat_config)
	winner = p.run(Fitness(envs).fitness, num_generations)
	print('\n\r\n\rEntraînement terminé.\n\r')

	print('Sauvegarde...', end='')
	date = datetime.datetime.now()
	net_file_path = 'train/Gym_trained_NEAT_network_{0}-{1}-{2}_{3}h{4}.json'.format(date.day, date.month, date.year, date.hour, date.minute)
	net = neat.nn.recurrent.RecurrentNetwork.create(winner, neat_config)
	neat_recurrent_network_file.save(net_file_path, net)
	print('\rRéseau final enregistré dans ', net_file_path, '.', sep='')

if __name__ == '__main__':
	main()
