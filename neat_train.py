#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create agents with NEAT algorithm

Agents are trained in gym MineRLNavigateDense environment

Created : 13/12/2019 by Guilhem Le Moigne
Updated : 16/12/2019 by Guilhem Le Moigne
Updated : 17/12/2019 by Guilhem Le Moigne
Updated : 18/12/2019 by Guilhem Le Moigne
Updated : 07/01/2020 by Guilhem Le Moigne
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


#sigmoid centered on 15 generations that converge towards 1 min
TRAINING_TIME_FUNC = lambda generation : int(5 + 55/(1+math.exp(-(generation-15))))


def standardize(v, scope, offset=0) :
	"""Puts values that are between 0 and scope between 0 and 1"""
	try :
		l = list()
		for i in range(len(v)) :
			l.append((v[i]+offset)/scope)
		return l
	except TypeError : return (v+offset)/scope

def set_inputs(compass, reward, pov) :
	"""Format inputs"""
	inputs = [standardize(compass, 360, 180)]
	inputs.extend([standardize(reward, 100)])
	inputs.extend(standardize(pov.flatten(), 255))
	return inputs

def set_actions(noop, output) :
	"""Make actions from output"""
	noop['camera'] = output[0]*360-180, output[1]*360-180
	noop['jump'] = 0 if output[2]<=0.5 else 1
	noop['forward'] = 0 if output[3]<=0.5 else 1
	noop['back'] = 0 if output[4]<=0.5 else 1
	noop['left'] = 0 if output[5]<=0.5 else 1
	noop['right'] = 0 if output[6]<=0.5 else 1
	noop['sprint'] = 0 if output[7]<=0.5 else 1
	noop['sneak'] = 0 if output[8]<=0.5 else 1
	return noop

class Fitness:
	"""
	See fitness method

	ARGUMENTS :
	envs : list of training environments (gym format)

	METHODS:
	fitness

	"""
	def __init__(self, envs):
		self.envs = envs
		self.generation = 1

	def fitness(self, population, config) :
		"""
		Assign each genome fitness

		Fitness is defined with the reward earned by the agent born from the genome in a MineRLNavigateDense environment

		"""
		print('\n\rGénération : ', self.generation, '.\n\r', sep='')
		#put agents in minecraft env for training
		for genome_id, genome in population :
			net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
			env_id = 0
			while self.envs[env_id].used :
				env_id = (env_id+1)%len(self.envs)
				time.sleep(0.1)
			if self.envs[env_id].last_genome_trained != None :
				#retrieve former genome's fitness
				dict(population)[self.envs[env_id].last_genome_trained[0]].fitness = self.envs[env_id].last_genome_trained[1]
			self.envs[env_id].call(genome_id, net.activate, TRAINING_TIME_FUNC(self.generation))
			print('Environnement ', env_id,' utilisé par le génome ', self.envs[env_id].genome_id, '.', sep='')
		#wait for all envs to be freed
		for env in self.envs :
			while env.used : time.sleep(0.1)
			if env.last_genome_trained != None :
				#retrieve former genome's fitness
				dict(population)[env.last_genome_trained[0]].fitness = env.last_genome_trained[1]
		#saving every 1000 generations
		if self.generation%1000 == 0 :
			print('Sauvegarde...', end='')
			best_gen = population[0][1]
			for genome_id, genome in population :
				if genome.fitness > best_gen.fitness : best_gen = genome
			net_file_path = 'saved/Gym_trained_NEAT_network_{0}_generations.json'.format(self.generation)
			net = neat.nn.recurrent.RecurrentNetwork.create(best_gen, config)
			neat_recurrent_network_file.save(net_file_path, net)
			print('\rRéseau enregistré dans ', net_file_path, '.', sep='')
		self.generation += 1


class MinerlEnv(threading.Thread):
	"""
	Gets a gym MineRLNavigateDense env on a separate thread for it to be used to train networks

	ARGUMENTS:
	env_id

	METHODS:
	reset
	call
	run

	"""
	def __init__(self, env_id):
		super(MinerlEnv, self).__init__()
		self.env_id = env_id
		self.is_ready = False
		self.training_time = 1
		self.stop = False

	def reset(self, last_genome_trained=None) :
		"""Reset genome linked parameters, reset env and store last genome data"""
		self.last_genome_trained = last_genome_trained
		obs = self.env.reset()
		self.genome_id = None
		self.net_activate = None
		self.fitness = 0
		self.used = False
		return obs

	def call(self, genome_id, net_activate, training_time) :
		"""Method to call to start training a network"""
		self.genome_id = genome_id
		self.net_activate = net_activate
		self.training_time = training_time
		self.used = True

	def run(self):
		"""Open a gym MineRLNavigateDense env to train networks when self.call is called"""
		with gym.make('MineRLNavigateDense-v0') as self.env :
			obs = self.reset()
			done = False
			reward = 0
			self.is_ready = True
			t = time.time()
			while not self.stop :
				if not self.used :
					time.sleep(0.1)
					t = time.time()
				else :
					inputs = set_inputs(obs['compassAngle'], reward, obs['pov'])
					output = self.net_activate(inputs)
					action = set_actions(self.env.action_space.noop(), output)
					obs, reward, done, _ = self.env.step(action)
					self.fitness += reward
					duration = time.time() - t
					if duration > self.training_time or done :
						print('Le génome ', self.genome_id, ' libére l\'environnement ', self.env_id,'.', sep='')
						if done : self.fitness += self.training_time - duration
						self.fitness = self.fitness/duration
						obs = self.reset((self.genome_id, self.fitness))
						print("Environnement", self.env_id, "reset.")


def main() :
	#get configurations
	local_dir = os.path.dirname(__file__)
	config_path = os.path.join(local_dir, 'config-file.txt')
	neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
	config = configparser.ConfigParser()
	config.read(config_path)
	num_generations = int(config['PERSONAL']['num_generations'])
	num_environments = int(config['PERSONAL']['num_environments'])

	#launch minecraft environments
	print('Lancement des environnements...', end='')
	envs = [MinerlEnv(env_id) for env_id in range(num_environments)]
	for env in envs : env.start()
	for env in envs : 
		while not env.is_ready : time.sleep(0.1)
	print('\rTous les environnements sont prêts.\n\r')

	#train
	print('Demarage de l\'entraînement.')
	p = neat.Population(neat_config)
	winner = p.run(Fitness(envs).fitness, num_generations)
	print('\n\r\n\rEntraînement terminé.\n\r')

	#close env
	for env in envs : env.stop = True

	#save final network
	print('Sauvegarde...', end='')
	date = datetime.datetime.now()
	net_file_path = 'train/Gym_trained_NEAT_network_{0}-{1}-{2}_{3}h{4}.json'.format(date.day, date.month, date.year, date.hour, date.minute)
	net = neat.nn.recurrent.RecurrentNetwork.create(winner, neat_config)
	neat_recurrent_network_file.save(net_file_path, net)
	print('\rRéseau final enregistré dans ', net_file_path, '.', sep='')

if __name__ == '__main__':
	main()
