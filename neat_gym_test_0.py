#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tester for neat

Created : 18/12/2019 by Guilhem Le Moigne
"""


import os
import NEAT_recurrent_network_file


def standardize(v, scope, offset=0) :
	try :
		l = list()
		for i in range(len(v)) :
			l.append((v[i]+offset)/scope)
		return l
	except TypeError : return (v+offset)/scope


def main() :
	local_dir = os.path.dirname(__file__)
	config_path = os.path.join(local_dir, 'config-file.txt')
	network_file = list(filter(lambda file : file[-5:] == '.json', os.listdir('train')))[0]
	network_path = os.path.join(local_dir, 'train/{0}'.format(network_file))

	network = NEAT_recurrent_network_file.load(network_path, config_path)

	with gym.make('MineRLNavigateDense-v0') as env :
		obs = env.reset()
		done = False
		reward = 0
		while not done :
			inputs = [standardize(obs['compassAngle'], 360, 180)]
			inputs.extend([standardize(reward, 100)])
			inputs.extend(standardize(obs['pov'].flatten(), 255))
			output = network.activate(inputs)
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

if __name__ == '__main__':
	main()