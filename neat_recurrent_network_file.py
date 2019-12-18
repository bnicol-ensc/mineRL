#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Get a network stored in a json file

Made json usable for neat networks

Created : 17/12/2019 by Guilhem Le Moigne
Updated : 18/12/2019 by Guilhem Le Moigne
"""

import json
import configparser
import neat


def load(network_path, config_path) :
	config_file = configparser.ConfigParser()
	config_file.read(config_path)

	input_keys = [-i - 1 for i in range(int(config_file['DefaultGenome']['num_inputs']))]
	output_keys = [i for i in range(int(config_file['DefaultGenome']['num_outputs']))]

	with open(network_path, 'r') as json_file :
		node_evals = json.load(json_file)
		for node_eval in node_evals :
			node_eval[1] = getattr(neat.activations, node_eval[1])
			node_eval[2] = getattr(neat.aggregations, node_eval[2])

	return neat.nn.recurrent.RecurrentNetwork(input_keys, output_keys, node_evals)


def save(file_path, network) :
	with open(file_path, 'w') as file :
		fnode_evals = []
		for node_eval in network.node_evals :
			node, bias, response, links = [node_eval[i] for i in [0, 3, 4, 5]]
			activation = str(node_eval[1]).split(' ')[1]
			aggregation = str(node_eval[2]).split(' ')[1]
			fnode_evals.append((node, activation, aggregation, bias, response, links))
		json.dump(fnode_evals, file)
