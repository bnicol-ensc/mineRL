#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
trainer for neat

TODO

Created : 10/12/2019 by Guilhem Le Moigne
Updated : 11/12/2019 by Guilhem Le Moigne, Victor Leroy
"""

import os
import time
import threading
import numpy
import minerl
import gym

def standardize(v, scope, offset=0) :
    try :
        l = list()
        for i in range(len(v)) :
            l.append((v[i]+offset)/scope)
        return l
    except TypeError : return (v+offset)/scope

class Trainer(threading.Thread) :
    def __init__(self, network, time) :
        '''network must have activate method'''
        os.environ['MINERL_DATA_ROOT'] = '/Users/guilhem/Documents/MineRL/neat/data'
        self.network = network
        self.time = time
        self.fitness = 0
        threading.Thread.__init__(self)
    
    def run(self) :
        with gym.make('MineRLNavigateDense-v0') as env :
            obs = env.reset()
            reward = 0
            done = False
            total_rew = 0
            t = time.time()
            while time.time() - t < self.time and not done :
                inputs = [standardize(obs['compassAngle'], 360, 180)]
                inputs.extend([standardize(reward, 100)])
                inputs.extend(standardize(obs['pov'].flatten(), 255))
                output = self.network.activate(inputs)
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
            self.fitness = total_rew
