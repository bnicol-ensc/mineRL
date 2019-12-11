#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
trainer for neat

Created : 10/12/2019 by Guilhem Le Moigne
Updated : 11/12/2019 by Guilhem Le Moigne, Victor Leroy
"""

import os
import time
import threading
import numpy
import minerl
import gym

class Trainer(threading.Thread) :
    def __init__(self, network) :
        '''network must have activate method'''
        os.environ['MINERL_DATA_ROOT'] = '/Users/guilhem/Documents/MineRL/neat/data'
        self.network = network
        self.fitness = 0
        threading.Thread.__init__(self)
    
    def run(self) :
        env = gym.make('MineRLNavigateDense-v0')
        obs = env.reset()
        reward = 0
        done = False
        total_rew = 0
        t = time.time()
        while time.time() - t < 100 or not done :
            output = self.network.activate(numpy.array(([((obs['compassAngle'][0]+180)%360)/360, obs['pov'], reward/100]).flatten()))
            action = env.action_space.noop()
            action['camera'] = [output[0]*360-180, output[1]*360-180]
            action['jump'] = [0 if output[2]<=0.5 else 1]
            action['forward'] = [0 if output[3]<=0.5 else 1]
            action['back'] = [0 if output[4]<=0.5 else 1]
            action['left'] = [0 if output[5]<=0.5 else 1]
            action['right'] = [0 if output[6]<=0.5 else 1]
            action['sprint'] = [0 if output[7]<=0.5 else 1]
            action['sneak'] = [0 if output[8]<=0.5 else 1]
            obs, reward, done, _ = env.step(action)
            total_rew += reward
        self.fitness = total_rew
