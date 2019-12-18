
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
trainer for neat

TODO

Created : 12/12/2019 by Guilhem Le Moigne
"""

import os
import time
import threading
import numpy
import minerl
import gym

def diff(d1, d2) :
    dif = 0
    for i in d1 :
        if i != 'camera' :
            if d1[i] == d2[i][0] : dif += 1
            else : dif -= 1
    #camera pas normee
    dif += 180 - abs(180 - abs(d1['camera'][0]-d2['camera'][0][0]))
    dif += 180 - abs(180 - abs(d1['camera'][1]-d2['camera'][0][1]))
    return dif

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
        self.fitness = 0
        self.time = time
        threading.Thread.__init__(self)
    
    def run(self) :
        data = minerl.data.make('MineRLNavigateDense-v0')
        reward = 0
        done = False
        t = time.time()
        for obs, act, reward, next_state, done in data.sarsd_iter(num_epochs=1, max_sequence_len=1) :
            inputs = [standardize(obs['compassAngle'][0], 360, 180)]
            inputs.append(standardize(reward[0], 100))
            inputs.extend(standardize(obs['pov'][0].flatten(), 255))
            output = self.network.activate(inputs)
            action = {'camera':None, 'jump':None, 'forward':None, 'back':None, 'left':None, 'right':None, 'sprint':None, 'sneak':None}
            action['camera'] = output[0]*360-180, output[1]*360-180
            action['jump'] = 0 if output[2]<=0.5 else 1
            action['forward'] = 0 if output[3]<=0.5 else 1
            action['back'] = 0 if output[4]<=0.5 else 1
            action['left'] = 0 if output[5]<=0.5 else 1
            action['right'] = 0 if output[6]<=0.5 else 1
            action['sprint'] = 0 if output[7]<=0.5 else 1
            action['sneak'] = 0 if output[8]<=0.5 else 1
            self.fitness += diff(action, act)
            if time.time()-t > self.time : break
