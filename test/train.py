#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
trainer for neat

Created : 10/12/2019 by Guilhem Le Moigne
"""

import os
import threading
import minerl

class trainer(threading.Thread) :
    def __init__(self, network) :
        '''network must have activate method'''
        os.environ['MINERL_DATA_ROOT'] = '/Users/guilhem/Documents/MineRL/neat/data'
        self.data = minerl.data.make("MineRLNavigateDense-v0")
        self.network = network
        self.fitness = 0
        threading.Thread.__init__(self)
    
    def run(self) :
#        for obs, action, reward, next_state, done in self.data.sarsd_iter(num_epochs=1, max_sequence_len=1):#len : 7686
        pass
