#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 13:37:47 2021

@author: scottoshiro
"""

import numpy as np
from math import pi

import matplotlib.pyplot as plt
from scipy.stats import rv_continuous

from QuiKo_Backend import *

from qiskit import IBMQ
from qiskit import QuantumCircuit, Aer, assemble, transpile
from qiskit.visualization import plot_bloch_multivector, plot_histogram, array_to_latex

from pythonosc.udp_client import SimpleUDPClient

class Beat_Construction:
    """
    This needs to have is own module so that it doesn't become confusing

    Start writing first this comes later
    """
    def __init__( self, Beat_matrix: dict ):
        self.Beat_matrix = []
        ip = "127.0.0.1"
        port = 8888
        self.client = SimpleUDPClient(ip, port)

        for beat in Beat_matrix:
            self.Beat_matrix.append( Beat_matrix[beat] )

    def unique_samples( self ):
        unique_samp = []; pattern=[]
        beats = self.Beat_matrix.keys()
        for index, key in enumerate( beats ):
            for samp in self.Beat_matrix[key]:
                unique_samp.append( samp )
        
        return np.unique(unique_samp)

    def send_to_chuck( self, start_constr=False ):
        if start_constr == False:
            self.client.send_message("/layer/BEAT0", self.Beat_matrix[0] ) # Send message with int, float and string
            self.client.send_message("/layer/BEAT1", self.Beat_matrix[1] )
            self.client.send_message("/layer/BEAT2", self.Beat_matrix[2] )
            self.client.send_message("/layer/BEAT3", self.Beat_matrix[3] )
            self.client.send_message("/layer/BEAT4", self.Beat_matrix[4] )
            self.client.send_message("/layer/BEAT5", self.Beat_matrix[5] )
            self.client.send_message("/layer/BEAT6", self.Beat_matrix[6] )
            self.client.send_message("/layer/BEAT7", self.Beat_matrix[7] )

        if start_constr == True:
            self.client.send_message("/construct/start", "start" )

        pass
    
    """ before I can do this effectively I need to build out the database and the interface"""