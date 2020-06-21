
"""
Graph Utility functions

Author: Apoorva Vinod Gorur
"""

import numpy as np
import networkx as nx
from collections import defaultdict
from time import perf_counter
from datetime import timedelta


def timer(msg):
    def inner(func):
        def wrapper(*args, **kwargs):
            t1 = perf_counter()
            ret = func(*args, **kwargs)
            t2 = perf_counter()
            print("Time elapsed for "+msg+" ----> "+str(timedelta(seconds=t2-t1)))
            print("\n---------------------------------------\n")
            return ret
        return wrapper
    return inner



class Graph():
    
    def __init__(self, graph, probs, p, q, max_walks, walk_len, workers):

        self.graph = graph
        self.probs = probs
        self.p = p
        self.q = q
        self.max_walks = max_walks
        self.walk_len = walk_len
        self.workers = workers 
        return
    
    @timer('Computing probabilities')   
    def compute_probabilities(self):
        
        G = self.graph
        for source_node in G.nodes():
            for current_node in G.neighbors(source_node):
                probs_ = list()
                for destination in G.neighbors(current_node):

                    if source_node == destination:
                        prob_ = G[current_node][destination].get('weight',1) * (1/self.p)
                    elif destination in G.neighbors(source_node):
                        prob_ = G[current_node][destination].get('weight',1)
                    else:
                        prob_ = G[current_node][destination].get('weight',1) * (1/self.q)

                    probs_.append(prob_)

                self.probs[source_node]['probabilities'][current_node] = probs_/np.sum(probs_)
        
        return
    
    @timer('Generating Biased Random Walks')
    def generate_random_walks(self):
        
        G = self.graph
        walks = list()
        for start_node in G.nodes():
            for i in range(self.max_walks):
                
                walk = [start_node]
                walk_options = list(G[start_node])
                if len(walk_options)==0:
                    break
                first_step = np.random.choice(walk_options)
                walk.append(first_step)
                
                for k in range(self.walk_len-2):
                    walk_options = list(G[walk[-1]])
                    if len(walk_options)==0:
                        break
                    probabilities = self.probs[walk[-2]]['probabilities'][walk[-1]]
                    next_step = np.random.choice(walk_options, p=probabilities)
                    walk.append(next_step)
                
                walks.append(walk)
        np.random.shuffle(walks)
        walks = [list(map(str,walk)) for walk in walks]
        
        return walks

