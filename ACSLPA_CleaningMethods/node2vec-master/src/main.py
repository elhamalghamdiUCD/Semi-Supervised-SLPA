
# coding: utf-8

# In[15]:

import argparse
import numpy as np
import networkx as nx
import node2vec
import scipy.io
from collections import defaultdict
from time import perf_counter
from datetime import timedelta
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from multiprocessing import cpu_count


def read_graph(input_path, directed=False): 
    
    if(input_path.split('.')[-1] == 'edgelist'):
        G = nx.read_edgelist(input_path, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
        
    elif(input_path.split('.')[-1] == 'mat'):
        edges = list()
        mat = scipy.io.loadmat(input_path)
        nodes = mat['network'].tolil()
        G = nx.DiGraph()
        for start_node,end_nodes in enumerate(nodes.rows, start=0):
            for end_node in end_nodes:
                edges.append((start_node,end_node))
        
        G.add_edges_from(edges)
        
    else:
        import sys
        sys.exit('Unsupported input type')


    if not directed:
        G = G.to_undirected()
        
    probs = defaultdict(dict)
    for node in G.nodes():
        probs[node]['probabilities'] = dict()
        
    print(nx.info(G) + "\n---------------------------------------\n")
    return G, probs


@node2vec.timer('Generating embeddings')
def generate_embeddings(corpus, dimensions, window_size, num_workers, p, q, input_file, output_file):
    
    model = Word2Vec(corpus, size=dimensions, window=window_size, min_count=0, sg=1, workers=num_workers)
    #model.wv.most_similar('1')
    w2v_emb = model.wv
    
    if output_file == None:
        import re
        file_name = re.split('[. /]', input_file)
        output_file = 'embeddings/' + file_name[-2] + '_embeddings_'+'dim-' + str(dimensions) + '_p-'+ str(p)+'_q-'+str(q)+'.txt'
    
    print('Saved embeddings at : ',output_file)
    w2v_emb.save_word2vec_format(output_file)

    return model, w2v_emb


def process(args):
    
    Graph, init_probabilities = read_graph(args.input, args.directed)
    G = node2vec.Graph(Graph, init_probabilities, args.p, args.q, args.walks, args.length, args.workers)
    G.compute_probabilities()
    walks = G.generate_random_walks()
    model, embeddings = generate_embeddings(walks, args.d, args.window, args.workers, args.p, args.q, args.input, args.output) 
    

    return    


def main():

    parser = argparse.ArgumentParser(description = "node2vec implementation")

    parser.add_argument('--input', default='graph/karate.edgelist', help = 'Path for input edgelist')

    parser.add_argument('--output', default=None, help = 'Path for saving output embeddings')

    parser.add_argument('--p', default='1.0', type=float, help = 'Return parameter')

    parser.add_argument('--q', default='1.0', type=float, help = 'In-out parameter')

    parser.add_argument('--walks', default=10, type=int, help = 'Walks per node')

    parser.add_argument('--length', default=80, type=int, help = 'Length of each walk')

    parser.add_argument('--d', default=128, type=int, help = 'Dimension of output embeddings')

    parser.add_argument('--window', default=10, type=int, help = 'Window size for word2vec')

    parser.add_argument('--workers', default=cpu_count(), type=int, help = 'Number of workers to assign for random walk and word2vec')

    parser.add_argument('--directed', dest='directed', action ='store_true', help = 'Specify if graph is directed. Default is undirected')
    parser.set_defaults(directed=False)

    args = parser.parse_args()
    process(args)

    return


if __name__ == '__main__':
    main()
    


