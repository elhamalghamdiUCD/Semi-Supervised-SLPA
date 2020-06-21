import numpy as np
import time
import itertools
import random
import sys
import networkx as nx
from operator import itemgetter
from collections import defaultdict
import copy
import csv
import math
from node2vec import Node2Vec
from simrank import simrank
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.ensemble import IsolationForest
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import LocalOutlierFactor
import os
import json
import glob
import time
import argparse
import numpy as np
import pandas as pd
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.models import Model
from keras import losses
from datetime import timedelta
from sklearn.metrics import roc_auc_score
from params import Params
import models
from sklearn import svm



def find_communities_Stand_SLPA(G, T, r):
    ##Stage 1: Initialization
    memory = {i: {i: 1} for i in G.nodes()}
    ##Stage 2: Evolution
    for t in range(T):
        listenersOrder = list(G.nodes())
        np.random.shuffle(listenersOrder)
        for listener in listenersOrder:
            speakers = G[listener].keys()
            if len(speakers) == 0:
                continue
            labels = defaultdict(int)
            for j, speaker in enumerate(speakers):
                # Speaker Rule
                total = float(sum(memory[speaker].values()))
                labels[list(memory[speaker].keys())[
                    np.random.multinomial(1, [freq / total for freq in memory[speaker].values()]).argmax()]] += 1

            # Listener Rule
            random_key = get_rnd_key_of_max_value(labels)
            acceptedLabel = random_key
            # acceptedLabel = max(labels, key=labels.get)
            # Update listener memory
            if acceptedLabel in memory[listener]:
                memory[listener][acceptedLabel] += 1
            else:
                memory[listener][acceptedLabel] = 1

    ## Stage 3:
    for node, mem in list(memory.items()):
        for label, freq in list(mem.items()):
            if freq / float(T + 1) < r:
                del mem[label]
    # Find nodes membership
    communities = {}
    for node, mem in list(memory.items()):
        for label in list(mem.keys()):
            if label in communities:
                communities[label].add(node)
            else:
                communities[label] = set([node])
    # Remove nested communities
    nestedCommunities = set()
    keys = list(communities.keys())
    for i, label0 in enumerate(keys[:-1]):
        comm0 = communities[label0]
        for label1 in keys[i + 1:]:
            comm1 = communities[label1]
            if comm0.issubset(comm1):
                nestedCommunities.add(label0)
            elif comm0.issuperset(comm1):
                nestedCommunities.add(label1)
    for comm in nestedCommunities:
        del communities[comm]
    return communities


def find_communities(G, T, r, must_constraints, cannot_constraints):
    ## Stage 1: Initialization
    memory = {i: {i: 1} for i in G.nodes()}

    ## After Stage 1: Must-link processing: find connected must-link nodes (clique of must-link), and assign every node in clique with the label of the highest degree.
    ## put the must-link const. into graph
    G_must_links = nx.Graph()
    for pair in must_constraints:
        pair = list(pair)
        if str(pair[1]) in memory[str(pair[0])]:
            memory[str(pair[0])][str(pair[1])] += 1
        else:
            memory[str(pair[0])][str(pair[1])] = 1

        if str(pair[0]) in memory[str(pair[1])]:
            memory[str(pair[1])][str(pair[0])] += 1
        else:
            memory[str(pair[1])][str(pair[0])] = 1
        G_must_links.add_edge(pair[0], pair[1])

    ## find max cliques
    G_must_links_clique = list(nx.find_cliques(G_must_links))

    ## build a map of cliques for each node
    node_map_cliq = {}
    nodes1 = G.nodes()
    for node in nodes1:
        node_map_cliq[node] = set()

    for cliq in G_must_links_clique:
        for pair in itertools.combinations(cliq, 2):
            node_map_cliq[str(pair[0])].add(str(pair[1]))
            node_map_cliq[str(pair[1])].add(str(pair[0]))

    ##Stage 2: Evolution
    ##During Stage 2:Processing both: include the must-link nodes in the speakers set of the listener and delete the cannot-link nodes from the set.
    communities = {}
    for node, mem in list(memory.items()):
        for label in list(mem.keys()):
            if label in communities:
                communities[label].add(node)
            else:
                communities[label] = set([node])

    for t in range(T):
        listenersOrder = list(G.nodes())
        np.random.shuffle(listenersOrder)

        for listener in listenersOrder:

            ##include must-link nodes in speakers
            must_link_set = node_map_cliq[listener]
            if len(must_link_set) != 0:
                speakers = list(set(G[listener].keys()).union(node_map_cliq[listener]))
            else:
                speakers = list(G[listener].keys())

            ##delete cannot-link nodes in speakers
            for node in speakers:
                pair4 = frozenset([int(listener), int(node)])
                if pair4 in cannot_constraints:
                    speakers.remove(node)

            if len(speakers) == 0:
                continue

            labels = defaultdict(int)
            for j, speaker in enumerate(speakers):
                # Speaker Rule
                total = float(sum(memory[speaker].values()))
                labels[list(memory[speaker].keys())[
                    np.random.multinomial(1, [freq / total for freq in memory[speaker].values()]).argmax()]] += 1

            # Listener Rule
            random_key = get_rnd_key_of_max_value(labels)
            acceptedLabel = random_key

            nodes_of_label_acceptedLabel = list(communities[acceptedLabel])
            # Update listener memory
            checker3 = False
            for node in nodes_of_label_acceptedLabel:
                pair = frozenset([int(node), int(listener)])
                if pair in cannot_constraints:
                    checker3 = True
                    break

            if checker3 == False:
                if acceptedLabel in memory[listener]:
                    memory[listener][acceptedLabel] += 1
                    if acceptedLabel in communities:
                        communities[acceptedLabel].add(listener)
                    else:
                        communities[acceptedLabel] = set([listener])
                else:
                    memory[listener][acceptedLabel] = 1

    ## Before Stage 3:
    # Find nodes membership
    communities = {}
    for node, mem in list(memory.items()):
        for label in list(mem.keys()):
            if label in communities:
                communities[label].add(node)
            else:
                communities[label] = set([node])

    # must-link processing:
    for pair in must_constraints:
        pair = list(pair)
        nodeOne_Set = set(memory[str(pair[0])])
        nodeTwo_Set = set(memory[str(pair[1])])
        nodeOne = str(pair[0])
        nodeTwo = str(pair[1])
        # print('nodeOne=',nodeOne,', ',set(memory[str (pair[0])].items()))
        # print('nodeTwo',nodeTwo,', ', set(memory[str (pair[1])].items()))



        node_One_labels_values = frozenset((memory[str(pair[0])]).values())
        node_Two_labels_values = frozenset((memory[str(pair[1])]).values())

        node_One_max_label_value = max(node_One_labels_values)
        node_Two_max_label_value = max(node_Two_labels_values)

        node_One_max_label_keys = [k for k, v in (memory[str(pair[0])]).items() if v == node_One_max_label_value]
        node_Two_max_label_keys = [k for k, v in (memory[str(pair[1])]).items() if v == node_Two_max_label_value]
        commenMax = set(node_One_max_label_keys).intersection(set(node_Two_max_label_keys))
        # print('node_One_max_label_keys= ',node_One_max_label_keys)
        # print('node_Two_max_label_keys= ',node_Two_max_label_keys)


        if len(commenMax) != 0:
            continue
        else:
            ii = 0
            done_Two = True
            while done_Two and ii < len(node_Two_max_label_keys):
                nodes_of_label_Two = list(communities[node_Two_max_label_keys[ii]])
                checker = False
                for node in nodes_of_label_Two:
                    pair = frozenset([int(node), int(nodeOne)])
                    if pair in cannot_constraints:
                        checker = True
                        # print("checker:dont accept node")
                        break

                if checker == False:
                    # print("checker == False")
                    if str(node_Two_max_label_keys[ii]) in memory[nodeOne]:
                        memory[nodeOne][str(node_Two_max_label_keys[ii])] += memory[nodeTwo][
                            str(node_Two_max_label_keys[ii])]
                        done_Two = False
                    else:
                        memory[nodeOne][str(node_Two_max_label_keys[ii])] = memory[nodeTwo][
                            str(node_Two_max_label_keys[ii])]
                        done_Two = False
                ii = ii + 1

            ij = 0
            done_One = True
            while done_One and ij < len(node_One_max_label_keys):
                nodes_of_label_One = list(communities[node_One_max_label_keys[ij]])
                # print('nodes_of_label_One',nodes_of_label_One)
                # print('nodes_of_label_One',nodes_of_label_One)

                checker2 = False
                for node in nodes_of_label_One:
                    pair = frozenset([int(node), int(nodeTwo)])
                    if pair in cannot_constraints:
                        checker2 = True
                        # print("checker2:dont accept node")
                        break

                if checker2 == False:
                    # print("checker2 == False")
                    if str(node_One_max_label_keys[ij]) in memory[nodeTwo]:
                        memory[nodeTwo][str(node_One_max_label_keys[ij])] += memory[nodeOne][
                            str(node_One_max_label_keys[ij])]
                        done_One = False
                    else:
                        memory[nodeTwo][str(node_One_max_label_keys[ij])] = memory[nodeOne][
                            str(node_One_max_label_keys[ij])]
                        done_One = False
                ij = ij + 1

    # Processing Cannot-Link constraints:compare the memories of cannot-link pairs, if there is common label, delete the one with less frequency.
    for pair in cannot_constraints:
        pair = list(pair)
        nodeOne_Set = set(memory[str(pair[0])])
        nodeTwo_Set = set(memory[str(pair[1])])
        commenKeys = nodeOne_Set.intersection(nodeTwo_Set)

        if len(commenKeys) == 0:
            continue
        else:
            for label in list(commenKeys):
                if memory[str(pair[0])][label] > memory[str(pair[1])][label]:
                    del memory[str(pair[1])][label]
                elif memory[str(pair[0])][label] < memory[str(pair[1])][label]:
                    del memory[str(pair[0])][label]
                else:
                    labels = []
                    labels.append(pair[0])
                    labels.append(pair[1])
                    random_key = random.choice(labels)
                    del memory[str(random_key)][label]

    ## Stage 3:post-processing
    for node, mem in list(memory.items()):
        for label, freq in list(mem.items()):
            if freq / float(T + 1) < r:
                del mem[label]

    # Find nodes membership
    communities = {}
    for node, mem in list(memory.items()):
        for label in list(mem.keys()):
            if label in communities:
                communities[label].add(node)
            else:
                communities[label] = set([node])

    # Remove nested communities
    nestedCommunities = set()
    keys = list(communities.keys())
    for i, label0 in enumerate(keys[:-1]):
        comm0 = communities[label0]
        for label1 in keys[i + 1:]:
            comm1 = communities[label1]
            if comm0.issubset(comm1):
                nestedCommunities.add(label0)
            elif comm0.issuperset(comm1):
                nestedCommunities.add(label1)

    for comm in nestedCommunities:
        del communities[comm]

    return communities


def get_rnd_key_of_max_value(labels):
    # Randomly choose the index of a maximum value.
    # Invert the dictionary (this is expensive, can be optimized).
    label_values = frozenset(labels.values())
    # print("label_values")
    # print(label_values)
    max_label_value = max(label_values)
    # print("max_label_value")
    # print(max_label_value)

    max_label_keys = [k for k, v in labels.items()
                      if v == max_label_value]

    # Pick a random key of a maximum value.
    random_key = random.choice(max_label_keys)
    return random_key


def read_communities(in_path):
    """
    Read communities from the specified file.
    We assume each node identifier is an integer, and there is one community per line.
    """
    communities = []
    fin = open(in_path, "r")
    for line in fin.readlines():
        community = set()
        for node in line.strip().split(" "):
            community.add(int(node))
        if len(community) > 0:
            communities.append(community)
    fin.close()
    return communities


def assigned_nodes(communities):
    """
    Get all nodes assigned to at least one community.
    """
    assigned = set()
    for community in communities:
        assigned = assigned.union(community)
    return assigned

def selectCons(ImportantPairs,node_map,must_constraints_tobenoised,cannot_constraints_tobenoised):
    must_constraints = set()
    cannot_constraints = set()
    for pair in ImportantPairs:
        pairlist = list(pair)
        if int(pairlist[0]) in node_map[int(pairlist[1])]:
            if (not pair in must_constraints):
                must_constraints.add(pair)
        else:
            if (not pair in cannot_constraints):
                cannot_constraints.add(pair)

    return must_constraints, cannot_constraints



def find_overlapping_nodes(communities):
    community_counts = defaultdict(int)
    for comm in communities.values():
        for node_index in comm:
            community_counts[node_index] += 1
    overlapping_nodes = set()
    for node_index in community_counts:
        if community_counts[node_index] > 1:
            overlapping_nodes.add(node_index)
    return overlapping_nodes


def Node_Pair_Selection_Method_3(G, communities):
    Pair_list = []
    NodeI_list=[]
    # print('communities',communities)
    ##select overlapping nodes
    overlapping_nodes= find_overlapping_nodes(communities)

    # print ('overlapping_nodes',overlapping_nodes)
    ##select boundary nodes
    boundary_nodes=set()
    for community in communities.values():
        B_nodes = Find_Boundary_Node(G, community)
        boundary_nodes.update(B_nodes)
    # print ('boundary_nodes',boundary_nodes)
    Selected_Nodes =list(boundary_nodes)+ list(overlapping_nodes)

    # print ('Selected_Nodes',Selected_Nodes)

    for node in list(Selected_Nodes):
        selected_comm = []
        neighbors=list(G.neighbors(str(node)))
        # print (neighbors)
        for n in neighbors:
            for comm in communities.values():
                if n in comm:
                    if not comm in selected_comm:
                        selected_comm.append(comm)
        for comm in list(selected_comm):
            HighDegNode = highDFunction(G, comm, node,overlapping_nodes)
            value= [int(node), int(HighDegNode), int(G.degree(str(node)))]
            if len(value) > 1:
                NodeI_list.append(value)

    NodeI_list.sort(key = itemgetter(2))
    # print('NodeI_list',NodeI_list)
    for value in NodeI_list:
        pair = frozenset([int(value[0]), int(value[1])])
        if len(pair) > 1 and not pair in Pair_list:
            Pair_list.append(pair)

    return Pair_list



def highDFunction(G, community, A,overlapping_nodes):
    MaxDegree = 0
    HighDegNode = -1
    for node in community:
        if node == A or node in overlapping_nodes :
            continue
        degree = G.degree(node)
        if degree > MaxDegree:
            MaxDegree = G.degree(node)
            HighDegNode = node
    if HighDegNode== -1:
        for node in community:
            if node == A:
                continue
            degree = G.degree(node)
            if degree > MaxDegree:
                MaxDegree = G.degree(node)
                HighDegNode = node
    return HighDegNode

def Find_Boundary_Node(G, comm):
    Boundary_Nodes = set()
    for node in comm:
        N_neighbors = list(G.neighbors(str(node)))
        if (set(N_neighbors) == set(comm)) or (set(N_neighbors).issubset(set(comm))):
            continue
        else:
            Boundary_Nodes.add(node)
    return Boundary_Nodes


def noise_const_method(communities_GroundTruth, num_must_noise, num_cannot_noise, G):
    node_map = {}
    must_constraints = set()
    # must_constraints_noised = set()
    cannot_constraints = set()
    # cannot_constraints_noised = set()

    n = len(G.nodes())

    nodes = list(assigned_nodes(communities_GroundTruth))
    for node in nodes:
        node_map[node] = set()
    for community in communities_GroundTruth:
        for pair in itertools.combinations(community, 2):
            node_map[pair[0]].add(pair[1])
            node_map[pair[1]].add(pair[0])

    while (len(must_constraints) < num_cannot_noise) or (len(cannot_constraints) < num_must_noise):
        node_index1, node_index2 = random.randint(0, n - 1), random.randint(0, n - 1)
        # ignore self-constraints
        if node_index1 == node_index2:
            continue
        node1, node2 = nodes[node_index1], nodes[node_index2]
        pair = frozenset([node1, node2])
        # is this a must-link? then flip it to cannot
        if node2 in node_map[node1]:
            # avoid duplicates
            if (len(must_constraints) < num_must_noise) and not pair in must_constraints and not pair in cannot_constraints:
                must_constraints.add(pair)
                node_map[node1].remove(node2)
                if node1 in node_map[node2]:
                    node_map[node2].remove(node1)
        # then it is cannot-link. flip it to must
        else:
            # avoid duplicates
            if (len(cannot_constraints) < num_cannot_noise) and not pair in cannot_constraints and not pair in must_constraints:
                cannot_constraints.add(pair)
                node_map[node1].add(node2)
                node_map[node2].add(node1)
    print('total must-link', len(must_constraints), 'total cannot-link', len(cannot_constraints))
    return node_map, must_constraints, cannot_constraints

def path_length(g,pairwise_cost):
    # g.has_edge(pair[0],pair[1])
    num_pairs = 0
    temp = []
    # temp.append('pl')
    for pair in pairwise_cost:
        pair=list(pair)
        # print(pair)
        temp.append(nx.shortest_path_length(g, source=str(pair[0]), target=str(pair[1])))
        num_pairs += 1
    pl=np.array(temp)
    pl = pl.reshape(pl.shape[-1],1)
    # print("Wrote values for %d pairs" % num_pairs)
    return pl

def Has_edge(g,pairwise_cost):
    num_pairs = 0
    temp = []
    # temp.append('HE')
    for pair in pairwise_cost:
        pair=list(pair)
        # print(pair, int(g.has_edge(str(pair[0]),str(pair[1]))) )
        temp.append(int(g.has_edge(str(pair[0]),str(pair[1]))))
        # print(hasEdge)
        # if hasEdge=='False' or hasEdge==False:  
        #     temp.append(0)
        # else:
        #     temp.append(1)
        num_pairs += 1
    HE=np.array(temp)
    HE=HE.astype(int)
    HE = HE.reshape(HE.shape[-1],1)
    # print("Wrote values for %d pairs" % num_pairs)
    return HE

def mutual_neighbour_fun(g,pairwise_cost, degrees ):

    num_pairs = 0
    temp = []
    # temp.append('mn')
    for pair in pairwise_cost:
        pair=list(pair)
        # print(pair)
        temp.append(mutual_neighbour( str(pair[0]), str(pair[1]), g, degrees ))
        num_pairs += 1
    HE=np.array(temp)
    HE = HE.reshape(HE.shape[-1],1)
    # print("Wrote values for %d pairs" % num_pairs)
    return HE

def mutual_neighbour( node1, node2, g, degrees ):
    d_n1 = degrees[node1]
    d_n2 = degrees[node2]
    n1_neighbors = set(g.neighbors(str(node1)))
    n2_neighbors = set(g.neighbors(str(node2)))
    shared=len(n1_neighbors.intersection(n2_neighbors))
    denom = (d_n1+d_n2)
    if denom == 0:
        return 0
    return shared/denom

def simrank_fun(g,pairwise_cost,scores):

    num_pairs = 0
    temp = []
    for pair in pairwise_cost:
        pair=list(pair)
        # print(pair)
        temp.append(scores[str(pair[0])][str(pair[1])])
        num_pairs += 1
    simR=np.array(temp)
    simR = simR.reshape(simR.shape[-1],1)
    return simR

def cosine_similarity( node1, node2, g, degrees ):
    d_n1 = degrees[node1]
    d_n2 = degrees[node2]
    n1_neighbors = set(g.neighbors(str(node1)))
    n2_neighbors = set(g.neighbors(str(node2)))
    shared = len(n1_neighbors.intersection(n2_neighbors))
    denom = math.sqrt(d_n1)*math.sqrt(d_n2)
    if denom == 0:
        return 0
    return shared/denom
    
# --------------------------------------------------------------

def cosine_fun(g,pairwise_cost):
    degrees = dict(g.degree())
    num_pairs = 0
    temp = []
    for pair in pairwise_cost:
        pair=list(pair)
        temp.append(cosine_similarity(str(pair[0]), str(pair[1]), g, degrees))
        num_pairs += 1
    sim=np.array(temp)
    sim = sim.reshape(sim.shape[-1],1)
    return sim

def Node2Vec_model(g):

    large_graph = False
    temp_folder = "/tmp/node2vec"

    # embedding parameters
    dimensions = 128
    window_size = 10
    # performance parameters
    num_threads = 4
    num_walks = 200

    
    # Precompute probabilities and generate walks
    print("Generating walks on the network ...")
    ## if d_graph is too big  to fit in the memory, pass temp_folder which has enough disk space
    if large_graph:
        # Note: It will trigger "sharedmem" in Parallel, which will be slow on smaller graphs
        node2vec = Node2Vec(g, dimensions=dimensions, walk_length=30, num_walks=num_walks, workers=num_threads, temp_folder=temp_folder)
    else:
        node2vec = Node2Vec(g, dimensions=dimensions, walk_length=30, num_walks=num_walks, workers=num_threads)

    # Embed
    print("Building the embedding (%d dimensions) ..." % dimensions)
    model = node2vec.fit(window=window_size, min_count=1, batch_words=4)

    return model

def Node2Vec_fun(g, pairwise_cost, model):
    num_pairs = 0
    temp = []
    for pair in pairwise_cost:
        pair=list(pair)
        temp.append(model.wv.similarity(str(pair[0]), str(pair[1])))
        num_pairs += 1
    n2v=np.array(temp)
    n2v = n2v.reshape(n2v.shape[-1],1)
    return n2v


def AE(X,AE_Model):
    params = Params("hparams.yaml", AE_Model)
    net= getattr(models, params.model_name)
    input_dim = X.shape[-1]
    model = net(input_dim, l1_factor=params.l1_factor, )
    X=X.astype('float32').values
    model.compile(loss=losses.mean_squared_error, optimizer=Adam(lr=params.lr), metrics=['mse'])

    training_logs = model.fit(X, X,
                    epochs=params.num_epochs,verbose=0,
                    batch_size=params.batch_size,
                    shuffle=True)
    encoder = Model(model.input, model.layers[1].output)
    print("ENCODER", encoder)
    latent = np.array(encoder.predict(X))
    print("LATENT SHAPE", latent.shape)
    return latent

def ISO(X):
    ISF=IsolationForest(behaviour='new',contamination=0.1)
    # score=ISF.decision_function(X) # for auc
    score=ISF.fit(X).predict(X)
    print('IsolationForest data done')
    return score

def SVM_1(X):
    SVM=svm.OneClassSVM(nu=0.25, kernel="rbf", gamma=0.1)
    score=SVM.fit(X).predict(X)
    # score=-SVM.decision_function(X) # for auc
    print('OneClassSVM data done')
    return score
    

def min_max_norm(df, numeric_cols, training_set=False, scalar=None):
    '''
    df (Dataframe): Dataframe to normalize.
    numeric_cols (list): List of numeric columns.
    training_set (Boolean): Boolean indicating whether this is the training set or not.
    scalar (sklearn.preprocessing.data.MinMaxScaler object): scalar from training set if training_set=False
    '''
    df_norm = df.copy()
    if training_set:
        scalar = MinMaxScaler()
        scalar.fit(df_norm[numeric_cols]) 
    else:
        if not scalar:
            raise ValueError("'scalar' argument is None. If not testing or validation set, pass scalar object from training set.")
        small_min = pd.Series(dict(zip(numeric_cols,scalar.data_min_)))
        small_max = pd.Series(dict(zip(numeric_cols,scalar.data_max_)))
        df_norm[numeric_cols] = df_norm[numeric_cols].clip(small_min,small_max, axis=1)
    df_norm[numeric_cols] = scalar.transform(df_norm[numeric_cols])
    return df_norm, scalar


def prepare_data_without_label(df_to_prep):
    df = df_to_prep.copy()
    numeric_cols = [
        "simC",
        "mn",
        "n2v",
        "pl",
        "simR"
        ]

    binary_cols = [
        "HE" 
    ]

    # Check for expected columns
    all_cols = numeric_cols+binary_cols
    assert sorted(df.columns) == sorted(all_cols)

    # Range normalization of numeric columns    
    df, _ = min_max_norm(df, numeric_cols, training_set=True, scalar=None)
    X_col_names = numeric_cols+["HE"]
    
    return df[X_col_names]


def Detect_noisy_pairwise_constraints_must(G,constraints,scores,N2V_model,degrees,node_map2,threshold,AE_Model,must_discard_5,AE_Model_var_must):

    pairwise_cost_pairs=[]

    must_constraints_1=set()

    for nodes in constraints:
        nodes=list(nodes)
        pair=[]
        pair.append(nodes[0])
        pair.append(nodes[1])
        pair.append(int(1)) # label for must_link
        pairwise_cost_pairs.append(pair)

    pairwise_cost_pairs=np.array(pairwise_cost_pairs)

    label=[]
    for p in pairwise_cost_pairs:
        label.append(int(p[2]))
    label=np.array(label)
    label=label.astype(int)
    label=label.reshape(label.shape[-1],1)
    simR=simrank_fun(G,pairwise_cost_pairs,scores)
    n2v=Node2Vec_fun(G,pairwise_cost_pairs,N2V_model)
    simC=cosine_fun(G,pairwise_cost_pairs)
    pl=path_length(G,pairwise_cost_pairs)
    HE=Has_edge(G,pairwise_cost_pairs)
    mn=mutual_neighbour_fun(G,pairwise_cost_pairs,degrees)


    pairwise_const_features=np.concatenate((simC,mn,n2v,pl,simR,HE), axis=1)
    pairwise_const_features=pd.DataFrame(pairwise_const_features, columns = ['simC','mn','n2v','pl','simR','HE']) 
    pairwise_const_norm=prepare_data_without_label(pairwise_const_features)

    scores=SVM_1(pairwise_const_norm)

    scores = scores.reshape(scores.shape[-1],1)
    pairwise_const_results=np.concatenate((pairwise_cost_pairs,scores), axis=1)
    pairwise_const_results = pairwise_const_results[pairwise_const_results[:,3].argsort()[::-1]]
    
    num=int(len(pairwise_const_results)*threshold)
    counter=0
    must_discard_5=set(must_discard_5)
    for pair in pairwise_const_results:
        if  pair[3]<1.0 or pair[3]<1:
            pair1 = frozenset([int(pair[0]), int(pair[1])])
            must_discard_5.add(pair1)
        else:
            pair1 = frozenset([int(pair[0]), int(pair[1])])
            must_constraints_1.add(pair1)
        counter =counter+1
        
    return list(must_constraints_1), list(must_discard_5), AE_Model_var_must

def Detect_noisy_pairwise_constraints_cannot(G,constraints,scores,N2V_model,degrees,node_map2,threshold,AE_Model,cannot_discard_5,AE_Model_var_cannot):

    pairwise_cost_pairs=[]
    cannot_constraints_1=set()
    for nodes in constraints:
        nodes=list(nodes)
        pair=[]
        pair.append(nodes[0])
        pair.append(nodes[1])
        pair.append(int(0)) # label for cannot_link
        pairwise_cost_pairs.append(pair)

    pairwise_cost_pairs=np.array(pairwise_cost_pairs)

    label=[]
    for p in pairwise_cost_pairs:
        label.append(int(p[2]))
    label=np.array(label)
    label=label.astype(int)
    label=label.reshape(label.shape[-1],1)

    n2v=Node2Vec_fun(G,pairwise_cost_pairs,N2V_model,)
    simR=simrank_fun(G,pairwise_cost_pairs,scores)
    simC=cosine_fun(G,pairwise_cost_pairs)
    pl=path_length(G,pairwise_cost_pairs)
    HE=Has_edge(G,pairwise_cost_pairs)
    mn=mutual_neighbour_fun(G,pairwise_cost_pairs,degrees)


    pairwise_const_features=np.concatenate((simC,mn,n2v,pl,simR,HE), axis=1)
    pairwise_const_features=pd.DataFrame(pairwise_const_features, columns = ['simC','mn','n2v','pl','simR','HE'])
    pairwise_const_norm=prepare_data_without_label(pairwise_const_features)
    scores=ISO(pairwise_const_norm)

    scores = scores.reshape(scores.shape[-1],1)
    pairwise_const_results=np.concatenate((pairwise_cost_pairs,scores), axis=1)
    pairwise_const_results = pairwise_const_results[pairwise_const_results[:,3].argsort()[::-1]] # highest to lowest 

    

    num=int(len(pairwise_const_results)*threshold)
    counter=0
    cannot_discard_5=set(cannot_discard_5)
    for pair in pairwise_const_results:
        if  pair[3]<1.0 or pair[3]<1:
            pair1 = frozenset([int(pair[0]), int(pair[1])])
            cannot_discard_5.add(pair1)
        else:
            pair1 = frozenset([int(pair[0]), int(pair[1])])
            cannot_constraints_1.add(pair1)
        counter =counter+1
    return list(cannot_constraints_1),list(cannot_discard_5), AE_Model_var_cannot

def main():
  
    Must_Pair_With_Noise=set()
    Cannot_Pair_With_Noise=set()

    fh = open(sys.argv[1], "rb")
    names = str(fh).split('/')
    length = len(names)
    name = str(names[length - 1]).split('.')
    print ('file name', name[0])
    G = nx.read_edgelist(fh)

    N2V_model=0
    scores=0
    p_sim=str("Models/"+name[0]+"_simRank_model.sav")
    p_n2v=str("Models/"+name[0]+"_N2V_model.sav")

    try:
        scores = pickle.load( open( p_sim, "rb" ) )
        N2V_model=pickle.load( open( p_n2v, "rb" ) )
    except FileNotFoundError:
        print("Computing simrank ...")
        scores= simrank(G)
        pickle.dump( scores, open(p_sim, "wb" ) )

        print("Computing NODE2VEC ...")
        N2V_model=Node2Vec_model(G)
        pickle.dump( N2V_model, open( p_n2v, "wb" ) )

    # Set random state
    random.seed(time.time())
    degrees = dict(G.degree())
    must_constraints = set()
    cannot_constraints = set()
    # Load the ground-truth communities ..../ as oracle
    communities_groundtruth = read_communities(sys.argv[2])
    nodes = list(assigned_nodes(communities_groundtruth))
    node_map2 = {}
    for node in nodes:
        node_map2[str(node)] = set()
    for community in communities_groundtruth:
        for pair in itertools.combinations(community, 2):
            node_map2[str(pair[0])].add(str(pair[1]))
            node_map2[str(pair[1])].add(str(pair[0]))

    n = len(nodes)
    RP = float(sys.argv[3])
    max_id = int(max(nodes))

    # Print basic stats for nodes
    print("Nodes assigned to communities: %d" % len(nodes))
    print("Node ID range: [%d,%d]" % (int(min(nodes)), int(max_id)))
    unique_pairs = (n * (n - 1)) / 2
    print("Number of unique pairs of nodes: %d" % unique_pairs)
    print("max_id: %d" % max_id)
    budget = int(unique_pairs * (float(sys.argv[4])))
    print ('budget', budget)


    # Build the pairwise co-assignment matrix
    print("Counting possible constraints ...")
    S = np.zeros((max_id + 1, max_id + 1))
    for community in communities_groundtruth:
        # get all unique pairs of nodes in this community
        for i, j in itertools.combinations(list(community), 2):
            S[i, j] += 1
            S[j, i] += 1

    # Count the number of must link and cannot link pairs
    num_most, num_cannot = 0, 0

    # get all pairs of nodes
    for i, j in itertools.combinations(nodes, 2):
        # do this pair belong to a ground truth community?
        if S[i, j] > 0:
            num_most += 1
        else:
            num_cannot += 1

    print("Must pairs: %d" % num_most)
    print("Cannot pairs: %d" % num_cannot)
    print("Total pairs: %d" % (num_most + num_cannot))
    noise_percent = float(sys.argv[5])
    num_must_noise = round(num_most * noise_percent)
    num_cannot_noise = round(num_most * noise_percent)
    AE_Model_must=str(sys.argv[6])
    print('AE_Model_must',AE_Model_must)
    AE_Model_cannot=str(sys.argv[7])
    print('AE_Model_cannot',AE_Model_cannot)
    print ("noised must", num_must_noise)
    print ("noised cannot", num_must_noise)

    node_map, must_constraints_tobenoised, cannot_constraints_tobenoised = noise_const_method(communities_groundtruth,
                                                                                              num_must_noise,
                                                                                              num_must_noise, G)
    print('must_constraints_tobenoised', len(must_constraints_tobenoised), 'cannot_constraints_tobenoised', len(cannot_constraints_tobenoised))
    
    termination = int(sys.argv[8])
    AE_Model=sys.argv[9]
    min_chunck_size=int(sys.argv[10])
    RUN = sys.argv[11]

    ## Initialization: Apply unsupervised SLPA to generate set of initial communities

    print ('Initialization: Apply unsupervised SLPA to generate set of initial communities:')
    start_time_Initialization = time.time()
    communities = find_communities_Stand_SLPA(G, 100, RP)

    while len(communities) <=1:
        communities = find_communities_Stand_SLPA(G, 100, RP)
    print('initail len(communities)',len(communities))
    end_time_Initialization = time.time()
    elapsed = timedelta(seconds = end_time_Initialization - start_time_Initialization)
    hours, remainder = divmod(elapsed.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    print("--- Initialization time Running time = %02d:%02d:%02d" % (hours, minutes, seconds) )

    Total_Constraints = []
    must_constraints_Total = []
    cannot_constraints_Total = []
    must_constraints_Total_cleaned = set()
    cannot_constraints_Total_cleaned  = set()
    must_discard=set()
    cannot_discard=set()
    percentage=0.90
    iteration = 1
    AE_Model_var_must=None
    AE_Model_var_cannot=None

    start_time_Total = time.time()
    while len(Total_Constraints) < budget:
        ############# Phase 1: Apply Node Pair Selection method
        print ('iteration:', iteration)
        print ('Phase 1: Apply Node Pair Selection method')

        start_time_Phase1 = time.time()
        ImportantPairs = Node_Pair_Selection_Method_3(G, communities)

        print ('ImportantPairs', len(ImportantPairs))
        print ('Total_Constraints', len(Total_Constraints))
        temp = Total_Constraints
        Constraints = []
        print ('Constraints', len(Constraints))
        for pair in ImportantPairs:
            if not pair in Total_Constraints and (len(Total_Constraints) + len(Constraints)) <= budget:
                Constraints.append(pair)
        Total_Constraints = Total_Constraints + Constraints
        print ('Total_Constraints after', len(Total_Constraints))
        print ('Constraints after', len(Constraints))

        end_time_Phase1 = time.time()
        elapsed = timedelta(seconds = end_time_Phase1 - start_time_Phase1)
        hours, remainder = divmod(elapsed.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        print("--- Phase 1 time Running time = %02d:%02d:%02d" % (hours, minutes, seconds) )

        ############## Phase 2: Generate the pairwise constraints
        print ('Phase 2: Generate the pairwise constraints')
        start_time_Phase2 = time.time()
        must_constraints, cannot_constraints = selectCons(Constraints, node_map,must_constraints_tobenoised,cannot_constraints_tobenoised)
        must_constraints_Total=list(must_constraints_Total)
        cannot_constraints_Total=list(cannot_constraints_Total)
        print ('new chunck: must_constraints after labeling', len(must_constraints), 'new chunck: cannot_constraints after labeling', len(cannot_constraints))
        for pair in must_constraints:
            if not pair in must_constraints_Total and (not pair in must_constraints_Total_cleaned) :
                must_constraints_Total.append(pair)
                Must_Pair_With_Noise.add(pair)
        for pair in cannot_constraints :
            if not pair in cannot_constraints_Total and not pair in cannot_constraints_Total_cleaned:
                cannot_constraints_Total.append(pair)
                Cannot_Pair_With_Noise.add(pair)
        print ('must_constraints_Total after labeling', len(must_constraints_Total), 'cannot_constraints_Total after labeling',len(cannot_constraints_Total))
        
        end_time_Phase2 = time.time()
        elapsed = timedelta(seconds = end_time_Phase2 - start_time_Phase2)
        hours, remainder = divmod(elapsed.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        print("--- Phase 2 time Running time = %02d:%02d:%02d" % (hours, minutes, seconds) )
        print ('Phase 3: Cleaning the pairwise constraints')
        start_time_Phase3 = time.time()


        cond_must=int(len(must_constraints_Total)*0.50)
        cond_cannot=int(len(cannot_constraints_Total)*0.50)
        Flag_must=False
        Flag_cannot=False

        for i in range(3):
            if len(must_constraints_Total)>=min_chunck_size:            
                if len(must_constraints_Total)>cond_must:
                    must_constraints_Total_1, must_discard, AE_Model_var_must=Detect_noisy_pairwise_constraints_must(G,must_constraints_Total,scores,N2V_model,degrees,node_map2,0.95,AE_Model_cannot,must_discard,None)
                    must_constraints_Total=must_constraints_Total_1
                else:
                    Flag_must=True

            if len(cannot_constraints_Total)>=min_chunck_size:
                if len(cannot_constraints_Total)>cond_cannot: 
                    cannot_constraints_Total_1, cannot_discard, AE_Model_var_cannot=Detect_noisy_pairwise_constraints_cannot(G,cannot_constraints_Total,scores,N2V_model,degrees,node_map2,0.95,AE_Model_cannot,cannot_discard,None)
                    cannot_constraints_Total=cannot_constraints_Total_1
                else:
                    Flag_cannot=True

            if Flag_must==True and Flag_cannot==True:
                break

        end_time_Phase3 = time.time()

        elapsed = timedelta(seconds = end_time_Phase3 - start_time_Phase3)
        hours, remainder = divmod(elapsed.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        print("--- Phase 3 time Running time = %02d:%02d:%02d" % (hours, minutes, seconds))
        print ('Must_constraints_Total after cleaning ', len(must_constraints_Total), 'Cannot_constraints_Total after cleaning ',len(cannot_constraints_Total))
        must_constraints_Total_cleaned=set(must_constraints_Total_cleaned).union(set(must_constraints_Total))
        cannot_constraints_Total_cleaned=set(cannot_constraints_Total_cleaned).union(set(cannot_constraints_Total))
        print ('The whole set of cleaned must_constraints ', len(must_constraints_Total_cleaned), 'The whole set of cleaned cannot_constraints ',len(cannot_constraints_Total_cleaned))
        must_constraints_Total=list(must_constraints_Total_cleaned)
        cannot_constraints_Total=list(cannot_constraints_Total_cleaned)

        communities = {}
        ################ Phase 3: Apply PC-SLPA algorithm
        print ('Phase 4: Apply PC-SLPA algorithm')
        start_time_Phase4 = time.time()
        communities = find_communities(G, 100, RP, set(must_constraints_Total_cleaned), set(cannot_constraints_Total_cleaned))
        end_time_Phase4 = time.time()

        elapsed = timedelta(seconds = end_time_Phase4 - start_time_Phase4)
        hours, remainder = divmod(elapsed.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        print("--- Phase 4 time Running time = %02d:%02d:%02d" % (hours, minutes, seconds))
        iteration_copy=iteration
        iteration =iteration+1
        print('end iteration,Total_Constraints ',len(Total_Constraints))
        if ( len(Total_Constraints) >= budget or len(temp) == len(Total_Constraints) or iteration > 70 or (len(Constraints) < termination and iteration_copy>1)) and Total_Constraints != 0 :
            break

    print('out the loop')
    end_time_Total = time.time()
    elapsed = timedelta(seconds = end_time_Total - start_time_Total)
    hours, remainder = divmod(elapsed.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    print("--- All Running time = %02d:%02d:%02d" % (hours, minutes, seconds))
    print('must_constraints_Total', len(must_constraints_Total_cleaned), 'cannot_constraints_Total',len(cannot_constraints_Total_cleaned))
    print('communities',len(communities))
    start_time_Phase5 = time.time()


    condition_must=True
    condition_cannot=True
    temp_must=0
    temp_cannot=0
    final_must_discard_1=set()
    final_cannot_discard_1=set()
    while(True):
        must_discard_1=set()
        cannot_discard_1=set()
        whole_used_PC=len(must_constraints_Total_cleaned)+len(cannot_constraints_Total_cleaned)
        whole_used_must=len(must_constraints_Total_cleaned)
        whole_used_cannot=len(cannot_constraints_Total_cleaned)
        for k in range(7):
            print ('Start Cleaning discarded labels')
            if condition_must==True and len(must_discard)>=min_chunck_size:
                if i <= 4:
                    must_discard_11, must_discard_1, AE_Model_var_must=Detect_noisy_pairwise_constraints_must(G,list(must_discard),scores,N2V_model,degrees,node_map2,percentage,AE_Model_cannot,must_discard_1,AE_Model_var_must)
                    must_discard=must_discard_11
                    discard_labels_must=int((len(must_discard_1)/whole_used_must)*100)
                    if (discard_labels_must <= 20):
                        condition_must=False

            if condition_cannot==True and len(cannot_discard)>=min_chunck_size:
                cannot_discard_11, cannot_discard_1, AE_Model_var_cannot=Detect_noisy_pairwise_constraints_cannot(G,list(cannot_discard),scores,N2V_model,degrees,node_map2,percentage,AE_Model_cannot,cannot_discard_1,AE_Model_var_cannot)
                cannot_discard=cannot_discard_11
                discard_labels_cannot=int((len(cannot_discard_1)/whole_used_cannot)*100)
                if (discard_labels_cannot <= 20):
                    condition_cannot=False

        must_constraints_Total_cleaned=list(must_constraints_Total_cleaned)
        cannot_constraints_Total_cleaned=list(cannot_constraints_Total_cleaned)
        for pair in must_discard:
            if not pair in must_constraints_Total_cleaned:
                must_constraints_Total_cleaned.append(pair)

        for pair in cannot_discard:
            if not pair in cannot_constraints_Total_cleaned:
                cannot_constraints_Total_cleaned.append(pair)

        whole_used_PC=len(must_constraints_Total_cleaned)+len(cannot_constraints_Total_cleaned)
        whole_used_must=len(must_constraints_Total_cleaned)
        whole_used_cannot=len(cannot_constraints_Total_cleaned)
        discard_labels=((len(must_discard_1)+len(cannot_discard_1))/whole_used_PC)*100
        discard_labels_must=int((len(must_discard_1)/whole_used_must)*100)
        discard_labels_cannot=int((len(cannot_discard_1)/whole_used_cannot)*100)

        must_discard=must_discard_1
        cannot_discard=cannot_discard_1
        if (discard_labels_must <= 20):
            condition_must=False

        if (discard_labels_cannot <= 20):
            condition_cannot=False

        if (discard_labels_must <= 20 ) and (discard_labels_cannot <= 20) or ((temp_must==discard_labels_must) and (temp_cannot==discard_labels_cannot)) :
            break
        temp_must=discard_labels_must
        temp_cannot=discard_labels_cannot
        final_must_discard_1=final_must_discard_1.union(must_discard_1)
        final_cannot_discard_1=final_cannot_discard_1.union(cannot_discard_1)

    end_time_Phase5 = time.time()
    elapsed = timedelta(seconds = end_time_Phase5 - start_time_Phase5)
    hours, remainder = divmod(elapsed.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    print("--- time Running time = %02d:%02d:%02d" % (hours, minutes, seconds))

    print('final must_discard_1',len(final_must_discard_1))
    print('final cannot_discard_1',len(final_cannot_discard_1))
       
    must_constraints_Total_cleaned=list(must_constraints_Total_cleaned)
    cannot_constraints_Total_cleaned=list(cannot_constraints_Total_cleaned)
    for pair in must_discard:
        if not pair in must_constraints_Total_cleaned:
            must_constraints_Total_cleaned.append(pair)

    for pair in cannot_discard:
        if not pair in cannot_constraints_Total_cleaned:
            cannot_constraints_Total_cleaned.append(pair)

    for i in range(1):
        print ('Final_loop')
        must_constraints_Total_1, must_discard, AE_Model_var_must=Detect_noisy_pairwise_constraints_must(G,must_constraints_Total_cleaned,scores,N2V_model,degrees,node_map2,0.98,AE_Model_cannot,must_discard, AE_Model_var_must)
        cannot_constraints_Total_1, cannot_discard, AE_Model_var_cannot=Detect_noisy_pairwise_constraints_cannot(G,cannot_constraints_Total_cleaned,scores,N2V_model,degrees,node_map2,0.98,AE_Model_cannot,cannot_discard, AE_Model_var_cannot)
        must_constraints_Total_cleaned=must_constraints_Total_1
        cannot_constraints_Total_cleaned=cannot_constraints_Total_1

    communities = find_communities(G, 100, RP, set(must_constraints_Total_cleaned), set(cannot_constraints_Total_cleaned))
    for value in communities:
        for subvalue in list(communities[value]):
            print(subvalue, end=' ')
        print()
    fh.close()
   


if __name__ == "__main__":
    main()
