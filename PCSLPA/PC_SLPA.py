import numpy as np
import time
import itertools
import random
import sys
import re
import networkx as nx
import operator
import copy
import collections
from collections import defaultdict
from random import randint

"""
    Sample usage:

    python PW_SLPA.py network.dat community.dat num_must num_cannot num_initial

    -usually   num_initial = 0.45 * num_must  (with network size 1000)
               num_initial = 0.33 * num_must  (with karat network)

"""

def find_communities(G, T, r, must_constraints, cannot_constraints):


    ## the initialization step:
    memory = {i: {i: 1} for i in G.nodes()}

    ## Must-link processing: For each pair of nodes that have a must-link relationship,
    # the two nodes ex-change labels (i.e. update each node’s memory with the other node’s label).

    G_must_links = nx.Graph()  # Create graph of must-link pairs, where edges are the must-link relationships,
    # then build a map of cliques for each node (for each node find all the nodes that have a must link with this node) to be used in the next stage
    for pair in must_constraints:
        pair = list(pair)
        if str (pair[1]) in memory[str (pair[0])]:
            memory[str (pair[0])][str (pair[1])] += 1
        else:
            memory[str (pair[0])][str (pair[1])] = 1

        if str (pair[0]) in memory[str (pair[1])]:
            memory[str (pair[1])][str (pair[0])] += 1
        else:
            memory[str (pair[1])][str (pair[0])] = 1
        G_must_links.add_edge(pair[0],pair[1])

    ## find max cliques
    G_must_links_clique = list(nx.find_cliques(G_must_links))

    ## build a map of cliques for each node
    node_map_cliq = {}
    for node in list(G.nodes()):
        node_map_cliq[node] = set()

    for cliq in G_must_links_clique:
        for pair in itertools.combinations(cliq, 2):
            node_map_cliq[str(pair[0])].add(str(pair[1]))
            node_map_cliq[str(pair[1])].add(str(pair[0]))


    ## Nodes membership
    communities = {}
    for node, mem in list(memory.items()):
        for label in list(mem.keys()):
            if label in communities:
                communities[label].add(node)
            else:
                communities[label] = set([node])

    ## The evaluation step:
    for t in range(T):
        listenersOrder = list(G.nodes())
        np.random.shuffle(listenersOrder)

        for listener in listenersOrder:
            ##include must-link nodes in speakers set
            must_link_set=node_map_cliq[listener]
            if len(must_link_set) != 0:
                speakers = list(set( G[listener].keys()).union(node_map_cliq[listener]))
            else:
                speakers = list(G[listener].keys())

            ##delete cannot-link nodes in speakers set
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
                labels[list(memory[speaker].keys())[np.random.multinomial(1, [freq / total for freq in memory[speaker].values()]).argmax()]] += 1


            # Listener Rule
            random_key = get_rnd_key_of_max_value(labels)
            acceptedLabel = random_key

            # get all the nodes that assigned to 'acceptedLabel', to check if there is any cannot-link between any of these nodes and listener,
            #if there is , the listener will reject the label
            nodes_of_label_acceptedLabel=list(communities[acceptedLabel])

            # Update listener memory
            checker3 = False
            for node in nodes_of_label_acceptedLabel:
                pair=frozenset([int(node), int(listener)])
                if pair in cannot_constraints:
                    checker3=True
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


    # Find nodes membership
    communities = {}
    for node, mem in list(memory.items()):
        for label in list(mem.keys()):
            if label in communities:
                communities[label].add(node)
            else:
                communities[label] = set([node])

    ##The constraint processing:

    #must-link processing: For each must-link pair, compare the memories of the two nodes to ensure they both share the
    # same highest occurrence frequency label. If they do not, both nodes exchange their most frequently-occurring labels
    # with each other under a condition that each node does not has a cannot link relationship with any node assign to that label.
    for pair in must_constraints:
        pair = list(pair)
        nodeOne_Set = set(memory[str (pair[0])])
        nodeTwo_Set = set(memory[str (pair[1])])
        nodeOne=str (pair[0])
        nodeTwo=str (pair[1])

        node_One_labels_values=frozenset((memory[str (pair[0])]).values())
        node_Two_labels_values=frozenset((memory[str (pair[1])]).values())

        node_One_max_label_value=max(node_One_labels_values)
        node_Two_max_label_value=max(node_Two_labels_values)

        node_One_max_label_keys=[k for k, v in (memory[str (pair[0])]).items() if v == node_One_max_label_value]
        node_Two_max_label_keys=[k for k, v in (memory[str (pair[1])]).items() if v == node_Two_max_label_value]
        commenMax=set(node_One_max_label_keys).intersection(set(node_Two_max_label_keys))

       #if it is not empty mean, they share the highest occurrence frequency label . so continue to the next
        if len(commenMax)!=0:
            continue
        else:
            ii=0
            done_Two=True
            while done_Two and ii< len(node_Two_max_label_keys):
                nodes_of_label_Two= list(communities[node_Two_max_label_keys[ii]])
                checker = False
                for node in nodes_of_label_Two:
                    pair=frozenset([int(node), int(nodeOne)])
                    if pair in cannot_constraints:
                        checker=True
                        break

                if checker == False:
                    if str (node_Two_max_label_keys[ii]) in memory[nodeOne]:
                        memory[nodeOne][str (node_Two_max_label_keys[ii])] +=  memory[nodeTwo][str (node_Two_max_label_keys[ii])]
                        done_Two=False
                    else:
                        memory[nodeOne][str (node_Two_max_label_keys[ii])] =  memory[nodeTwo][str (node_Two_max_label_keys[ii])]
                        done_Two=False
                ii=ii+1

            ij=0
            done_One=True
            while done_One and ij<len(node_One_max_label_keys):
                nodes_of_label_One= list(communities[node_One_max_label_keys[ij]])

                checker2 = False
                for node in nodes_of_label_One:
                    pair=frozenset([int(node), int(nodeTwo)])
                    if pair in cannot_constraints:
                        checker2=True
                        break

                if checker2 == False:
                    if str (node_One_max_label_keys[ij]) in memory[nodeTwo]:
                        memory[nodeTwo][str (node_One_max_label_keys[ij])] += memory[nodeOne][str (node_One_max_label_keys[ij])]
                        done_One=False
                    else:
                        memory[nodeTwo][str (node_One_max_label_keys[ij])] = memory[nodeOne][str (node_One_max_label_keys[ij])]
                        done_One=False
                ij=ij+1

    # Processing Cannot-Link constraints:compare the memories of cannot-link pairs, if there is common label, delete the one with less frequency.
    for pair in cannot_constraints:
        pair = list(pair)
        nodeOne_Set = set(memory[str (pair[0])])
        nodeTwo_Set = set(memory[str (pair[1])])
        commenKeys = nodeOne_Set.intersection(nodeTwo_Set)

        if len(commenKeys)==0:
            continue
        else:
            for label in list(commenKeys):
                if memory[str (pair[0])][label] > memory[str (pair[1])][label]:
                    del memory[str (pair[1])][label]
                elif memory[str (pair[0])][label] < memory[str (pair[1])][label]:
                    del memory[str (pair[0])][label]
                else:
                    labels=[]
                    labels.append(pair[0])
                    labels.append(pair[1])
                    random_key = random.choice(labels)
                    del memory[str (random_key)][label]

    ## post-processing step:
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
    max_label_value = max(label_values)
    max_label_keys = [k for k, v in labels.items()
                      if v == max_label_value]
    # Pick a random key of a maximum value.
    random_key = random.choice(max_label_keys)
    return random_key

def read_communities( in_path ):
    """
    Read communities from the specified file.
    We assume each node identifier is an integer, and there is one community per line.
    """
    # print("Loading communities from %s ..." % in_path)
    communities = []
    fin = open(in_path, "r")
    for line in fin.readlines():
        community = set()
        for node in line.strip().split(" "):
            community.add( int(node) )
        if len(community) > 0:
            communities.append( community )
    fin.close()
    return communities

def assigned_nodes( communities ):
    """
    Get all nodes assigned to at least one community.
    """
    assigned = set()
    for community in communities:
        assigned = assigned.union(community)
    return assigned

def main():
    fh = open(sys.argv[1], "rb")
    G = nx.read_edgelist(fh)
    num_must = int(sys.argv[3])
    num_cannot = int(sys.argv[4])
    num_initial = int(sys.argv[5])

    # Set random state
    random.seed(time.time())
    # print("Random seed: %s" % time.time())

    # Load the ground-truth communities
    communities_groundtruth = read_communities(sys.argv[2])
    nodes = list(assigned_nodes(communities_groundtruth))
    n = len(nodes)
    # print("Nodes assigned to communities: %d" % len(nodes))

    # Build an assignment map for all nodes
    # print("Building node assignment map ...")
    node_map = {}
    for node in nodes:
        node_map[node] = set()
    for community in communities_groundtruth:
        for pair in itertools.combinations(community, 2):
            node_map[pair[0]].add(pair[1])
            node_map[pair[1]].add(pair[0])

    # Select the constraints---------------------------------------------------------------
    # print("Target: %d must-link, %d cannot-link constraints" % (num_must, num_cannot))
    must_constraints = set()
    cannot_constraints = set()
    start_time = time.time()
    iteration = 0
    while (len(must_constraints) < num_must) or (len(cannot_constraints) < num_cannot):
        iteration += 1
        # print("- Iteration %d" % iteration)
        # Choose the initial set
        must_initial_constraints = set()
        cannot_initial_constraints = set()
        while (len(must_initial_constraints) < num_initial) or (len(cannot_initial_constraints) < num_initial):
            node_index1, node_index2 = random.randint(0, n - 1), random.randint(0, n - 1)
            # ignore self-constraints
            if node_index1 == node_index2:
                continue
            node1, node2 = nodes[node_index1], nodes[node_index2]
            pair = frozenset([node1, node2])
            # is this a must-link?
            if node2 in node_map[node1]:
                # do we have enough already? also avoid duplicates
                if (len(must_initial_constraints) < num_initial) and (not pair in must_initial_constraints) and (
                        not pair in must_constraints):
                    must_initial_constraints.add(pair)
            # otherwise must be cannot-link
            else:
                # do we have enough already? also avoid duplicates
                if (len(cannot_initial_constraints) < num_initial) and (
                        not pair in cannot_initial_constraints) and (not pair in cannot_constraints):
                    cannot_initial_constraints.add(pair)
        # print("Selected initial constraints: %d must-link, %d cannot-link" % (len(must_initial_constraints), len(cannot_initial_constraints)))

        # Add new must-link constraints, but do not go over the required amount
        must_initial_constraints = list(must_initial_constraints)
        while len(must_initial_constraints) > 0 and len(must_constraints) < num_must:
            must_constraints.add(must_initial_constraints.pop())
        # Add new cannot-link constraints, but do not go over the required amount
        cannot_initial_constraints = list(cannot_initial_constraints)
        while len(cannot_initial_constraints) > 0 and len(cannot_constraints) < num_cannot:
            cannot_constraints.add(cannot_initial_constraints.pop())
        # print("Now have %d must-link/ %d  Target must-link , %d cannot-link / %d Target cannot-link " % (len(must_constraints), num_must, len(cannot_constraints), num_cannot))

        # Find all the possible overlapping pairs in this initial must-link set to query the oracle about their relationship
        # print("Expanding initial constraints ...")
        list_must = list(must_constraints)
        num_current_must = len(list_must)
        num_expanded_must, num_expanded_cannot = 0, 0
        np = 0
        for combo in itertools.combinations(list_must, 2):
            pair_i, pair_j = combo[0], combo[1]
            np += 1
            # do the two pairs share a node?
            new_pair = pair_i.symmetric_difference(pair_j)
            if len(new_pair) == 2:
                node1, node2 = list(new_pair)
                # query oracle for this pair - is it a must-link?
                if node2 in node_map[node1]:
                    if (len(must_constraints) < num_must) and (not new_pair in must_constraints):
                        must_constraints.add(new_pair)
                        num_expanded_must += 1
                # otherwise a cannot link
                else:
                    if (len(cannot_constraints) < num_cannot) and (not new_pair in cannot_constraints):
                        cannot_constraints.add(new_pair)
                        num_expanded_cannot += 1
        # enough?
        if len(must_constraints) >= num_must and len(cannot_constraints) >= num_cannot:
            # print("Found sufficient constraints")
            break
        # print("Checked %d pairs" % np)
        # print("Expansion added %d must-link, %d cannot-link" % (num_expanded_must, num_expanded_cannot))
        # print("Now have %d must-link/ %d  Target must-link , %d cannot-link / %d Target cannot-link " % (len(must_constraints), num_must, len(cannot_constraints), num_cannot))
    # --------------------------------------------------------------

    communities = {}
    communities =find_communities(G, 100,  float(sys.argv[6]),must_constraints,cannot_constraints)
    for value in communities:
        for subvalue in list(communities[value]):
            print(subvalue, end=' ')
        print()
    fh.close()
if __name__ == "__main__":
    main()
