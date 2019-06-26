import numpy as np
import networkx as nx
import random
from collections import defaultdict
import sys

"""
Sample usage:

python Standard_SLPA.py network.dat R

"""

def find_communities(G, T, r):

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

def main():
    fh = open(sys.argv[1], "rb")
    G = nx.read_edgelist(fh)
    G.nodes()
    communities = {}
    communities = find_communities(G, 100, float(sys.argv[2]))
    for value in communities:
        value_sort=sorted(list(communities[value]))
        for subvalue in value_sort:
            print(subvalue, end=' ')
        print()
    fh.close()
    
if __name__ == "__main__":
    main()
