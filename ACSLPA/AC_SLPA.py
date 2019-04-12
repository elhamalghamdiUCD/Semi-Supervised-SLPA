import numpy as np
import time
import itertools
import random
import sys
import networkx as nx
from operator import itemgetter
from collections import defaultdict




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

        node_One_labels_values = frozenset((memory[str(pair[0])]).values())
        node_Two_labels_values = frozenset((memory[str(pair[1])]).values())

        node_One_max_label_value = max(node_One_labels_values)
        node_Two_max_label_value = max(node_Two_labels_values)

        node_One_max_label_keys = [k for k, v in (memory[str(pair[0])]).items() if v == node_One_max_label_value]
        node_Two_max_label_keys = [k for k, v in (memory[str(pair[1])]).items() if v == node_Two_max_label_value]
        commenMax = set(node_One_max_label_keys).intersection(set(node_Two_max_label_keys))
        

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
                checker2 = False
                for node in nodes_of_label_One:
                    pair = frozenset([int(node), int(nodeTwo)])
                    if pair in cannot_constraints:
                        checker2 = True
                        break

                if checker2 == False:
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
    # print("Loading communities from %s ..." % in_path)
    communities = []
    fin = open(in_path, "r")
    for line in fin.readlines():
        community = set()
        for node in line.strip().split(" "):
            community.add(int(node))
        if len(community) > 0:
            communities.append(community)
    fin.close()
    # print("Number of communities: %d" % len(communities))
    return communities


def assigned_nodes(communities):
    """
    Get all nodes assigned to at least one community.
    """
    assigned = set()
    for community in communities:
        assigned = assigned.union(community)
    return assigned


def selectCons(communities_groundtruth, ImportantPairs):
    nodes = list(assigned_nodes(communities_groundtruth))
    n = len(nodes)
    node_map = {}
    for node in nodes:
        node_map[node] = set()
    for community in communities_groundtruth:
        for pair in itertools.combinations(community, 2):
            node_map[int(pair[0])].add(int(pair[1]))
            node_map[int(pair[1])].add(int(pair[0]))

    must_constraints = set()
    cannot_constraints = set()
    start_time = time.time()

    for pair in ImportantPairs:
        pairlist = list(pair)
        # is this a must-link?
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

    ##select overlapping nodes
    overlapping_nodes= find_overlapping_nodes(communities)


    ##select boundary nodes
    boundary_nodes=set()
    for community in communities.values():
        B_nodes = Find_Boundary_Node(G, community)
        boundary_nodes.update(B_nodes)

    Selected_Nodes =list(boundary_nodes)+ list(overlapping_nodes)

    for node in list(Selected_Nodes):
        selected_comm = []
        neighbors=list(G.neighbors(str(node)))
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


def main():
    fh = open(sys.argv[1], "rb")
    G = nx.read_edgelist(fh)
    n=len(G.nodes())
    unique_pairs = (n * (n - 1)) / 2
    persent=int(unique_pairs*(float(sys.argv[4])))
    print ('persent',persent)

    # Set random state
    random.seed(time.time())

    must_constraints = set()
    cannot_constraints = set()
    # Load the ground-truth communities ..../ as oracle
    communities_groundtruth = read_communities(sys.argv[2])
    
    ## Initialization: Apply unsupervised SLPA to generate set of initial communities
    print ('Initialization: Apply unsupervised SLPA to generate set of initial communities:')
    start_time_Initialization = time.time()
    communities = find_communities_Stand_SLPA(G, 100, float(sys.argv[3]))
    end_time_Initialization = time.time()
    print("--- %s Initialization time in seconds ---" % (end_time_Initialization - start_time_Initialization))
    i=1
    Total_Constraints=[]
    must_constraints_Total=[]
    cannot_constraints_Total=[]
    while  len(Total_Constraints) < persent:
        start_time_Total = time.time()
        ############# Phase 1: Apply Node Pair Selection method
        print ('iteration:', i)
        print ('Phase 1: Apply Node Pair Selection method')
        start_time_Phase1 = time.time()
        ImportantPairs = Node_Pair_Selection_Method_3(G, communities)

        print ('ImportantPairs', len(ImportantPairs))
        print ('Total_Constraints',len(Total_Constraints))
        temp=Total_Constraints
        Constraints=[]
        print ('Constraints',len(Constraints))
        for pair in ImportantPairs:
            if not pair in  Total_Constraints and (len(Total_Constraints)+len(Constraints)) <=persent:
                Constraints.append(pair)
        Total_Constraints=Total_Constraints+Constraints
        print ('Total_Constraints after',len(Total_Constraints))
        print ('Constraints after',len(Constraints))

        end_time_Phase1 = time.time()
        print("--- %s Phase 1 time in seconds ---" % (end_time_Phase1 - start_time_Phase1))

        ############## Phase 2: Generate the pairwise constraints
        print ('Phase 2: Generate the pairwise constraints')
        start_time_Phase2 = time.time()
        must_constraints, cannot_constraints = selectCons(communities_groundtruth, Constraints)
        print ('must_constraints',len(must_constraints),'cannot_constraints',len(cannot_constraints))
        for pair in must_constraints:
            must_constraints_Total.append(pair)
        for pair in cannot_constraints:
            cannot_constraints_Total.append(pair)
        print ('must_constraints_Total', len(must_constraints_Total), 'cannot_constraints_Total', len(cannot_constraints_Total))
        end_time_Phase2 = time.time()


        print("--- %s Phase 2 time in seconds ---" % (end_time_Phase2 - start_time_Phase2))
        communities = {}
        ################ Phase 3: Apply PC-SLPA algorithm
        print ('Phase 3: Apply PC-SLPA algorithm')
        start_time_Phase3 = time.time()
        communities = find_communities(G, 100, float(sys.argv[3]), set(must_constraints_Total), set(cannot_constraints_Total))
        end_time_Phase3 = time.time()
        print("--- %s Phase 3 time in seconds ---" % (end_time_Phase3 - start_time_Phase3))
        end_time_Total = time.time()
        print("--- %s Total time in seconds ---" % (end_time_Total - start_time_Total))
        print()
        i+=1
        if len(Total_Constraints) >= persent or len(temp)==len(Total_Constraints):
            break


    fout = open(sys.argv[5], "w")
    for community in communities:
        for com in list(communities[community]):
            fout.write("%s " % (com))
        fout.write("\n")
    fout.close()
    """
    for value in communities:
        for subvalue in list(communities[value]):
            print(subvalue, end=' ')
        print()
    fh.close()
    """


if __name__ == "__main__":
    main()
