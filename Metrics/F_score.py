import sys
from collections import defaultdict

def find_overlapping_nodes( communities ):
	"""
	Find the set of all nodes that have been assigned to more than one community
	in the specified community set.
	"""
	community_counts = defaultdict(int)	
	for comm in communities:
		for node_index in comm:
			community_counts[node_index] += 1
	overlapping_nodes = set()
	for node_index in community_counts:
		if community_counts[node_index] > 1:
			overlapping_nodes.add( node_index )
	return overlapping_nodes

def calc_fscore( comm_ground, comm_algorithm ):
	"""
	Calculates the F1-score, based on the identification of nodes that are assigned
	to more than one community in a ground truth community set.
	"""
	ground_overlapping_nodes = find_overlapping_nodes(comm_ground)
	algorithm_overlapping_nodes = find_overlapping_nodes(comm_algorithm)
	# measure the overlap
	inter_overlapping = len( ground_overlapping_nodes.intersection(algorithm_overlapping_nodes) )
	# precision is the fraction of identified overlapping nodes that are correct
	if len(algorithm_overlapping_nodes) == 0:
		precision = 0
	else:
		precision = float(inter_overlapping)/len(algorithm_overlapping_nodes)
	# recall is the fraction of all overlapping nodes that were found
	if len(ground_overlapping_nodes) == 0:
		recall = 0
	else:
		recall = float(inter_overlapping)/len(ground_overlapping_nodes)
	# combine precision and recall into F1 score, avoiding division by zero
	if precision == 0 or recall == 0:
		fscore = 0
	else:
		fscore = ((2*precision*recall)/(precision+recall))
	return (fscore,precision,recall)

# --------------------------------------------------------------
"""
    python3 F_score.py Ground_truth_communities resulted_communities
"""

def main():
	comm_ground = []
	with open(sys.argv[1],"r") as fin:
		for line in fin.readlines():
			comm_ground.append( line.strip().split( " ") )

	comm_algorithm = []
	with open(sys.argv[2],"r") as fin:
		for line in fin.readlines():
			comm_algorithm.append( line.strip().split( " ") )

	fscore,precision,recall = calc_fscore( comm_ground, comm_algorithm )
	print(fscore)

# --------------------------------------------------------------

if __name__ == "__main__":
	main()
