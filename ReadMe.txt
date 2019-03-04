

SLPA:

python Standerd_SLPA.py network.dat R

-network.dat: the network file as edgelist
-R: a given threshold R with the range [0, 1]. If the probability of seeing a particular label during the whole process is less than the given R, this label is deleted from a node’s memory.
---------------------------------------------------------------------------------------------------------------------

PCSLPA:

python PC_SLPA.py network.dat community.dat num_must num_cannot num_initial R

-network.dat: the network file as edgelist
-community.dat: the ground truth communities to be used as an oracle
-num_must: number of must-link constraints
-num_cannot: number of cannot-link constraints
-num_initial: number of constraints as initial small set chosen at random to be used for constraints processing
-R: a given threshold R with the range [0, 1]. If the probability of seeing a particular label during the whole process is less than the given R, this label is deleted from a node’s memory.

---------------------------------------------------------------------------------------------------------------------

ACSLPA:

python AC_SLPA.py network.dat community.dat R Max_budget

-network.dat: the network file as edgelist
-community.dat: the ground truth communities to be used as an oracle
-R: a given threshold R with the range [0, 1]. If the probability of seeing a particular label during the whole process is less than the given R, this label is deleted from a node’s memory.
-Max_budget: the max constraints budget for annotaion, ( given as precentage of the total constraints - ex 1% would be (0.01) or 0.5% would be (0.005))
