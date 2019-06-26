# Semi-Supervised-SLPA

This repository contains supplementary material (code and data) for the paper: *"Active Semi-Supervised Overlapping Community Finding with Pairwise Constraints"* (2019).

### SLPA:

To run an implementation of the original Speaker-listener Label Propagation Algorithm (SLPA) from Xie et al (2011):

```python Standard_SLPA.py network.dat R```

Required arguments:

- network.dat: the network file as an edgelist
- R: a user-specified threshold R with the range [0, 1]. If the probability of seeing a particular label during the whole process is less than the given R, this label is deleted from a node’s memory.

### PC-SLPA:

To run Pairwise Constrained SLPA:

```python PC_SLPA.py network.dat community.dat num_must num_cannot num_initial R```

Required arguments:

- network.dat: the network file as an edgelist
- community.dat: the ground truth communities to be used as an oracle
- num_must: number of must-link constraints
- num_cannot: number of cannot-link constraints
- num_initial: number of constraints in theinitial small set chosen at random to be used for constraint processing
- R: a user-specified threshold R with the range [0, 1]. If the probability of seeing a particular label during the whole process is less than the given R, this label is deleted from a node’s memory.


### AC-SLPA:

To run Active Semi-supervised SLPA:

```python AC_SLPA.py network.dat community.dat R Max_budget```

Required arguments:

- network.dat: the network file as an edgelist
- community.dat: the ground truth communities to be used as an oracle
- R: a user-specified threshold R with the range [0, 1]. If the probability of seeing a particular label during the whole process is less than the given R, this label is deleted from a node’s memory.
- Max_budget: the maximum constraints budget for annotaion, given as precentage of the total constraints - e.g. 1% would be (0.01), 0.5% would be (0.005) etc
