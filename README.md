# Semi-Supervised-SLPA

This repository contains supplementary material (code and data) for the paper: *"Active Semi-Supervised Overlapping Community Finding with Pairwise Constraints"* (2019, and the paper: *"Overlapping Community Finding with Noisy Pairwise Constraints".

The algorithm implementations require Python 3.x and the [NetworkX](https://networkx.github.io/) library.

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

### AC-SLPA with Cleaning Methods:

### To run AC-SLPA with Hybrid (Autoencoder - Encoder Function+IF)

```python AC_SLPA_denoise_hybrid.py network.dat community.dat R Max_budget Noise_Percentage AE_Model_Must AE_Model_Cannot Termination min_chunck_size```

Required arguments:

- network.dat: the network file as an edgelist
- community.dat: the ground truth communities to be used as an oracle
- R: a user-specified threshold R with the range [0, 1]. If the probability of seeing a particular label during the whole process is less than the given R, this label is deleted from a node’s memory.
- Max_budget: the maximum constraints budget for annotaion, given as precentage of the total constraints - e.g. 1% would be (0.01), 0.5% would be (0.005) etc
- Noise_Percentage: the percentage of noise added to constraints by randomly flipping the labels of a subset of must-link and cannot-link pairs.
- AE_Model_Must: the autoencoder architecture for cleaning must-link constraints.
- AE_Model_Cannot: the autoencoder architecture for cleaning cannot-link constraints.
- Termination: the minimum number of pairwise constraints are chosen at each iteration, if it is less than the given 'Termination' value, terminate the loop and continue to use the selected pairwise constraints. 
- min_chunck_size: the minimum size of pairwise constraints chunck chosen at each iteration to be used with autoencoder models


### To run AC-SLPA with Encoder Function + Outlier detection (SVM-IF)

```python AC_SLPA_denoise_AE_SVM_ISO.py network.dat community.dat R Max_budget Noise_Percentage AE_Model_Must AE_Model_Cannot Termination min_chunck_size```

Required arguments:

- network.dat: the network file as an edgelist
- community.dat: the ground truth communities to be used as an oracle
- R: a user-specified threshold R with the range [0, 1]. If the probability of seeing a particular label during the whole process is less than the given R, this label is deleted from a node’s memory.
- Max_budget: the maximum constraints budget for annotaion, given as precentage of the total constraints - e.g. 1% would be (0.01), 0.5% would be (0.005) etc
- Noise_Percentage: the percentage of noise added to constraints by randomly flipping the labels of a subset of must-link and cannot-link pairs.
- AE_Model_Must: the autoencoder architecture for cleaning must-link constraints.
- AE_Model_Cannot: the autoencoder architecture for cleaning cannot-link constraints.
- Termination: the minimum number of pairwise constraints are chosen at each iteration, if it is less than the given 'Termination' value, terminate the loop and continue to use the selected pairwise constraints. 
- min_chunck_size: the minimum size of pairwise constraints chunck chosen at each iteration to be used with autoencoder models


### To run AC-SLPA with Autoencoders(AE)

```python AC_SLPA_denoise_AE.py network.dat community.dat R Max_budget Noise_Percentage AE_Model_Must AE_Model_Cannot Termination min_chunck_size```

Required arguments:

- network.dat: the network file as an edgelist
- community.dat: the ground truth communities to be used as an oracle
- R: a user-specified threshold R with the range [0, 1]. If the probability of seeing a particular label during the whole process is less than the given R, this label is deleted from a node’s memory.
- Max_budget: the maximum constraints budget for annotaion, given as precentage of the total constraints - e.g. 1% would be (0.01), 0.5% would be (0.005) etc
- Noise_Percentage: the percentage of noise added to constraints by randomly flipping the labels of a subset of must-link and cannot-link pairs.
- AE_Model_Must: the autoencoder architecture for cleaning must-link constraints.
- AE_Model_Cannot: the autoencoder architecture for cleaning cannot-link constraints.
- Termination: the minimum number of pairwise constraints are chosen at each iteration, if it is less than the given 'Termination' value, terminate the loop and continue to use the selected pairwise constraints. 
- min_chunck_size: the minimum size of pairwise constraints chunck chosen at each iteration to be used with autoencoder models

### To run AC-SLPA with Outlier detection only (SVM-IF)

```python AC_SLPA_denoise_SVM_ISO.py network.dat community.dat R Max_budget Noise_Percentage Termination```

Required arguments:

- network.dat: the network file as an edgelist
- community.dat: the ground truth communities to be used as an oracle
- R: a user-specified threshold R with the range [0, 1]. If the probability of seeing a particular label during the whole process is less than the given R, this label is deleted from a node’s memory.
- Max_budget: the maximum constraints budget for annotaion, given as precentage of the total constraints - e.g. 1% would be (0.01), 0.5% would be (0.005) etc
- Noise_Percentage: the percentage of noise added to constraints by randomly flipping the labels of a subset of must-link and cannot-link pairs.
- Termination: the minimum number of pairwise constraints are chosen at each iteration, if it is less than the given 'Termination' value, terminate the loop and continue to use the selected pairwise constraints. 
