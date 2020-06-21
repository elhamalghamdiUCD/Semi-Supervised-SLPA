# node2vec

*This is a Python3 implementation of Stanford University's node2vec model*

## General Methodology of node2vec

1. Compute transition probabilities for all the nodes. (2nd order Markov chain)

2. Generate biased walks based on probabilities

3. Generate embeddings with SGD


### Pre-requisites

Install pre-reqs by running the following command:
`pip3 install -r req.txt`

## Usage

To run node2vec with default arguments, execute the following command from the home directory:
`python3 src/main.py`


A full list of command line arguments are shown by entering:
```
python3 src/main.py -h
```

```
usage: main.py [-h] [--input INPUT] [--output OUTPUT] [--p P] [--q Q]
               [--walks WALKS] [--length LENGTH] [--d D] [--window WINDOW]
               [--workers WORKERS] [--directed]

node2vec implementation

optional arguments:
  -h, --help         show this help message and exit
  --input INPUT      Path for input edgelist
  --output OUTPUT    Path for saving output embeddings
  --p P              Return parameter
  --q Q              In-out parameter
  --walks WALKS      Walks per node
  --length LENGTH    Length of each walk
  --d D              Dimension of output embeddings
  --window WINDOW    Window size for word2vec
  --workers WORKERS  Number of workers to assign for random walk and word2vec
  --directed         Flad to specify if graph is directed. Default is undirected. 
```

*Note: Zachary's Karate club network is used by default if no argument is provided for the input flag. Do check the default values in main.py*


### Example Usage:

To generate embeddings for Zachary's Karate club network with custom arguments, the following can be used
```
python3 src/main.py --p 0.4 --q 1 --walks 20 --length 80 --d 256
```


### Consolidated report with performance benchmarks are included in node2vec_report.pdf


**References:**

[node2vec: Scalable Feature Learning for Networks - Aditya Grover, Jure Leskovec](https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf)



Contact me at apoorva.v94@gmail.com


