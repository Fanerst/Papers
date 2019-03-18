# Paper Repository
Papers of my interest

## Learning

### Learning theory
- Geometrical and Statistical Properties of Systems of Linear Inequalities with Applications in Pattern Recognition [[paper link](https://ieeexplore.ieee.org/document/4038449)]

### Generative models

#### Autoregressive models
- Fast Generation for Convolutional Autoregressive Models [[paper link](https://arxiv.org/abs/1704.06001)]  
   <font color=gray>
  **Comments**: proposed a way to accelerate autoregreesive models like wave net and pixelcnn, need second thought about whether it can be used into VAN </font>

#### Flow-based networks

- Normalizing Flow [[paper link](https://arxiv.org/abs/1505.05770)]
- Monge-Ampere Flow [[paper link](https://arxiv.org/abs/1809.10188)]
- Neural Ordinary Differential Equations [[paper link](https://arxiv.org/abs/1806.07366)]
- Density Estimation using Real NVP [[paper link](https://arxiv.org/pdf/1605.08803.pdf)]
- Improved Variational Inference with Inverse Autoregressive Flow [[paper link](https://arxiv.org/abs/1606.04934)]
- Glow: Generative Flow with Invertible 1×1 Convolutions [[paper link](https://arxiv.org/abs/1807.03039)]

### Unclassified

- Variational discriminator bottleneck: improving imitation learning, inverse RL, and GANs by constraining information flow [[paper link](https://arxiv.org/abs/1810.00821)]

### Graph Neural Networks

#### Reviews
- Graph Neural Networks: A Review of Methods and Applications [[paper link](https://arxiv.org/abs/1812.08434)]
- A Comprehensive Survey on Graph Neural Networks [[paper link](https://arxiv.org/pdf/1901.00596v1.pdf)] *to read
- Relational inductive biases, deep learning, and graph networks [[paper link](https://arxiv.org/abs/1806.01261)] *to read

#### Graph Convolutional Networks
- Semi-Supervised Classification with Graph Convolutional Networks(GCN) [[paper link](https://arxiv.org/abs/1609.02907)]
- Simplifying Graph Convolutional Networks(SGCN) [[paper link](https://arxiv.org/pdf/1902.07153v1.pdf)]
- HOW POWERFUL ARE GRAPH NEURAL NETWORKS?(GRAPH ISOMORPHISM NETWORK) [[paper link](https://arxiv.org/pdf/1810.00826.pdf)]
- GRAPH WAVELET NEURAL NETWORK [[paper link](https://openreview.net/pdf?id=H1ewdiR5tQ)]
- Predict then Propagate: Graph Neural Networks meet Personalized PageRank [[paper link](https://arxiv.org/abs/1810.05997)]
- Graph Attention Networks [[paper link](https://arxiv.org/abs/1710.10903)]

#### Relational networks

- A simple neural network module for relational reasoning(RN) [[paper link](https://arxiv.org/abs/1706.01427)]
- Recurrent Relational Networks [[paper link](https://arxiv.org/abs/1711.08028)]

#### GNN for combinatorial optimization problems
- Learning Combinatorial Optimization Algorithms over Graphs [[paper link](https://arxiv.org/abs/1704.01665)]  
  <font color=gray>
  **Comments**: pretty interesting, but don't understand the point for now, probably because of lacking basic knownledge about reinforcement learning. </font>

- Coloring Big Graphs with AlphaGoZero [[paper link](https://arxiv.org/abs/1902.10162)]

#### GNN adversial attack
- Adversarial Attacks on Neural Networks for Graph Data [[paper link](https://arxiv.org/pdf/1805.07984.pdf)]
- Can Adversarial Network Attack be Defended? [[paper link](https://arxiv.org/pdf/1903.05994.pdf)]

### Topological interpretation of ML
- Topology of Learning in Artificial Neural Networks [[paper link](https://arxiv.org/abs/1902.08160)]  
  <font color=gray> 
  **Commments**: find a way to visualize evolution of weights through training, hinting that the learning ability comes from branching off of weights </font>

## Thermodynamics and Statistical Mechanics

- Stochastic Thermodynamic Interpretation of Information Geometry [[paper link](https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.121.030605), [support material link](https://journals.aps.org/prl/supplemental/10.1103/PhysRevLett.121.030605/supplemenatry_information.pdf)]

### Replica symmetry breaking
- Infinite Number of Order Parameters for Spin-Glasses [[paper link](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.43.1754)] *to read
- Order Parameter for Spin-Glasses [[paper link](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.50.1946)] *to read
- The order parameter for spin glasses: a function on the interval 0-1 [[paper link](http://iopscience.iop.org/article/10.1088/0305-4470/13/3/042/meta)] *to read

## Message Passing

### Belief propagation

- Understanding Belief Propagation and its Generalizations [[paper link](http://www.merl.com/publications/TR2001-22)]


## Approximate Inference

### ADATAP
- Tractable Approximations for Probabilistic Models: The Adaptive Thouless-Anderson-Palmer Mean Field Approach [[paper link](https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.86.3695)]
- Adaptive and self-averaging Thouless-Anderson-Palmer mean-field theory
for probabilistic modeling [[paper link](https://journals.aps.org/pre/pdf/10.1103/PhysRevE.64.056131)]

### Expectation propagation

- Expectation Propagation for Approximate Bayesian Inference(EP) [[paper link](https://tminka.github.io/papers/ep/minka-ep-uai.pdf)]
- Expectation Consistent Approximate Inference(EC) [[paper link](http://www.jmlr.org/papers/volume6/opper05a/opper05a.pdf)]
- Expectation Consistent Approximate Inference:
Generalizations and Convergence(GEC) [[paper link](https://arxiv.org/pdf/1602.07795.pdf)]
- Expectation Propagation for Exponential Families [[paper link](https://infoscience.epfl.ch/record/161464/files/epexpfam.pdf)]

### Density consistency

- Loop corrections in spin models through density consistency [[paper link](https://arxiv.org/abs/1810.10602)]


## Compressed Sensing

- Message-passing algorithms for compressed sensing(AMP) [[paper link](https://www.pnas.org/content/pnas/106/45/18914.full.pdf)]
- Vector Approximate Message Passing(VAMP) [[paper link](https://arxiv.org/abs/1610.03082)]
- Compressed Sensing by Shortest-Solution Guided Decimation(SSD) [[paper link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8262619)] *to read


## Physics and Learning

- Neural Network Renormalization Group [[paper link](https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.121.260601), [support material link](https://journals.aps.org/prl/supplemental/10.1103/PhysRevLett.121.260601/SM.pdf)]
- Uncover the Black Box of Machine Learning Applied to Quantum Problem by an
Introspective Learning Architecture [[paper link](https://arxiv.org/pdf/1901.11103.pdf)]


## Markov Chain Monte Carlo

### Hamiltonian Monte Carlo

- MCMC Using Hamiltonian Dynamics [[paper link](http://www.mcmchandbook.net/HandbookChapter5.pdf)] *to read
- A Conceptual Introduction to Hamiltonian Monte Carlo [[paper link](https://arxiv.org/abs/1701.02434)] *to read

## Network Science

- Estimating network structure from unreliable measurements [[paper link](https://journals.aps.org/pre/pdf/10.1103/PhysRevE.98.062321)]
- Spectral redemption in clustering sparse networks [[paper link](https://www.pnas.org/content/pnas/110/52/20935.full.pdf)]  
  <font color=gray>
  **Comments**: using non-backtracking operator in spectral method to do clustering in sparse networks, proving superiority of non-backtracking operator than other structural matrices like adjacency matrix. </font>
- Asymptotic analysis of the stochastic block model for modular networks and its
algorithmic applications [[paper link](https://arxiv.org/pdf/1109.3041.pdf)]  
  <font color=gray>
  **Comments**: using cavity method to construct belief propagation algorithm to solve stochastic block model, some equations needed to be revisited. </font>

## Self
### Gravity
- The effects of massive graviton on the equilibrium between the black hole and radiation gas in an isolated box [[paper link](https://www.sciencedirect.com/science/article/pii/S0370269317305750)]