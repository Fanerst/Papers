# Paper Repository
Papers of my interest

## Learning

### Learning theory

- Geometrical and Statistical Properties of Systems of Linear Inequalities with Applications in Pattern Recognition [[paper](https://ieeexplore.ieee.org/document/4038449)]

### Generative models

#### VAEs
- Auto-Encoding Variational Bayes [[paper](https://arxiv.org/abs/1312.6114)]
- Hierarchical Representations with Poincaré Variational Auto-Encoders [[paper](https://arxiv.org/abs/1901.06033)]

#### Autoregressive models
- The Neural Autoregressive Distribution Estimator(NADE) [[paper](http://proceedings.mlr.press/v15/larochelle11a/larochelle11a.pdf)]
- MADE: Masked Autoencoder for Distribution Estimation(MADE) [[paper](https://arxiv.org/abs/1502.03509)]
- Fast Generation for Convolutional Autoregressive Models [[paper](https://arxiv.org/abs/1704.06001)]  
   <font color=gray>
  **Comments**: proposed a way to accelerate autoregreesive models like wave net and pixelcnn, need second thought about whether it can be used into VAN </font>
- Autoregressive Quantile Networks for Generative Modeling [[paper](https://arxiv.org/abs/1806.05575)]
- Transformation Autoregressive Networks [[paper](https://arxiv.org/abs/1801.09819)]

#### Flow-based networks

- Normalizing Flow [[paper](https://arxiv.org/abs/1505.05770)]
- Monge-Ampere Flow [[paper](https://arxiv.org/abs/1809.10188)]
- Neural Ordinary Differential Equations [[paper](https://arxiv.org/abs/1806.07366)]
- Density Estimation using Real NVP [[paper](https://arxiv.org/pdf/1605.08803.pdf)]
- Improved Variational Inference with Inverse Autoregressive Flow [[paper](https://arxiv.org/abs/1606.04934)]
- Glow: Generative Flow with Invertible 1×1 Convolutions [[paper](https://arxiv.org/abs/1807.03039)]

### Unclassified

- Variational discriminator bottleneck: improving imitation learning, inverse RL, and GANs by constraining information flow [[paper](https://arxiv.org/abs/1810.00821)]

### Nerual network and MCMC

- Variational Rejection Sampling [[paper](https://arxiv.org/abs/1804.01712)]
- Discriminator Rejection Sampling [[paper](https://arxiv.org/abs/1810.06758)]
- Generative Adversarial Training for Markov Chains [[paper](https://openreview.net/pdf?id=S1L-hCNtl)]
- A-NICE-MC: Adversarial Training for MCMC [[paper](https://arxiv.org/pdf/1706.07561.pdf)]  
  <font color=gray>
  **Comments**: improvement of above paper using HMC and NICE to improve MCMC, could it be used to physical system? 
  </font>

### Graph Neural Networks

#### Reviews
- Graph Neural Networks: A Review of Methods and Applications [[paper](https://arxiv.org/abs/1812.08434)]
- A Comprehensive Survey on Graph Neural Networks [[paper](https://arxiv.org/pdf/1901.00596v1.pdf)] *to read
- Relational inductive biases, deep learning, and graph networks [[paper](https://arxiv.org/abs/1806.01261)] *to read
- A Tutorial on Network Embeddings [[paper](https://arxiv.org/abs/1808.02590)]

#### Graph embeddings
- GraphGAN: Graph Representation Learning with Generative Adversarial Nets [[paper](https://arxiv.org/abs/1711.08267)]
- Learning Steady-States of Iterative Algorithms over Graphs [[paper](http://proceedings.mlr.press/v80/dai18a/dai18a.pdf)]

#### Graph Convolutional Networks
- Semi-Supervised Classification with Graph Convolutional Networks(GCN) [[paper](https://arxiv.org/abs/1609.02907)]
- Simplifying Graph Convolutional Networks(SGCN) [[paper](https://arxiv.org/pdf/1902.07153v1.pdf)]
- HOW POWERFUL ARE GRAPH NEURAL NETWORKS?(GRAPH ISOMORPHISM NETWORK) [[paper](https://arxiv.org/pdf/1810.00826.pdf)]
- GRAPH WAVELET NEURAL NETWORK [[paper](https://openreview.net/pdf?id=H1ewdiR5tQ)]
- Predict then Propagate: Graph Neural Networks meet Personalized PageRank [[paper](https://arxiv.org/abs/1810.05997)]
- Graph Attention Networks [[paper](https://arxiv.org/abs/1710.10903)]
- PAN: Path Integral Based Convolution for Deep Graph Neural Networks [[paper](https://arxiv.org/pdf/1904.10996.pdf)]

#### Relational networks

- A simple neural network module for relational reasoning(RN) [[paper](https://arxiv.org/abs/1706.01427)]
- Recurrent Relational Networks [[paper](https://arxiv.org/abs/1711.08028)]

#### GNN for combinatorial optimization problems
- Learning Combinatorial Optimization Algorithms over Graphs [[paper](https://arxiv.org/abs/1704.01665)]  
  <font color=gray>
  **Comments**: pretty interesting, but don't understand the point for now, probably because of lacking basic knownledge about reinforcement learning. </font>

- Coloring Big Graphs with AlphaGoZero [[paper](https://arxiv.org/abs/1902.10162)]

#### GNN adversial attack
- Adversarial Attacks on Neural Networks for Graph Data [[paper](https://arxiv.org/pdf/1805.07984.pdf)]
- Can Adversarial Network Attack be Defended? [[paper](https://arxiv.org/pdf/1903.05994.pdf)]

### Topological interpretation of ML
- Topology of Learning in Artificial Neural Networks [[paper](https://arxiv.org/abs/1902.08160)]  
  <font color=gray> 
  **Commments**: find a way to visualize evolution of weights through training, hinting that the learning ability comes from branching off of weights </font>

- A Geometric View of Optimal Transportation and Generative Model [[paper](https://arxiv.org/abs/1710.05488)]

### Optimal transport at ML
- Latent Space Optimal Transport for Generative Models [[paper](https://arxiv.org/abs/1809.05964)]
- Wasserstein Dependency Measure for Representation Learning [[paper](https://arxiv.org/abs/1903.11780)]
  
### Energy Based Model(EBM)
- A Tutorial on Energy-Based Learning [[paper](http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf)]
- Implicit Generation and Generalization in Energy-Based Models [[paper](https://arxiv.org/abs/1903.08689)]

### Gradient Estimators

- Neural Variational Inference and Learning in Belief Networks[[paper](https://arxiv.org/abs/1402.0030)]
- Reweighted Wake-Sleep [[paper](https://arxiv.org/abs/1406.2751)]
- Importance Weighted Autoencoders [[paper](https://arxiv.org/abs/1509.00519)]
- Variational Inference for Monte Carlo Objectives [[paper](https://arxiv.org/pdf/1602.06725.pdf)]

## Tensor Networks

- Matrix Product Operators for Sequence to Sequence Learning [[paper](https://arxiv.org/abs/1803.10908)]

## Thermodynamics and Statistical Mechanics

- Stochastic Thermodynamic Interpretation of Information Geometry [[paper](https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.121.030605), [support material](https://journals.aps.org/prl/supplemental/10.1103/PhysRevLett.121.030605/supplemenatry_information.pdf)]

### Replica symmetry breaking
- Infinite Number of Order Parameters for Spin-Glasses [[paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.43.1754)] *to read
- Order Parameter for Spin-Glasses [[paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.50.1946)] *to read
- The order parameter for spin glasses: a function on the interval 0-1 [[paper](http://iopscience.iop.org/article/10.1088/0305-4470/13/3/042/meta)] *to read

## Message Passing

### Belief propagation

- Understanding Belief Propagation and its Generalizations [[paper](http://www.merl.com/publications/TR2001-22)]
- Correctness of belief propagation in Gaussian graphical models of arbitrary topology [[paper](http://www.merl.com/publications/docs/TR99-33.pdf)]


## Approximate Inference

### ADATAP
- Tractable Approximations for Probabilistic Models: The Adaptive Thouless-Anderson-Palmer Mean Field Approach [[paper](https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.86.3695)]
- Adaptive and self-averaging Thouless-Anderson-Palmer mean-field theory
for probabilistic modeling [[paper](https://journals.aps.org/pre/pdf/10.1103/PhysRevE.64.056131)]

### Expectation propagation

- Expectation Propagation for Approximate Bayesian Inference(EP) [[paper](https://tminka.github.io/papers/ep/minka-ep-uai.pdf)]
- Expectation Consistent Approximate Inference(EC) [[paper](http://www.jmlr.org/papers/volume6/opper05a/opper05a.pdf)]
- Expectation Consistent Approximate Inference:
Generalizations and Convergence(GEC) [[paper](https://arxiv.org/pdf/1602.07795.pdf)]
- Expectation Propagation for Exponential Families [[paper](https://infoscience.epfl.ch/record/161464/files/epexpfam.pdf)]

### Density consistency

- Loop corrections in spin models through density consistency [[paper](https://arxiv.org/abs/1810.10602)]


## Compressed Sensing

- Message-passing algorithms for compressed sensing(AMP) [[paper](https://www.pnas.org/content/pnas/106/45/18914.full.pdf)]
- Vector Approximate Message Passing(VAMP) [[paper](https://arxiv.org/abs/1610.03082)]
- Compressed Sensing by Shortest-Solution Guided Decimation(SSD) [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8262619)] *to read


## Physics and Learning

- Neural Network Renormalization Group [[paper](https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.121.260601), [support material](https://journals.aps.org/prl/supplemental/10.1103/PhysRevLett.121.260601/SM.pdf)]
- Uncover the Black Box of Machine Learning Applied to Quantum Problem by an Introspective Learning Architecture [[paper](https://arxiv.org/pdf/1901.11103.pdf)]


## Markov Chain Monte Carlo

### Hamiltonian Monte Carlo

- MCMC Using Hamiltonian Dynamics [[paper](http://www.mcmchandbook.net/HandbookChapter5.pdf)] *to read
- A Conceptual Introduction to Hamiltonian Monte Carlo [[paper](https://arxiv.org/abs/1701.02434)] *to read

## Network Science

- Estimating network structure from unreliable measurements [[paper](https://journals.aps.org/pre/pdf/10.1103/PhysRevE.98.062321)]
- Spectral redemption in clustering sparse networks [[paper](https://www.pnas.org/content/pnas/110/52/20935.full.pdf)]  
  <font color=gray>
  **Comments**: using non-backtracking operator in spectral method to do clustering in sparse networks, proving superiority of non-backtracking operator than other structural matrices like adjacency matrix. </font>
- Asymptotic analysis of the stochastic block model for modular networks and its
algorithmic applications [[paper](https://arxiv.org/pdf/1109.3041.pdf)]  
  <font color=gray>
  **Comments**: using cavity method to construct belief propagation algorithm to solve stochastic block model, some equations needed to be revisited. </font>

## Self
### Gravity
- The effects of massive graviton on the equilibrium between the black hole and radiation gas in an isolated box [[paper](https://www.sciencedirect.com/science/article/pii/S0370269317305750)]


## To be read
### PRL
- Low-Scaling Algorithm for Nudged Elastic Band Calculations Using a Surrogate Machine Learning Model [[paper](https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.122.156001)]
- Dynamical Computation of the Density of States and Bayes Factors Using Nonequilibrium Importance Sampling [[paper](https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.122.150602)]
- Random Language Model [[paper](https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.122.128301)]

### Spin Glass and Statistical Physics
- Materials Physics and Spin Glasses [[paper](https://arxiv.org/pdf/1903.11414.pdf)]
- Compressed sensing reconstruction using Expectation Propagation [[paper](https://arxiv.org/pdf/1904.05777.pdf)]
- Kinked Entropy and Discontinuous Microcanonical Spontaneous Symmetry Breaking [[paper](https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.122.160601), [support material](https://journals.aps.org/prl/supplemental/10.1103/PhysRevLett.122.160601/SIforPottsV03.pdf)]

#### Cavity method and matrix product state
- A matrix product algorithm for stochastic dynamics on networks, applied to
non-equilibrium Glauber dynamics [[paper](https://arxiv.org/pdf/1508.03295.pdf)]
- The matrix product approximation for the dynamic cavity method [[paper](https://arxiv.org/pdf/1904.03312.pdf)]

### Physics and Learning
- Learning a Gauge Symmetry with Neural-Networks [[paper](https://arxiv.org/pdf/1904.07637.pdf)]

### Machine Learning
#### Theory
- Theoretical Impediments to Machine Learning With Seven Sparks from the Causal Revolution [[paper](https://arxiv.org/pdf/1801.04016.pdf)]
- A lattice-based approach
to the expressivity of deep ReLU neural networks [[paper](https://arxiv.org/pdf/1902.11294v1.pdf)]
- Deep learning as closure for irreversible processes: A data-driven
generalized Langevin equation [[paper](https://arxiv.org/pdf/1903.09562.pdf)]
- On the Impact of the Activation Function on Deep
Neural Networks Training [[paper](https://arxiv.org/pdf/1902.06853v1.pdf)]
- MEAN-FIELD ANALYSIS OF BATCH NORMALIZATION [[paper](https://arxiv.org/pdf/1903.02606.pdf)]
- A MEAN FIELD THEORY OF BATCH NORMALIZATION [[paper](https://arxiv.org/pdf/1902.08129.pdf)]

#### Generative Model
- Latent Translation:
Crossing Modalities by Bridging Generative Models [[paper](https://arxiv.org/pdf/1902.08261v1.pdf)]
- GANSYNTH:
ADVERSARIAL NEURAL AUDIO SYNTHESIS [[paper](https://arxiv.org/pdf/1902.08710.pdf)]
- Semi-supervised Learning with Deep Generative Models [[paper](https://arxiv.org/pdf/1406.5298.pdf)]
- Diagnosing and Enhancing VAE Models [[paper](https://arxiv.org/pdf/1903.05789v1.pdf)]

#### Structured Data
- The Emerging Field of Signal Processing on Graphs [[paper](https://arxiv.org/pdf/1211.0053v2.pdf)]
- Discriminative Embeddings of Latent Variable Models
for Structured Data [[paper](https://arxiv.org/pdf/1603.05629.pdf)]

#### Others
- Hyperbolic Neural Networks [[paper](https://arxiv.org/pdf/1805.09112v2.pdf)]
- Bayesian Neural Networks at Finite Temperature [[paper](https://arxiv.org/pdf/1904.04154.pdf)]