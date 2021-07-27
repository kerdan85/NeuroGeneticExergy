# NeurogeneticExergy
ANN/GA for neural network structure optimisation

Cite the code: [![DOI](https://zenodo.org/badge/264465907.svg)](https://zenodo.org/badge/latestdoi/264465907)

**1. Overral Framework**
The proposed exergy-based neuro-genetic framework can be summarised in the following steps:

  1.1. Building energy/exergy baseline simulation model development. In this step, a detailed building energy model is thermodynamically evaluated using ExRET-Opt [1], a recently developed building simulation tool based on the EnergyPlus engine [2] that further integrates exergy and exergoeconomics analyses.

  1.2. Parametric study design and simulation. Considering different building energy technologies and measures, a near-random sample is generated using Latin Hypercube Sampling (LHS) method and simulated using ExRET-Opt. The aim at this stage is to generate a comprehensive database that would be used to train the ANN.

  1.3. ANN training and surrogate model development. With the database in place, the neuro-genetic tool based on Python-based machine learning libraries is adopted to train the ANNs using backpropagation. Although build upon a limited amount of simulated results from dynamic thermal simulations (step 2), the framework allows the ANN to learn the complex thermodynamics interactions between the building physics and the building energy systems.

  1.4. ANN structure optimisation. At this stage, genetic optimisation is applied to find optimal ANN hyperparameters and structures preserving the essential behaviour of more complex modelling configurations. The optimisation process can define as a cost function different ANN performance metrics such as RMSE or MSE.

  1.5. Optimised ANN prediction performance. After the optimal ANN structure is identified by the tool, prediction performance against the physics-based energy modelling target outputs is conducted.

![image](https://user-images.githubusercontent.com/7305986/120885198-939f4680-c5df-11eb-9033-abe033bf538a.png)

**2.  ANN definition for surrogate model**
The main advantage of using ANNs over other machine learning techniques, is the ability to learn more efficiently complex interrelated parameters by ignoring non-significant variables and data. This has the potential to develop more efficient models that could be easier to optimise. The basic ANN structure is made of an input layer, hidden layer(s), and an output layer.

The structure of the ANN has been set as a multilayer perceptron (MLP), which is a feed-forward neural network with one or more layers between the input and output layers and where backpropagation is used as a supervision learning method. When designing an MLP, the main issue is to determine the number of layers and neurons. The neurons are made of three characteristics: 1) one or more weighted input connections, a transfer function, and one or more output connections.

**3. ANN structure optimisation**
Although the use of ANN has provided with several benefits in different research areas, the ANN structure and hyperparameters definition are problem-dependant. ANN hyperparameters can be catalogued as external parameters set by the network user. The selection of these will have important implications in the ANN performance, where the optimal combination of these is not a trivial task.

Normally, the design process of ANN structures is based on a trial and error approach, mostly driven by the experience of the user. To automate the process, it has been suggested to include a genetic optimisation procedure, allowing the exploration of a multidimensional space of possible structures. In this tool, the design of the MLP ANN networks is done by integrating TensorFlow and Python-based Keras neural network libraries into the presented tool. The selection of Keras allows the tool for rapid development capable of efficient computational times and parallelism. By default, tofind optimal ANN structures, the recombination of the following hyperparameters has been considered:

Training/Testing Data Share: [95/5, 90/10, 85/15, 80/20, 75/25, 70/30]

•Number of Layers: [1 ,2, 3, 4]

•Min/Max Number of Neurons per Layer: [1:20]

•Batch List: [10, 25, 50, 100, 200]

•Optimisers: [adam, adagrad, rmsprop, sgd]

•Kernel/weight Initialiser: [uniform, normal]

•Epochs: [50, 100, 150, 200]

•Dropout Rate: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

•Activation Type: [relu, elu, tanh, sigmoid]

However, this can be modified by the user. 

A Non-dominated Sorting Genetic Algorithm-II (NSGA-II) is used to optimise the ANN structure. The main advantaged of using NSGA-II is to reduce the probability of local minimum optimisation of the ANN structure. The complete neuro-genetic optimisation framework is illustrated below. It shows a snapshot of the ANN structure optimisation process at each iteration. In each generation the networks with the highest score are selected to recombine and by using the classic GA operators of crossover and mutation, a new generation of improved networks are created.

![image](https://user-images.githubusercontent.com/7305986/120885318-38218880-c5e0-11eb-809a-6e3f396c46f1.png)

The latest release has been archived in Zenodo (doi: https://doi.org/10.5281/zenodo.3893600). For simplification, the table below shows the pseudocode of the complete ANN-GA optimisation process described in this section.

![image](https://user-images.githubusercontent.com/7305986/120885350-56878400-c5e0-11eb-924a-6cb0dee9f952.png)

For more information and applications please read the following article:

García Kerdan, I. Morillón Gálvez, D. (2020). **Artificial neural network structure optimisation for accurately prediction of exergy, comfort and life cycle cost performance of a low energy building**. Applied Energy, 280, 115862. https://doi.org/10.1016/j.apenergy.2020.115862.
(https://www.sciencedirect.com/science/article/pii/S0306261920313350)
