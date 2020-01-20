This project is about implementing several multi-armed contextual bandits which most of them correspond to the methods mentioned in the 
[Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks for Thompson Sampling](https://arxiv.org/abs/1802.09127) paper.

## Getting Started ##  

1. The notebook "all methods explanation" is for the purpose of learning and testing all the algorithms separately with the data provided in the "datasets" folder. Each model is 
   explained and the references are provided. The "results" folder stores the tensorflow graph summary for tensorboard visualization.

2. The notebook "test" is for testing at the same time all the algorithms in "scripts/algorithms" which are dependent on functions in "scripts/core" and "scripts/data". 
   The "results" folder stores the tensorflow graph summary for tensorboard visualization.
   Note that these are the same algorithms as in the notebook "all methods explanation". They are named as following:


> * bootstrapped_bnn_sampling: bootstrapped sampling using q neural networks whose architecture is defined in "neural_bandit_model.py"
> * linear_thompson_sampling: Thompson Sampling based on bayesian linear regression
> * linucb: Upper Confidence Bound method based on linear ridge regression 
> * posterior_bnn_sampling: Thompson Sampling based on (one of the following posterior approximation methods):
> > * multitask_gp: multi-task gaussian process. The class inherits from the "BayesianNN" class in "bayesian_nn". 
> > * variational_neural_bandit_model: stochastic variational inference. The class inherits from the "BayesianNN" class in "bayesian_nn".

## Other Notebooks ##

The "notebooks" folder contains other contextual bandits models:

> * contextual bandits: a test of "linucb" model on some artifitial data.
> * contextual bandits RL: contextual bandit solution using reinforcment learning. 
> * contextual bandits logistic regression: tests some of the methods in [Adapting multi-armed bandits policies to contextual bandits scenarios](https://arxiv.org/abs/1811.04383).
    The methods are all based on running independent logistic regression for each arm. The dataset used is "Bibtex_data.txt" in the "datasets" folder.
