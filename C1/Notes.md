# Notes of chapter 1

Q: What is generative modeling?
A: Train, new data, similar...

Concept of latent space:
Point in the latent space is a representation of some high dimensional data.

## Core probability theory

### Sample space
- The complete set of all values an observation x can take.

### Probability density function
- It is a function p(x) that maps a point x in the sample space to a number in the range [0, 1].
- The integral of p(x) over the sample space is 1.

### Parametric modeling
- This is a technique that we can use to structure our approach to finding a suitable <br>
- probability density function p_model(x) for a given dataset.

### Likelihood
- log-likehood
- Goal: Find the optimal value of the parameter set that maximizes the likelihood of the observed
- data.

### Maximum likelihood estimation
- Neural networks typically minimize a loss function.
- It equals to finding the set of parameters that minimizes the negative log-likelihood.
- Problem: p_theta(x) is intractable.

## Generative model taxonomy
Based on how to model the density function p(x).