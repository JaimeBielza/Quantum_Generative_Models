# Quantum Generative Modeling with Parameterized Circuits

## Overview

This project explores the use of **parameterized quantum circuits (PQCs)** for **generative modeling**.

The goal is to train a quantum circuit to learn a target probability distribution \( p(x) \) and approximate it with a learned distribution \( q_theta(x) \), where \( theta \) are the circuit parameters.

This work is inspired by:

> Benedetti et al., *A generative modeling approach for benchmarking and training shallow quantum circuits*, Science Advances (2019)

 

## Motivation

Generative AI models are powerful but come with significant challenges:

- Large datasets required for training  
- High computational cost  
- Long training times  
- Increasing energy consumption  

This project investigates whether **quantum computing** could contribute to generative modeling in the future.

 

## Approach

This is a **hybrid quantum-classical algorithm**:

### Quantum component
- A parameterized quantum circuit (ansatz)
- Generates samples via measurement
- Defines the distribution \( q_\theta(x) \)

### Classical component
- Optimizes circuit parameters \( \theta \)
- Minimizes KL divergence using:
  - Bayesian Optimization (BO)
  - Particle Swarm Optimization (PSO)

 

## Target Distribution

Instead of using complex datasets (e.g., images), we use a **discretized Gaussian distribution**:

- Defined over \( 2^n \) states
- Normalized probability distribution

This allows controlled experimentation and clear evaluation.

 

## Circuit Design

Each circuit consists of:

- **Single-qubit rotations**:
  - RZ → RX → RZ

- **Entangling layers**:
  - Star topology: (0,1), (0,2), (0,3)
  - Circular topology: fully connected pairs

- Configurable number of layers:
  - 1 layer
  - 2 layers

 

## Experiments

We evaluate different configurations:

### Optimizers
- Bayesian Optimization (BO)
- Particle Swarm Optimization (PSO)

### Entanglement structures
- Star
- Circular

### Circuit depth
- 1 layer
- 2 layers

### Simulation modes
- Ideal simulation
- Noisy simulation (with depolarizing + readout errors)

 

## Noise Model

The noisy simulation includes:

- Single-qubit depolarizing errors  
- Two-qubit depolarizing errors  
- Readout errors  

This approximates realistic quantum hardware conditions.

 

## Results

The model successfully learns an approximation of the target distribution:

- \( p(x) \): target distribution  
- \( q(x) \): learned quantum distribution  

Results show:

- Reasonable convergence in ideal simulations  
- Degradation under noise (expected)  
- Sensitivity to circuit depth and entanglement  



## Key Insights

- Hybrid quantum-classical models are viable for generative tasks  
- Optimization plays a critical role in performance  
- Noise significantly impacts learning quality  
- Circuit design (depth + entanglement) is crucial  



## Potential Applications

If scaled to fault-tolerant quantum computers, this approach could enable:

- Synthetic data generation  
- Molecular and physical simulations  
- Financial modeling  
- Probabilistic modeling  
- Complex system simulation  


## Future Work

- Run experiments on **real IBM Quantum hardware**  
- Explore deeper circuits and larger qubit systems  
- Investigate quantum-native optimizers  
- Improve noise robustness  
- Extend to more complex datasets (e.g., images)


## Installation

Clone the repository:

