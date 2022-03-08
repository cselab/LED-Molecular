# Framework for Learning Effective Dynamics for Molecular Systems (LED-Molecular)

Basic code implementation of the paper: *Accelerated Simulations of Molecular Systems through Learning of their Effective Dynamics*, PR. Vlachas, J. Zavadlav, M. Praprotnik, P. Koumoutsakos, J. Chem. Theory Comput. 2022, 18, 1, 538–549 *https://doi.org/10.1021/acs.jctc.1c00809*

The LED-Molecular employs the following neural architectures:
- (AEs) Autoencoders to capture the effective degrees of freedom
- (MDNs) Mixture density networks for stochastic dynamics and probabilistic decoding
- (RNNs) Recurrent neural networks to capture nonlinear markovian dynamics on the latent space (LSTMs and GRUs, etc.)

## Requirements

Code requirements are provided in the requirements.txt file.
The code has been compiled with python version 3.7.

## Relevant Publications

[1] *Accelerated Simulations of Molecular Systems through Learning of Effective Dynamics*, PR. Vlachas, J. Zavadlav, M. Praprotnik, P. Koumoutsakos
Journal of Chemical Theory and Computation 18 (1), 538-549, 2022

[2] *Multiscale Simulations of Complex Systems by Learning their Effective Dynamics*, PR. Vlachas, G. Arampatzis, C. Uhler, P. Koumoutsakos
Nature Machine Intelligence, 2022.

[3] *Backpropagation Algorithms and Reservoir Computing in Recurrent Neural Networks for the Forecasting of Complex Spatiotemporal Dynamics*, Pantelis R. Vlachas, Jaideep Pathak, Brian R. Hunt, Themistoklis P. Sapsis, Michelle Girvan, Edward Ott, Petros Koumoutsakos
Journal of Neural Networks, 2020.

[4] *Data-driven forecasting of high-dimensional chaotic systems with long short-term memory networks*, Pantelis R. Vlachas, Wonmin Byeon, Zhong Y. Wan, Themistoklis P. Sapsis and Petros Koumoutsakos
Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences 474 (2213), 2018.

