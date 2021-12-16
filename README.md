# Framework for Learning Effective Dynamics for Molecular Systems (LED-Molecular)

Basic code of the paper: *Accelerated Simulations of Molecular Systems through Learning of their Effective Dynamics*, PR. Vlachas, J. Zavadlav, M. Praprotnik, P. Koumoutsakos, Journal of Chemical Theory and Computation, 2021.

The LED-Molecular employs the following neural architectures:
- (MDNs) Mixture density networks for stochastic dynamics
- (AEs) Autoencoders to capture the effective degrees of freedom
- (RNNs) Recurrent neural networks to capture nonlinear markovian dynamics on the latent space (LSTMs and GRUs, etc.)

## Requirements

The code requirements are:
- python (version 3.7)
- torch (version 1.8.1)
- matplotlib (version 3.4.1)
- scipy (version 1.6.2)
- psutil
- tqdm
- tabulate
- hickle
- scikit-learn
- GPUtil
- pyemma (version 2.5.9)
- rmsd

## Installation of Requirements with pyenv

The packages can be installed in a virtual environment with the following commands:
```
brew install pyenv
pyenv install 3.7.0
```
Then, navigate to the LED project folder, and execute the commands:
```
pyenv local 3.7.0
python -m venv venv-LED-Molecular
source ./venv-LED-Molecular/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
```
Installation should not take more than a few minutes.

## Installation of Requirements with virtualenv

The packages can be installed in a virtual environment with the following commands:
```
pip3 install virtualenv
virtualenv venv-LED-Molecular --python=python3.7
source ~/venv-LED-Molecular/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt
```
Installation should not take more than a few minutes.


## Previous Relevant Publications From Our Group

[1] *Learning the Effective Dynamics of Complex Multiscale Systems*, PR. Vlachas, G. Arampatzis, C. Uhler, P. Koumoutsakos, (submitted)

[2] *Backpropagation Algorithms and Reservoir Computing in Recurrent Neural Networks for the Forecasting of Complex Spatiotemporal Dynamics*, Pantelis R. Vlachas, Jaideep Pathak, Brian R. Hunt, Themistoklis P. Sapsis, Michelle Girvan, Edward Ott, Petros Koumoutsakos
Journal of Neural Networks, 2020.

[3] *Data-driven forecasting of high-dimensional chaotic systems with long short-term memory networks*, Pantelis R. Vlachas, Wonmin Byeon, Zhong Y. Wan, Themistoklis P. Sapsis and Petros Koumoutsakos
Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences 474 (2213), 2018.



