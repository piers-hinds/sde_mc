![Build Status](https://www.travis-ci.com/Piers14/sde_mc.svg?branch=main)
![Pytest Status](https://github.com/Piers14/sde_mc/workflows/pytesting/badge.svg)
<img src="https://coveralls.io/repos/github/Piers14/sde_mc/badge.svg?branch=main&kill_cache=1" />

# Monte Carlo simulation for SDEs with variance reduction methods
This library provides PyTorch implementation (with GPU support) of:
1. Numerical integration of Stochastic Differential Equations of the form:


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://latex.codecogs.com/png.image?\dpi{110}&space;\bg_black&space;\inline&space;\hspace{20mm}&space;\text{d}X_t&space;=&space;b(t,&space;X_t)\text{d}t&space;&plus;&space;\sigma(t,&space;X_t)\text{d}W_t&space;&plus;&space;\int_{\mathbb{R}^q}F(t,&space;X_{t-})\xi&space;N(\text{d}t,&space;\text{d}\xi)." title="\bg_black \inline \hspace{20mm} \text{d}X_t = b(t, X_t)\text{d}t + \sigma(t, X_t)\text{d}W_t + \int_{\mathbb{R}^q}F(t, X_{t-})\xi N(\text{d}t, \text{d}\xi)." />

2. Monte Carlo methods to compute the expectation of a functional of the terminal value
3. Variance reduction techniques using Deep Learning
