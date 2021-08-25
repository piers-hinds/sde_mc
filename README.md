![Build Status](https://www.travis-ci.com/Piers14/sde_mc.svg?branch=main)
![Pytest Status](https://github.com/Piers14/sde_mc/workflows/pytesting/badge.svg)
<img src="https://coveralls.io/repos/github/Piers14/sde_mc/badge.svg?branch=main&kill_cache=1" />

# Monte Carlo simulation for SDEs with variance reduction methods
This library provides PyTorch implementation of:
1. Numerical integration of Stochastic Differntial Equations driven by Brownian motion and a Poisson process
2. Monte Carlo methods to compute the expectation of a functional of the terminal value
3. Variance reduction techniques using Deep Learning

(All with GPU support)

---
<p align="center">
  <img width="600" height="400" src="./viz/gbm_paths.gif">
</p>
