![Build Status](https://www.travis-ci.com/Piers14/sde_mc.svg?branch=main)
![Pytest Status](https://github.com/Piers14/sde_mc/workflows/pytesting/badge.svg)
<img src="https://coveralls.io/repos/github/Piers14/sde_mc/badge.svg?branch=main&kill_cache=1" />

# Monte Carlo simulation for SDEs with variance reduction methods
This library provides PyTorch implementation (with GPU support) of:
1. Numerical integration of Stochastic Differential Equations driven by Brownian motion and a Poisson random measure
2. Monte Carlo methods to compute the expectation of a functional of the terminal value
3. Variance reduction techniques using Deep Learning

---
<p align="center">
  <img width="600" height="400" src="./viz/gbm_paths.gif">
</p>
