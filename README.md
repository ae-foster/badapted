# Baselines for DAD
This repo is forked from the main [Badapted](https://github.com/drbenvincent/badapted) repo, and provides the implementation of the baselines used for the
hyperbolic temportal discounting experiment in the paper [Deep Adaptive Design: Amortizing Sequential Bayesian Experimental Design](https://arxiv.org/abs/2103.02438).

## Environment setup and computing infrastructure
We ran this code on a CPU-only machine with 40 CPU cores. For installation, please follow the instructions on the original repo.

## Badapted and Frye baselines
Use
```
$ python3 badapted_hyperbolic.py
```
to generate 10000 rollouts using the Badapted method.
Use
```
$ python3 frye_hyperbolic.py
```
to generate 10000 rollouts using the Frye method.
