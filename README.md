Information locality
--

Code for calculating predictive information from entropy rate curves from simulations and empirical data, supporting the paper Futrell \& Hahn (2024).

`infoloc.py` contains the core logic for calculating predictive information.

`experiments.py` runs most of the experiments reported in the paper.

The file `um_experiments.py` runs the morphology experiments. The file `featurecorr.py` runs the semantic feature experiments. The file `experiments.py` runs remaining experiments from the main text.

System requirements: `python` 3.11, `numpy`, `pandas`, `scipy`, `tqdm`, `plotnine`. 