Information locality
--

Code for calculating predictive information from entropy rate curves from simulations and empirical data, supporting the paper Futrell \& Hahn (2024).

`infoloc.py` contains the core logic for calculating predictive information. 

The file `um_experiments.py` runs the morphology experiments. Paths to UD corpora need to be set in the function `run.py`.

The file `featurecorr.py` runs the semantic feature experiments. Paths to UD corpora need to be set in the file.

The file `experiments.py` runs remaining experiments from the main text.

System requirements: Python 3.11, `numpy`, `pandas`, `scipy`, `matplotlib`, `tqdm`, `plotnine`. 