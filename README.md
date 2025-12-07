TF-CLIMB: Training-Free CLIP Few-Shot under Class Imbalance
===========================================================

This is the code for the course project
“Training-Free CLIP Few-Shot Adaptation under Class-Imbalanced Settings”.
It evaluates several training-free CLIP baselines (zero-shot, prototypes,
fused, TF-CLIMB) on CIFAR-100, EuroSAT and Oxford-IIIT Pets with imbalanced
few-shot splits.

Main idea of TF-CLIMB:
- build per-class prototypes in CLIP image feature space,
- fuse zero-shot text logits and prototype logits,
- add a logit bias term based on empirical class frequencies.


Files
-----

config.py
    Global config: paths, dataset list, shot numbers, imbalance ratios,
    TF-CLIMB hyperparameters, device.

data_utils.py
    Load datasets, run CLIP ViT-B/16, cache train/test features.

sampling.py
    Helper for building class indices and sampling imbalanced support
    counts/indices for given K and rho.

make_imbalanced_supports.py
    Script to pre-generate imbalanced few-shot splits and save them under
    splits/{dataset}/geom_K{K}_rho{rho}_seed{seed}.pt.

tfclimb.py
    Implementation of zero-shot, prototype, fused and TF-CLIMB predictors:
    text features, class prototypes, class priors and logit adjustment.

metrics.py
    Overall accuracy, macro accuracy, head- and tail-class macro accuracy.

run_experiment.py
    Main script. Loops over datasets, shot numbers and imbalance ratios,
    evaluates all methods and writes results/{dataset}_results.csv.

vis.py
    Simple plotting utilities for macro / head / tail metrics vs. rho
    using the saved CSV files.


Requirements
------------

- Python 3.8+
- PyTorch and torchvision
- numpy, pandas, matplotlib
- CLIP library (e.g. `pip install git+https://github.com/openai/CLIP.git`)


Quick Start
-----------

1. (Optional) Pre-generate imbalanced support splits:

       python make_imbalanced_supports.py

2. Run experiments for all datasets / K / rho specified in config.py:

       python run_experiment.py

   This will produce per-dataset result files:

       results/{dataset}_results.csv

3. Plot results for a given dataset and K (example: CIFAR-100, 4-shot):

       python vis.py --dataset cifar100 --K 4

To change datasets, shot numbers, imbalance ratios or TF-CLIMB
hyperparameters, edit config.py and re-run the scripts.
