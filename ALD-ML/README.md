# ALD-ML 

## Contents

file                      | description
------------------------- | --------------------------------------
data                      | publicly available data, unfortunately not everything to run all analysis.
docs                      | Contains [setup-instructions](docs/setup.md)
roc_comparison            | source files for DeLong test (forked)
src                       | functionality for notebooks
[ALD_META_ML](ALD_META_ML.ipynb) | contains different comparisons initially performed to select a model type or feature selection. This is a script for explorational analysis, assesing the sensitivty to certain design choices (model, etc.)
[ALD_ML](ALD_ML.ipynb)    | Contains data pre-processing, Feature Selection, <br> Cross-Validation runs, Final model calculation and diverse <br> plots. Some functionality is loaded from [`src`](ALD-ML/src)
helper.py                 | Snapshot of some helper function used in [ALD_META_ML](ALD_META_ML.ipynb) (better see `src`-folder)