# Setup


## Setup on Windows using Anaconda

Download MiniConda and add channels:

conda config --add channels conda-forge

```
conda create --name ald_study python=3.7 scikit-learn scipy numpy jupyterlab py-xgboost pandas seaborn ipywidgets

# in base-environment
conda activate base
jupyter labextension install @jupjuyterlab/toc # Table of Contents for IPYNBs
conda install nb_conda  # convienence 
conda install -c conda-forge nodejs
jupyter labextension install @jupyter-widgets/jupyterlab-manager

# in ald_study environment
conda activate ald_study
python -m ipykernel install --user --name ald_study --display-name "Python3.7 (ald_study)"
```
