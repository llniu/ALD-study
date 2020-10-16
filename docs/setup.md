# Setup

## Setup using Anaconda

These instructions are tested on Windows and Ubuntu.


## Get Conda package manager

Download MiniConda and add channels:

```
conda config --add channels conda-forge
```

### Install standalone in conda environment

```
conda create --name ald_study python=3.7 scikit-learn scipy numpy jupyterlab py-xgboost pandas seaborn ipywidgets
#conda create --name ald_study --file environment.yml

conda install -c conda-forge nodejs
jupyter labextension install @jupjuyterlab/toc # Table of Contents for IPYNBs
conda install nb_conda  # convienence 

jupyter labextension install @jupyter-widgets/jupyterlab-manager

```



### Install in conda environment without jupyter (notebook/lab)

Advanced, only for reference.

> You might have to check that the `ipykernel` version of your jupyter environment 
> and the newly install ald_study environment are compatible.

```
# in base-environment
conda activate base
conda install -c conda-forge nodejs
conda install jupyterlab
# conda update ipykernel # this might break other envs!
jupyter labextension install @jupjuyterlab/toc # Table of Contents for IPYNBs
conda install nb_conda  # convienence 

jupyter labextension install @jupyter-widgets/jupyterlab-manager

# install ald_study environment
conda create --name ald_study python=3.7 scikit-learn scipy numpy py-xgboost pandas seaborn ipykernel ipywidgets

# in ald_study environment
conda activate ald_study
python -m ipykernel install --user --name ald_study --display-name "Python3.7 (ald_study)"
```
