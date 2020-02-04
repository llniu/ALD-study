# Setup


## Setup on Windows using Anaconda

Download MiniConda and add channels:

conda config --add channels conda-forge

```
conda create --name ald_study python=3.7 scikit-learn scipy numpy jupyterlab py-xgboost pandas seaborn
conda install ipywidgets
jupyter labextension install @jupyterlab/toc # Table of Contents for IPYNBs
conda install nb_conda  # convienence 


conda activate ald_study
python -m ipykernel install --user --name ald_study --display-name "Python3.7 (ald_study)"
```

Install Jupyter Widgets:
```
conda install -c conda-forge nodejs
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```