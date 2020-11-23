# Dash App

A Dashboard App to interactively explore main results from the ALD study.

## Getting started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
The following python packages are required to run the code:
- pandas
- numpy
- dash
- seaborn
- plotly

### Insallation with conda

1. Open the console, ```cd``` to the directory where you want to install and run ```git clone https://github.com/llniu/ALD-study.git```. Alternatively, [download](https://github.com/llniu/ALD-study/archive/main.zip) the zip file and unzip it.
2. Change to the downloaded directory ```cd ALD-study/ALD-App``` 
3. Open the console and create a new conda environment: ```conda create --name env_ald_app python=3.7```
4. Activate the environment: ```source activate env_ald_app``` for Linux / Mac Os X or ```activate env_ald_app``` for Windows.
5. Install the required packages with pip (```pip install -r requirements.txt```)
6. Start the server with ```python ALD_app.py```
7. Copy the localhost address and open in a web browser
