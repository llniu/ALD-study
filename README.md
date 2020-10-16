
# Alcohol Related Liver Disease (ALD) project
- Link to repository: [github.com/llniu/ALD-study](https://github.com/llniu/ALD-study)
- summary of scripts used for diverse analysis in the project
- 459 ALD patients (to varying degrees) and 137 age-, bmi-, and gender matched healthy controls
- Datasets generated and used in this study include proteomics, clinical data, and liver histology characterizations.

## Contents

file                      | description
------------------------- | --------------------------------------
[ALD_ML](ALD-ML/ALD_ML.ipynb)    | Contains data pre-processing, Feature Selection, <br> Cross-Validation runs, Final model calculation and diverse <br> plots. Some functionality is loaded from [`src`](ALD-ML/src)
[ALD_ML_STATA](ALD-ML/ALD_ML_STATA.ipynb) | References to STATA `.do` files in main folder. Done in hospital on follow-up data.
[ALD_PA](ALD-PA/ALD_PA.ipynb)    | Contains Proteomics and Clinical data pre-processing, ANCOVA, Partial correlation, Integration between the liver- and plasma proteomes, and plots.
[ALD_App](ALD-App/ALD_app.py)     | Contains interactive data visualization Dash App. 

## BioRxiv

> Niu, L., Thiele, M., Geyer, Philipp E., Rasmussen, D. N., Webel, H. E., Santos, A.,
> Gupta, R., Meier, F., Strauss, M., Kjaergaard, M., Lindvig, K., Jocobson, S.,
> Rasmussen, S., Hansen, T., Krag, A., & Mann, M. (2020). A paired liver biopsy and plasma
> proteomics study reveals circulating biomarkers for alcohol-related liver disease.

## Disclaimer

All datasets cannot be made public available yet until approval from the Danish Data Protection Agency is in place.

## Summary of the study
![alt text](figures/Study%20overview.jpg)
