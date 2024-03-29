# Time Series Forecasting using Deep Neural Networks
Ensemble of autoregressive deep neural networks for time series forecasting that explicitly handles skewness, heavy-taildness and heteroscedasticity. Model selection and hyperparameter tuning is performed automatically using Bayesian Optimisation. 

Slides with details on how the model works [here](https://drive.google.com/file/d/1e2mFeVyjAcSaNFuNdTb21jyRTyYPqjQY/view?usp=sharing).

## Requirements
The current implementation makes use of the following Python packages (more recent versions are compatible, however older version may not be):
```
torch '1.7.0'
numpy '1.18.1'
pandas '1.1.1'
matplotlib '2.2.5'
sklearn '0.22.2'
scipy '1.5.2'
AutoDF.py 
```

## Usage
```Nile-basic.py``` and ```Nile-custom.py``` give some example usages of the model on the Nile river dataset. A copy of the data in csv format ```Nile.csv``` is also provided. 
