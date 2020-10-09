# Machine learning pipeline
This repo provides an example of how to incorporate MLflow and Hydra in your machine learning project for reproducibility. This example is pulled from my project on predicting aggressive tweets. Find the article for this repo [here](https://towardsdatascience.com/achieve-reproducibility-in-machine-learning-with-these-two-tools-7bb20609cbb8?source=friends_link&sk=8e1e186294f46df97e0325ce9790f2d7)

# Hydra
With [Hydra](https://hydra.cc/), you can compose your configuration dynamically, enabling you to easily get the perfect configuration for each run. To install Hydra, simply run
```
pip install hydra-core --upgrade
```
# MLflow
[MLflow](https://mlflow.org/) is an open-source platform to manage the ML lifecycle, including experimentation, reproducibility, and deployment. Install MLflow with 
```
pip install mlflow
```

# Structure's explanation
* **src**: file for source code
* **mlruns**: file for mlflow runs
* **configs**: to keep config files
* **outputs**: results from the runs of Hydra. Each time you run your function nested inside Hydra's decoration, the output will be saved here. If you want to change the directory in mlflow folder, use
```python
import mlflow
import hydra
from hydra import utils

mlflow.set_tracking_uri('file://' + utils.get_original_cwd() + '/mlruns')
```
* preprocessing.py: file for preprocessing
* train_pipeline.py: training's pipeline
* train.py: file for training and saving model
* predict.py: file for prediction and loading model

# How to run this file
To run the configs and see how these experiments are displayed on MLflow's server, clone this repo and run
```
python src/train.py
```
Once the run is completed, you can access to MLflow's server with
```
mlflow ui
```
Access http://localhost:5000/ from the same directory that you run the file, you should be able to see your experiment like this
![image](https://github.com/khuyentran1401/Machine-learning-pipeline/blob/master/Screenshot%20from%202020-05-03%2016-41-21.png)
