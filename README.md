# Machine learning pipeline
This repo provides an example of how to incorporate MLflow and Hydra in your machine learning project for reproducibility. This example is pulled from my project on predicting aggressive tweets.

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
* mlruns: file for mlflow runs
* experiment: to keep config files
* outputs: results from the runs of Hydra. Each time you run your function nested inside Hydra's decoration, the output will be saved here. If you want to change the directory in mlflow folder, use
```python
import mlflow
import hydra
from hydra import utils

mlflow.set_tracking_uri('file://' + utils.get_original_cwd() + '/mlruns')
```
* preprocessing.py: file for preprocessing
* train_pipeline.py: training's pipeline
* train.py: file for training
* predict.py: file for prediction

