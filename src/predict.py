import mlflow
from mlflow import sklearn

import hydra
from hydra import utils

import warnings
import pickle
import os

import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn import metrics, preprocessing

def read_file(name):

    with open(utils.to_absolute_path(name), 'rb') as fp:
    	f = pickle.load(fp)

    return f

@hydra.main(config_path='../configs/hyperparameters.yaml')
def predict(config):
	warnings.filterwarnings("ignore")
	np.random.seed(40)

	X_test = read_file(config.processed_data.text.val)
	y_test = read_file(config.processed_data.label.val)

	model = mlflow.sklearn.load_model(utils.to_absolute_path("mlruns/1/b1da1795d808496f8231f7c4fcc3697f/artifacts/kbest"))

	labels_pred = model.predict(X_test)

	print(confusion_matrix(y_test, labels_pred))
	print(metrics.f1_score(y_test, labels_pred, average='weighted'))

if __name__ == '__main__':
	predict()
