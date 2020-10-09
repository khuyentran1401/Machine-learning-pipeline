import csv
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn import metrics, preprocessing
import numpy as np

from sklearn import svm, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import mlflow
import mlflow.sklearn

import pickle
import warnings
import hydra
from hydra import utils

from train_pipeline import pipeline

from sklearn.model_selection import GridSearchCV
import os

def read_file(name):

	with open(utils.to_absolute_path(name), 'rb') as fp:
		f = pickle.load(fp)

	return f

@hydra.main(config_path='configs',
			config_name='hyperparameters')
def main(config):
	warnings.filterwarnings("ignore")
	np.random.seed(40)

	X_train = read_file(config.processed_data.text.train)
	X_test = read_file(config.processed_data.text.val)
	y_train = read_file(config.processed_data.label.train)
	y_test = read_file(config.processed_data.label.val)

	mlflow.set_tracking_uri('file://' + utils.get_original_cwd() + '/mlruns')
	mlflow.set_experiment(config.mlflow.experiment_name)

	tfidf_pipeline = pipeline(config)

	with mlflow.start_run():

		param_grid = dict(config.hyperparameters)

		for key, value in param_grid.items():
			if isinstance(value, str):
				param_grid[key] = eval(param_grid[key])

		param_grid = {ele: (list(param_grid[ele])) for ele in param_grid}			
		
		grid_search = GridSearchCV(tfidf_pipeline, param_grid=param_grid, scoring=config.GridSearchCV.scoring, cv=config.GridSearchCV.cv, n_jobs=config.GridSearchCV.n_jobs)

		grid_search.fit(X_train, y_train)

		labels_pred = grid_search.predict(X_test)

		print(confusion_matrix(y_test, labels_pred))
		print(metrics.f1_score(y_test, labels_pred, average='macro'))

		mlflow.log_params(grid_search.best_params_)
		mlflow.log_artifacts(utils.to_absolute_path("configs"))
		mlflow.log_metric('f1_macro', eval(config.metrics.score)(y_test, labels_pred, average = config.metrics.average))

		mlflow.sklearn.log_model(grid_search, 'kbest')
		#mlflow.sklearn.save_model(grid_search, utils.to_absolute_path('models/kbest-{}-{}'.format(grid_search.best_params_['kbest__k'], 
		#	grid_search.best_params_['svr__C'])))
		

if __name__== "__main__":
	main()

