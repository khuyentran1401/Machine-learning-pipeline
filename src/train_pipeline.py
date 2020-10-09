import numpy as np
import pandas as pd


from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def pipeline(config):

	tfidf_word_char = FeatureUnion(
		[
			('tfidf_word', TfidfVectorizer(use_idf=True, analyzer='char')),
			('tfidf_char', TfidfVectorizer(use_idf=True, analyzer='word'))
		]
	)

	tfidf_pipeline = Pipeline(
		[
			('word_char', tfidf_word_char),
			('kbest', SelectKBest(chi2, k = 2000)),
			('classifier', eval(config.model)())
		]
	)

	return tfidf_pipeline



