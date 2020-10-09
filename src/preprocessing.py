import numpy as np 
import pandas as pd 

import nltk
nltk.download('all')
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import hydra
from hydra import utils

import re
import pickle



def read_file(file):
    return open(utils.to_absolute_path(file)).read().split("\n")

def save_file(file, name):
    with open(utils.to_absolute_path(name), "wb") as fp:
        pickle.dump(file, fp)


class RegexReplacer(object):
    def __init__(self):

        self.patterns = [
                        (r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '<url>'),
                        (r'@\w+', '<user>'),
                        (r'&\w+', '') #Replace "&..." with ''
                        ]
        self.patterns = [(re.compile(regrex),repl) for (regrex, repl) in
                        self.patterns]
    
    #Replace the words that match the patterns with replacement words
    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            s = re.sub(pattern, repl, s)
        return s

class ProcessText:

    def __init__(self):
        self.tknz = TweetTokenizer()
        self.replacer = RegexReplacer()
        self.stopwords = set(stopwords.words('english'))
        self.punc = string.punctuation
        self.lemmatizer = WordNetLemmatizer()


    def normalize(self, doc):
        
        for i in range(len(doc)):
            
            #Tokenize with replacement
            doc[i] = self.tknz.tokenize(self.replacer.replace(doc[i]))
            
            #Filter stopwords, punctuations, and lowercase
            doc[i] = [w.lower() for w in doc[i] if w not in self.punc and w not in self.stopwords]
        
            #Stem words
                        
            doc[i] = [self.lemmatizer.lemmatize(w, pos='v') for w in doc[i]]
            
            #concat
            doc[i] = ' '.join(w for w in doc[i])
            
        return doc

@hydra.main(config_path='../configs/data_preprocess.yaml')
def main(config):
    train_text = read_file(config.data.text.train)
    val_text = read_file(config.data.text.val)
    test_text = read_file(config.data.text.test)
    train_label = read_file(config.data.label.train)
    val_label = read_file(config.data.label.val)

    process_text = ProcessText()

    train_text = process_text.normalize(train_text)
    val_text = process_text.normalize(val_text)
    test_text = process_text.normalize(test_text)
    train_label = process_text.normalize(train_label)
    val_label = process_text.normalize(val_label)


    save_file(train_text, config.processed_data.text.train)
    save_file(val_text, config.processed_data.text.val)
    save_file(test_text, config.processed_data.text.test)
    save_file(train_label, config.processed_data.label.train)
    save_file(val_label, config.processed_data.label.val)

if __name__== '__main__':
    main()
    



















