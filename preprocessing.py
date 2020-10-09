import numpy as np 
import pandas as pd 

import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import hydra

#nltk.download('all')

import re
import pickle



def read_file(file):
    return open(utils.to_absolute_path(file)).read().split("\n")

def save_file(file, name):
    with open(utils.to_absolute_path(name), "wb") as fp:
        pickle.dump(file, fp)

replacement_patterns = [
    #match url (i.e: https://t.co/5tF5G9VKtq)
    (r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '<url>'),

    #match user (i.e: @cerpintor )
    (r'@\w+', '<user>'),

    #match hashtag (i.e: #WomensMarchOnWashington)
    #(r'#\w+', '<hashtag>'),

    #Replace "&..." with ''
    (r'&\w+', '')
]

class RegexReplacer(object):
    def __init__(self, patterns = replacement_patterns):
        self.patterns = [(re.compile(regrex),repl) for (regrex, repl) in
                        patterns]
    
    #Replace the words that match the patterns with replacement words
    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            s = re.sub(pattern, repl, s)
        return s

tknz = TweetTokenizer()
replacer = RegexReplacer()
stopwords = set(stopwords.words('english'))
punc = string.punctuation
from hydra import utils


def normalize(doc):
    
    for i in range(len(doc)):
        
        #Tokenize with replacement
        doc[i] = tknz.tokenize(replacer.replace(doc[i]))
        
        #Filter stopwords, punctuations, and lowercase
        doc[i] = [w.lower() for w in doc[i] if w not in punc and w not in stopwords]
    
        #Stem words
        
        lemmatizer = WordNetLemmatizer()
        
        doc[i] = [lemmatizer.lemmatize(w, pos='v') for w in doc[i]]
        
        
        #concat
        doc[i] = ' '.join(w for w in doc[i])
        
    return doc

@hydra.main(config_path='configs/data_preprocess.yaml')
def main(config):
    train_text = read_file(config.data.text.train)
    val_text = read_file(config.data.text.val)
    test_text = read_file(config.data.text.test)
    train_label = read_file(config.data.label.train)
    val_label = read_file(config.data.label.val)

    train_text = normalize(train_text)
    val_text = normalize(val_text)
    test_text = normalize(test_text)
    train_label = normalize(train_label)
    val_label = normalize(val_label)


    save_file(train_text, config.processed_data.text.train)
    save_file(val_text, config.processed_data.text.val)
    save_file(test_text, config.processed_data.text.test)
    save_file(train_label, config.processed_data.label.train)
    save_file(val_label, config.processed_data.label.val)

if __name__== '__main__':
    main()
    



















