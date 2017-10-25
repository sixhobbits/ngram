from __future__ import print_function
import xmltodict
import glob
import os
import pandas as pd
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from statistics import mean
from datetime import datetime
from nltk.tokenize.casual import TweetTokenizer
from time import time
import logging
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
warnings.filterwarnings("ignore", category=UserWarning)

data_dir = "/media/training-datasets/author-profiling/"
pan17_root = "pan17-author-profiling-training-dataset-2017-03-10"
ar17 = os.path.join(pan17_root, "ar")
en17 = os.path.join(pan17_root, "en")
es17 = os.path.join(pan17_root, "es")
pt17 = os.path.join(pan17_root, "pt")
pan17 = [os.path.join(data_dir, d) for d in [ar17, en17, es17, pt17]]

class PanDataLoader:                               
    def load_17(self, directory):
        X_docs = glob.glob(os.path.join(directory, '*.xml'), recursive=True)
        Y_doc = os.path.join(directory, 'truth.txt')
        X_tmp = []
        for t in X_docs:
            with open(t) as f:
                doc = xmltodict.parse(f.read())
            author = os.path.splitext(os.path.basename(t))[0]
            lang = doc['author']['@lang']
            text = doc['author']['documents']['document']
            X_tmp.append((author, lang, text))
        text = pd.DataFrame(X_tmp, columns=["author", "lang", "text"])
        Y_tmp = pd.read_csv(Y_doc, sep='\:\:\:', names=['author', 'gender', 'variety'], engine='python')
        corpus = pd.merge(text, Y_tmp, on='author')
        return corpus

    def _load_all(self, loader_func, directories):
        """Concatenate across languages"""
        corpora = []
        for dr in directories:
            corpus = loader_func(dr)
            corpora.append(corpus)
        return pd.concat(corpora)
    
    def load_all_17(self, directories):
        return self._load_all(self.load_17, directories)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LinearSVC())
])

parameters = {
    'tfidf__lowercase': (True, False),
    'tfidf__max_df': (0.01, 1.0), # ignore words that occur as more than 1% of corpus
    'tfidf__min_df': (1, 2, 3), # we need to see a word at least (once, twice, thrice)
    'tfidf__use_idf': (False, True),
    'tfidf__sublinear_tf': (False, True),
    'tfidf__binary': (False, True),
    'tfidf__norm': (None, 'l1', 'l2'),
    'clf__C':(0.1, 0.5, 1, 1.5, 5)
}

'''
if __nf__C: 5
        tfidf__binary: False
        tfidf__lowercase: True
        tfidf__max_df: 1.0
        tfidf__min_df: 2
        tfidf__norm: 'l2'
        tfidf__sublinear_tf: True
        tfidf__use_idf: True
'''

if __name__ == "__main__":
    pdl = PanDataLoader()
    corpus17 = pdl.load_all_17(pan17)
    vectorizer = TfidfVectorizer(binary=False, max_df=1.0, min_df=2, norm='l2', sublinear_tf=True, use_idf=True)
    classifier = LinearSVC(C=5)

    print("loaded corpus...")
    for lang in ['ar', 'en', 'es', 'pt']:
        sub = corpus17[corpus17['lang'] == lang]
        sub['text'] = sub['text'].apply(lambda x: '\n'.join(x))
        print("joined. Vectorzing...")
        Xs = vectorizer.fit_transform(sub['text'])
        ys = sub['gender']
        gscore = cross_val_score(classifier, Xs, sub['gender'], cv=5, n_jobs=-1)
        vscore = cross_val_score(classifier, Xs, sub['variety'], cv=5, n_jobs=-1)
        print("--{}--".format(lang))
        print("Gender: {}; mean: {}".format(gscore, mean(gscore)))
        print("Variety: {}; mean: {}".format(vscore, mean(vscore)))
        print("--------")



# name == __main__ is important for multi process
if __name__ == "__main2__":
    pdl = PanDataLoader()
    corpus17 = pdl.load_all_17(pan17)
    ar = corpus17[corpus17['lang'] == 'ar']
    ar['text'] = ar['text'].apply(lambda x: '\n'.join(x))
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=2)
    t0 = time()
    grid_search.fit(ar['text'], ar['gender'])

    print("done in %0.3fs" % (time() - t0))
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
