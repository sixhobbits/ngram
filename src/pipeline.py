import emoji
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer

emojis = [e for e in emoji.UNICODE_EMOJI]

# pipeline = Pipeline([('features',
#                       FeatureUnion([('emj', TfidfVectorizer(analyzer='char',
#                                                             vocabulary=emojis,
#                                                             preprocessor=None)),
#                                     ('wrd', TfidfVectorizer(analyzer='word'))])),
#                      ('clf', SVC(kernel='linear'))])
'''
pipeline = Pipeline([('features',
                      FeatureUnion([('emj', TfidfVectorizer(analyzer='char',
                                                            vocabulary=emojis,
                                                            preprocessor=None))])),
                     ('clf', SVC(kernel='linear'))])
'''


pipeline = Pipeline([('features', FeatureUnion([
                          
                          ('wrd', TfidfVectorizer(binary=False, max_df=1.0, min_df=2, norm='l2', sublinear_tf=True, use_idf=True, lowercase=True)),
                          ('char',TfidfVectorizer(analyzer='char', ngram_range=(3,6), binary=False, max_df=1.0, min_df=2, norm='l2', sublinear_tf=True, use_idf=True, lowercase=True))
                    ])),
                    ('clf', LinearSVC(C=1.0)) ])



