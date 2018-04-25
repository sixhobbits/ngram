import xmltodict
import logging
import glob
import os
import pandas as pd


# Main dataset loading utility
class load_pan17(object):
    """Load and return the pan17 gender and variation twitter dataset.
    ==============                                      ==============
    Samples total                                                10800
    Targets            nominal [{male, female},
                                {ar, pt, es, en},
                                {'brazil', 'australia', 'venezuela',
                                 'portugal', 'great britain', 'chile',
                                 'levantine', 'egypt', 'colombia',
                                 'peru', 'ireland', 'argentina',
                                 'maghrebi', 'mexico', 'new zealand',
                                 'spain', 'canada', 'gulf'}]
    ==============                                      ==============
    Parameters
    ----------
    inputdir
    The directory containing the training data, i.e. /data/training.

    Returns
    -------
    data : Pandas dataframe
        The interesting attributes are:
        'text', the data to learn, ['gender','lang', variety],
        the regression targets,
    Examples
    --------
    >>> from datasets import load_pan17
    >>> df_training = load_pan17(inputdir)
    >>> print(df_training.corpus.shape)
    (10800, 5)
    """
    def __init__(self, inputdir):
        self.directory = inputdir

        X_docs = glob.glob(os.path.join(self.directory, '**/*.xml'),
                           recursive=True)

        Y_docs = glob.glob(os.path.join(self.directory, '**/truth.txt'),
                           recursive=True)
        # check that the dataset is loaded correctly
        try:
            assert len(X_docs) == 11400
            assert len(Y_docs) == 4
        except:
            logging.warning("Problems with data in %s" % self.directory)
        X_tmp = []
        for t in X_docs:
            with open(t) as f:
                doc = xmltodict.parse(f.read())
            author = os.path.splitext(os.path.basename(t))[0]
            lang = doc['author']['@lang']
            text = doc['author']['documents']['document']
            X_tmp.append((author, lang, text))

        text = pd.DataFrame(X_tmp, columns=["author", "lang", "text"])

        Y_tmp = [pd.read_csv(l,
                             sep='\:\:\:',
                             names=['author', 'gender', 'variety'],
                             engine='python') for l in Y_docs]
        labels = pd.concat(Y_tmp)

        self.corpus = pd.merge(text, labels, on='author')


class load_testpan17(object):
    """Load and return the pan17 gender and variation twitter test dataset.
    ==============                                      ==============
    Parameters
    ----------
    testdir
    The directory containing the test data, i.e. /data/test.
    This will be assigned via command line argument

    Returns
    -------
    data : Pandas dataframe
        The interesting attributes are:
        'text', the data to learn, ['gender','lang', variety],
        the regression targets,
    Examples
    --------
    >>> from datasets import load_pan17
    >>> df_test = load_testpan17(testdir)
    >>> print(df_test.corpus.shape)
    """
    def __init__(self, testdir):
        self.directory = testdir

        X_docs = glob.glob(os.path.join(self.directory, '**/*.xml'),
                           recursive=True)
        X_tmp = []
        for t in X_docs:
            with open(t) as f:
                doc = xmltodict.parse(f.read())
            author = os.path.splitext(os.path.basename(t))[0]
            lang = os.path.abspath(os.path.join(t, os.pardir)).split('/')[-1]
            text = doc['author']['documents']['document']
            X_tmp.append((author, lang, text))

        text = pd.DataFrame(X_tmp, columns=["author", "lang", "text"])

        self.corpus = text
