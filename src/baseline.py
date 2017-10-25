"""Blueprint for computing the baseline.
Use from command line:

[angelo@mangoni src]$ python baseline.py ./data/training ./output/
loading dataset...
setting up dummy pipeline...
performing cv on dummy pipeline...
Accuracy: 0.50 (+/- 0.00)
[angelo@mangoni src]$

"""
import datasets
import numpy as np
import logging
import os
import pprint
import argparse
import pipeline
from lxml.etree import tostring
from lxml.builder import E
from collections import defaultdict
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score

# accepts inputDir and outputDir variables from command line
parser = argparse.ArgumentParser(description='Run the pan17 pipeline')

parser.add_argument('inputDir', metavar='input', type=str, nargs='+',
                    help='input directory with the training data')
parser.add_argument('outputDir', metavar='output', type=str, nargs='?',
                    help='output directory for the XML files')
args = parser.parse_args()


logging.basicConfig(level=logging.DEBUG)

inputdir = args.inputDir[0]

try:
    outputdir = args.outputDir[0]
except:
    logging.info("Output dir not given: will not produce XML files")


def runbaseline():
    logging.info("loading dataset...")
    df = datasets.load_pan17(inputdir)
    corpus = df.corpus
    corpus['text'] = corpus.text.apply(lambda x: '\n'.join(x))  # join tweets

    # load pipeline
    pipe = pipeline.pipeline

    # initialize dict to store scores
    scores = defaultdict(dict)
    # initialize dict to store predictions for xml output
    output = {auth: {'id': auth,
                     'lang': lang} for auth, lang in zip(corpus.author,
                                                         corpus.lang)}
    # group by language and run cv for both variety and gender
    for language, subset in corpus.groupby('lang'):
        for target in [subset.variety, subset.gender]:
            logging.info("running cv on pipeline for %s on %s" %
                         (target.name, language))
            predictions = cross_val_predict(pipe,
                                            subset.text,  # X
                                            target,  # Y
                                            cv=2,
                                            verbose=False,
                                            n_jobs=-1)
            # compute scores from predictions
            scores[language][target.name] = accuracy_score(target,
                                                           predictions)
            # save predictions to output dict
            for auth, pred in zip(subset.author, predictions):
                output[auth][target.name] = pred
    # if outputdir is defined then:
    # shape output as xml and print to file in outputdir
    # otherwise we are done
    try:
        for entry in output:
            out = tostring(E.author(id=output[entry]['id'],
                                    lang=output[entry]['lang'],
                                    variety=output[entry]['variety'],
                                    gender=output[entry]['gender']),
                           pretty_print=True)
            with open(os.path.join(outputdir, entry), 'wb+') as f:
                f.write(out)
    except:
        logging.info("No output dir given. Done.")
    #
    logging.info(pprint.pprint(scores))
    logging.info("Averaged accuracy %0.2f" %
                 np.mean(list((list(x.values()) for x in scores.values()))))


if __name__ == "__main__":
    runbaseline()
