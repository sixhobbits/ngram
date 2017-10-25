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
from sklearn.metrics import accuracy_score
from lxml.etree import tostring
from lxml.builder import E

# accepts inputDir and outputDir variables from command line
parser = argparse.ArgumentParser(description='Run the pan17 pipeline')

parser.add_argument('testDir', metavar='testing', type=str, 
                    help='input directory with the test data')
parser.add_argument('outputDir', metavar='output', type=str, 
                    help='output directory for the XML files')
args = parser.parse_args()


logging.basicConfig(level=logging.DEBUG)

inputdir = "/media/training-datasets/author-profiling/pan17-author-profiling-training-dataset-2017-03-10/"
testdir = args.testDir

print(args)

try:
    outputdir = args.outputDir
except:
    logging.info("Output dir not given: will not produce XML files")


def runbaseline():
    logging.info("loading dataset...")
    df = datasets.load_pan17(inputdir)
    corpus = df.corpus
    corpus['text'] = corpus.text.apply(lambda x: '\n'.join(x))  # join tweets

    df_test = datasets.load_testpan17(testdir)
    test_corpus = df_test.corpus
    test_corpus['text'] = test_corpus.text.apply(lambda x: '\n'.join(x))

    # load pipeline
    pipe = pipeline.pipeline

    # group by language and run cv for both variety and gender
    for language, subset in corpus.groupby('lang'):
        os.makedirs(os.path.join(outputdir, language))
        # initialize dict to store predictions for xml output
        output = {auth: {'id': auth} for auth in test_corpus.author[language]}
        for target in [subset.variety, subset.gender]:
            logging.info("running pipeline for %s on %s" %
                         (target.name, language))
            pipe.fit(subset.text,  # X
                     target)  # Y
            test_subset = test_corpus.groupby('lang').get_group(language)
            Xtest = test_subset.text
            predictions = pipe.predict(Xtest)
            # print(accuracy_score(target, predictions))
            # save predictions to output dict
            for auth, pred in zip(test_subset.author, predictions):
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
            with open(os.path.join(os.path.join(outputdir, output[entry]['lang']), entry), 'wb+') as f:
                f.write(out)
    except:
        logging.info("No output dir given. Done.")
    #


if __name__ == "__main__":
    runbaseline()
