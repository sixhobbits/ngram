import datasets
import numpy as np
import logging
import os
import pprint

# import pipeline
# from lxml.etree import tostring
# from lxml.builder import E
from collections import defaultdict
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from collections import defaultdict
import time
import random
import dynet as dy

logging.basicConfig(level=logging.DEBUG)

logging.info("loading dataset...")
df = datasets.load_pan17("../data/training/")
corpus = df.corpus
corpus['text'] = corpus.text.apply(lambda x: '\n'.join(x))  # join tweets

corpus.head()

# split the dataset in train-dev-test
train_df = corpus[corpus.lang == 'en'].loc[3600:5600]
dev_df = corpus[corpus.lang == 'en'].loc[5601:6100]
test_df = corpus[corpus.lang == 'en'].loc[6101:6600]

# Functions to read in the corpus
w2i = defaultdict(lambda: len(w2i))
t2i_m = defaultdict(lambda: len(t2i_m)) # main labels
t2i_a = defaultdict(lambda: len(t2i_a)) # aux labels
UNK = w2i["<unk>"]


def read_dataset(dataframe):
    for index, row in dataframe.iterrows():
        main_label, aux_label, text = row.gender, row.variety, row.text
        yield ([w2i[x] for x in text.split(" ")], t2i_m[main_label], t2i_a[aux_label])


# Read in the data
train = list(read_dataset(train_df))[:30]
w2i = defaultdict(lambda: UNK, w2i)
dev = list(read_dataset(dev_df))[:30]
test = list(read_dataset(test_df))
nwords = len(w2i)
ntags1 = len(t2i_m)
ntags2 = len(t2i_a)

# Start DyNet and defin trainer
model = dy.Model()
trainer = dy.AdamTrainer(model)

# Define the model
EMB_SIZE = 64
HID_SIZE = 64
W_emb = model.add_lookup_parameters((nwords, EMB_SIZE))  # Word embeddings

fwdLSTM = dy.LSTMBuilder(1, EMB_SIZE, HID_SIZE, model)  # Forward LSTM
bwdLSTM = dy.LSTMBuilder(1, EMB_SIZE, HID_SIZE, model)

H_sm_main = model.add_parameters((64, 2 * HID_SIZE))  # Softmax weights
O_sm_main = model.add_parameters((1,64))  # Softmax bias

H_sm_aux = model.add_parameters((64, 2 * HID_SIZE))  # Softmax weights
O_sm_aux = model.add_parameters((ntags2,64))  # Softmax bias


# A function to calculate scores for one value
def calc_scores(words, main_tag, aux_tag):
    dy.renew_cg()
    word_embs = [dy.lookup(W_emb, x) for x in words]
    fwd_init = fwdLSTM.initial_state()
    bwd_init = bwdLSTM.initial_state()

    fwd_embs = fwd_init.transduce(word_embs)
    bwd_embs = bwd_init.transduce(reversed(word_embs))
    
    repr = dy.concatenate([fwd_embs[-1], bwd_embs[-1]]) # use last step as representation
    
    H_m = dy.parameter(H_sm_main)
    O_m = dy.parameter(O_sm_main)
    
    H_a = dy.parameter(H_sm_aux)
    O_a = dy.parameter(O_sm_aux)
    
    
    final_m = O_m*dy.tanh(H_m * repr) # MLP for main task
    final_a = O_a*dy.tanh(H_a * repr) # MLP for auxiliary task
    
    main = dy.binary_log_loss(final_m, dy.scalarInput(main_tag)) # gender
    aux = dy.pickneglogsoftmax(final_a, aux_tag) #variety
    return main, aux


for ITER in range(1):
    # Perform training
    random.shuffle(train)
    train_loss = 0.0
    start = time.time()
    for words, main_tag, aux_tag in train:
        loss = sum(calc_scores(words, main_tag, aux_tag))
        train_loss += loss.value()
        loss.backward()
        trainer.update()
    print("iter %r: train loss/sent=%.4f, time=%.2fs" % (ITER, train_loss / len(train), time.time() - start))

def predict_tag(words):
    dy.renew_cg()
    word_embs = [dy.lookup(W_emb, x) for x in words]
    fwd_init = fwdLSTM.initial_state()
    bwd_init = bwdLSTM.initial_state()
    ## Q2: run the forward pass of the LSTM
    fwd_embs = fwd_init.transduce(word_embs)
    bwd_embs = bwd_init.transduce(reversed(word_embs))
    repr = dy.concatenate([fwd_embs[-1], bwd_embs[-1]]) # use last step as representation
    
    H_m = dy.parameter(H_sm_main)
    O_m = dy.parameter(O_sm_main)
    
    H_a = dy.parameter(H_sm_aux)
    O_a = dy.parameter(O_sm_aux)
    
    final_m = O_m * dy.tanh(H_m*repr)
    main = dy.logistic(final_m)
    if main.value() > 0.5:
        gender =  1
    else:
        gender = 0
    
    final_a = O_a*dy.tanh(H_a * repr) # MLP for auxiliary task
    aux = dy.softmax(final_a) #variety
    variety = np.argmax(aux.npvalue())
    
    return gender, variety

# Eval
eval_correct_m = 0.0
eval_correct_a = 0.0
for words, main, aux in dev:
    gender, variety = predict_tag(words)
    if gender == main:
        eval_correct_m += 1
    if variety == aux:
        eval_correct_a += 1
print("iter %r: main eval acc=%.4f aux eval acc=%.4f" % (ITER, eval_correct_m / len(dev), eval_correct_a / len(dev)))
