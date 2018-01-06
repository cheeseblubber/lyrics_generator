from glob import glob

from torchtext import vocab, data
from fastai.learner import *
from fastai.rnn_reg import *
from fastai.rnn_train import *
from fastai.nlp import *
from fastai.lm_rnn import *

import spacy
from spacy.symbols import ORTH
import dill as pickle

#Process dataset - create dataset with 
PATH='/home/ubuntu/course-shortcut/competitions/lyrics_generator/'
TRN = "/home/ubuntu/course-shortcut/competitions/lyrics_generator/all/trn/"
VAL = "/home/ubuntu/course-shortcut/competitions/lyrics_generator/all/val/"

#Save the word list
trn_tokens = '/home/ubuntu/tokenized_words.pkl'
val_tokens = '/home/ubuntu/val_tokenized_words.pkl'
my_tok = spacy.load('en')
def my_spacy_tok(x): return [tok.text for tok in my_tok.tokenizer(x)]
text_field = data.Field(lower=True, tokenize=my_spacy_tok)

text = pickle.load(open(trn_tokens, 'rb'))
val_text = pickle.load(open(val_tokens, 'rb'))

def createDataSet(text_field, text):
    fields = [('text', text_field)]
    examples = [data.Example.fromlist([text], fields)]
    return data.Dataset(examples, fields)

trn_ds = createDataSet(text_field, text)
val_ds = createDataSet(text_field, val_text)
text_field.build_vocab(trn_ds, min_freq=10)

em_sz = 500
nh = 500
nl = 3
bs=32
bptt=300 #backpropogate through time: number of words it'll remember

FILES = dict(train=TRN, validation=VAL, test=VAL)

md = LanguageModelData.from_text_files(PATH, text_field, **FILES, bs=bs, bptt=bptt, min_freq=10)
optimization_function = partial(optim.Adam, betas=(0.7, 0.99))
learner = md.get_model(optimization_function, em_sz, nh, nl,
dropout=0.3, dropouth=0.3, dropouti=0.3, dropoute=0.3, wdrop=0.2)
learner.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)

learner.load('kam-test6')
learner.fit(3e-3, 1, wds=1e-6)
learner.save('kam-test7')
learner.fit(3e-4, 2, wds=1e-6, cycle_len=2, cycle_mult=2)
learner.save('kam-test8')
learner.fit(3e-5, 2, wds=1e-6, cycle_len=2, cycle_mult=2)
learner.save('kam-test9')
learner.fit(3e-5, 2, wds=1e-6, cycle_len=2, cycle_mult=2)
learner.save('kam-test10')
learner.fit(3e-5, 2, wds=1e-6, cycle_len=2, cycle_mult=2)
learner.save('kam-test11')
learner.fit(3e-5, 2, wds=1e-6, cycle_len=2, cycle_mult=2)
learner.save('kam-test12')
learner.fit(3e-6, 5, wds=1e-6, cycle_len=2, cycle_mult=2)
rearner.save('kam-test13')

