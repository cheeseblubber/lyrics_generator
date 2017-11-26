from django.db import models
import sys
import os
os_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(os_dir, '../fastai'))
# This file contains all the main external libs we'll use
from fastai.imports import *
from fastai.torch_imports import *
from fastai.core import *
from fastai.model import fit
from fastai.dataset import *

import torchtext
from torchtext import vocab, data
from torchtext.datasets import language_modeling
from torch.autograd import Variable
# import torch.onnx

from fastai.rnn_reg import *
from fastai.rnn_train import *
from fastai.nlp import *
from fastai.lm_rnn import *
import spacy

PATH='fastai/'
TRN = "trn/"
VAL = "val/"
from spacy.symbols import ORTH
from random import randint
import dill as pickle


class Lyrics_Generator():

    def get_model(self):
        em_sz = 500
        nh = 500
        nl = 3
        bs=32
        bptt=300 #backpropogate through time: number of words it'll remember
        my_tok = spacy.load('en')
        def my_spacy_tok(x): return [tok.text for tok in my_tok.tokenizer(x)]
        text_field = data.Field(lower=True, tokenize=my_spacy_tok)
        FILES = dict(train=TRN, validation=VAL, test=VAL)
        md = LanguageModelData(PATH, text_field, **FILES, bs=bs, bptt=bptt, min_freq=10)
        optimization_function = partial(optim.Adam, betas=(0.7, 0.99))
        learner = md.get_model(optimization_function, em_sz, nh, nl, dropout=0.05, dropouth=0.1, dropouti=0.05, dropoute=0.02, wdrop=0.2)
        learner.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
        learner.load('/Users/cheeseblubber/workspace/rap_generator/lyric_generator/kam-test10')

        self.model = learner.model
        self.text_field = text_field

    def sample_model(self, s, num_words=50, random_level=4):
        text_field = self.text_field
        model = self.model
        def proc_str(s): return text_field.preprocess(text_field.tokenize(s))
        def num_str(s): return text_field.numericalize([proc_str(s)], device=-1)
        words = s
        t = num_str(s)
        model[0].bs=1
        model.eval()
        model.reset()
        res,*_ = model(t)
        print('...', end='')
        results = s

        start = datetime.datetime.now()
        for i in range(num_words):
            pred = res
            # get top number of predictions
            rand = randint(0, random_level - 1)
            _, top_indices = pred[-1].topk(random_level)
            # Randomly sample one word
            n = top_indices.data[rand]
            word = text_field.vocab.itos[n]
            # print(word, end=' ')
            if word=='<eos>': break
            if word=='<unk>': continue
            # if word==',': print("\n")

            words += " " + word
            results += " " + word
            num_words_to_remember = 100
            if len(words.split(" ")) > num_words_to_remember:
                words = " ".join(words.split(" ")[-num_words_to_remember:])
            res,*_ = model(num_str(words))

        print((datetime.datetime.now() - start).total_seconds())

        return results

