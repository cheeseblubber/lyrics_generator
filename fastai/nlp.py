from .imports import *
from .torch_imports import *
from .core import *
from .model import *
from .dataset import *
from .learner import *
from .lm_rnn import *

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from torchtext.datasets import language_modeling

import spacy
from spacy.symbols import ORTH

re_br = re.compile(r'<\s*br\s*/?>', re.IGNORECASE)
def sub_br(x): return re_br.sub("\n", x)

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

def texts_from_files(src, names):
    texts,labels = [],[]
    for idx,name in enumerate(names):
        path = os.path.join(src, name)
        t = [o.strip() for o in open(path, encoding = "ISO-8859-1")]
        texts += t
        labels += ([idx] * len(t))
    return texts,np.array(labels)

def texts_from_folders(src, names):
    texts,labels = [],[]
    for idx,name in enumerate(names):
        path = os.path.join(src, name)
        for fname in sorted(os.listdir(path)):
            fpath = os.path.join(path, fname)
            texts.append(open(fpath).read())
            labels.append(idx)
    return texts,np.array(labels)

class DotProdNB(nn.Module):
    def __init__(self, nf, ny, w_adj=0.4, r_adj=10):
        super().__init__()
        self.w_adj,self.r_adj = w_adj,r_adj
        self.w = nn.Embedding(nf+1, 1, padding_idx=0)
        self.w.weight.data.uniform_(-0.1,0.1)
        self.r = nn.Embedding(nf+1, ny)

    def forward(self, feat_idx, feat_cnt, sz):
        w = self.w(feat_idx)
        r = self.r(feat_idx)
        x = ((w+self.w_adj)*r/self.r_adj).sum(1)
        return F.softmax(x)

class SimpleNB(nn.Module):
    def __init__(self, nf, ny):
        super().__init__()
        self.r = nn.Embedding(nf+1, ny, padding_idx=0)
        self.b = nn.Parameter(torch.zeros(ny,))

    def forward(self, feat_idx, feat_cnt, sz):
        r = self.r(feat_idx)
        x = r.sum(1)+self.b
        return F.softmax(x)

class BOW_Learner(Learner):
    def __init__(self, data, models, **kwargs):
        super().__init__(data, models, **kwargs)
        self.crit = F.l1_loss

def calc_pr(y_i, x, y, b):
    idx = np.argwhere((y==y_i)==b)
    p = x[idx[:,0]].sum(0)+1
    return p/((y==y_i)==b).sum()

def calc_r(y_i, x, y):
    return np.log(calc_pr(y_i, x, y, True) / calc_pr(y_i, x, y, False))

class BOW_Dataset(Dataset):
    def __init__(self, bow, y, max_len):
        self.bow,self.max_len = bow,max_len
        self.c = int(y.max())+1
        self.n,self.vocab_size = bow.shape
        self.y = one_hot(y,self.c).astype(np.float32)
        x = self.bow.sign()
        self.r = np.stack([calc_r(i, x, y).A1 for i in range(self.c)]).T

    def __getitem__(self, i):
        row = self.bow.getrow(i)

        num_row_entries = row.indices.shape[0]
        indices = (row.indices + 1).astype(np.int64)
        data = (row.data).astype(np.int64)

        if num_row_entries < self.max_len:
            # If short, pad
            indices = np.pad(indices, (self.max_len - num_row_entries, 0), mode='constant')
            data = np.pad(data, (self.max_len - num_row_entries, 0), mode='constant')
        else:
            # If long, truncate
            indices, data = indices[-self.max_len:], data[-self.max_len:]

        return indices, data, min(self.max_len, num_row_entries), self.y[i]

    def __len__(self): return len(self.bow.indptr)-1


class TextClassifierData(ModelData):
    @property
    def c(self): return self.trn_ds.c

    @property
    def r(self):
        return torch.Tensor(np.concatenate([np.zeros((1,self.c)), self.trn_ds.r]))

    def get_model(self, f, **kwargs):
        m = to_gpu(f(self.trn_ds.vocab_size, self.c, **kwargs))
        m.r.weight.data = to_gpu(self.r)
        m.r.weight.requires_grad = False
        model = BasicModel(m)
        return BOW_Learner(self, model, metrics=[accuracy_thresh(0.5)], opt_fn=optim.Adam)

    def dotprod_nb_learner(self, **kwargs): return self.get_model(DotProdNB, **kwargs)
    def nb_learner(self, **kwargs): return self.get_model(SimpleNB, **kwargs)

    @classmethod
    def from_bow(cls, trn_bow, trn_y, val_bow, val_y, sl):
        trn_ds = BOW_Dataset(trn_bow, trn_y, sl)
        val_ds = BOW_Dataset(val_bow, val_y, sl)
        trn_dl = DataLoader(trn_ds, 64, True)
        val_dl = DataLoader(val_ds, 64, False)
        return cls('.', trn_dl, val_dl)


class LanguageModelLoader():

    def __init__(self, ds, bs, bptt):
        self.bs,self.bptt = bs,bptt
        text = sum([o.text for o in ds], [])
        fld = ds.fields['text']
        nums = fld.numericalize([text], device=-1)
        self.data = self.batchify(nums)
        self.i,self.iter = 0,0
        self.n = len(self.data)

    def __iter__(self):
        self.i,self.iter = 0,0
        return self

    def __len__(self): return self.n // self.bptt - 1

    def __next__(self):
        if self.i >= self.n-1 or self.iter>=len(self): raise StopIteration
        bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        res = self.get_batch(self.i, seq_len)
        self.i += seq_len
        self.iter += 1
        return res

    def batchify(self, data):
        nb = data.size(0) // self.bs
        data = data[:nb*self.bs]
        data = data.view(self.bs, -1).t().contiguous()
        return to_gpu(data)

    def get_batch(self, i, seq_len):
        source = self.data
        seq_len = min(seq_len, len(source) - 1 - i)
        return source[i:i+seq_len], source[i+1:i+1+seq_len].view(-1)


class RNN_Learner(Learner):
    def __init__(self, data, models, **kwargs):
        super().__init__(data, models, **kwargs)
        self.crit = F.cross_entropy

    def save_encoder(self, name): save_model(self.model[0], self.get_model_path(name))
    def load_encoder(self, name): load_model(self.model[0], self.get_model_path(name))


# class ConcatTextDataset(torchtext.data.Dataset):
#     def __init__(self, path, text_field, newline_eos=True, **kwargs):
#         fields = [('text', text_field)]
#         text = []
#         if os.path.isdir(path): paths=glob(f'{path}/*.*')
#         else: paths=[path]
#         for p in paths:
#             for line in open(p): text += text_field.preprocess(line)
#             if newline_eos: text.append('<eos>')

#         examples = [torchtext.data.Example.fromlist([text], fields)]
#         super().__init__(examples, fields, **kwargs)


class ConcatTextDataset(torchtext.data.Dataset):
    def __init__(self, path, text_field, text, newline_eos=True, **kwargs):
        fields = [('text', text_field)]
        # text = kwargs['text']
        # text = []
        # if os.path.isdir(path): paths=glob(f'{path}/*.*')
        # else: paths=[path]
        # for p in paths:
        #     for line in open(p): text += text_field.preprocess(line)
        #     if newline_eos: text.append('<eos>')

        # import pdb; pdb.set_trace()

        examples = [torchtext.data.Example.fromlist([text], fields)]
        super().__init__(examples, fields, **kwargs)

        


class ConcatTextDatasetFromDataFrames(torchtext.data.Dataset):
    def __init__(self, df, text_field, col, newline_eos=True, **kwargs):
        fields = [('text', text_field)]
        text = []

        text += text_field.preprocess(df[col].str.cat(sep=' <eos> '))
        if (newline_eos): text.append('<eos>')

        examples = [torchtext.data.Example.fromlist([text], fields)]

        super().__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, train_df=None, val_df=None, test_df=None, **kwargs):
        train_data = None if train_df is None else cls(train_df, **kwargs)
        val_data = None if val_df is None else cls(val_df, **kwargs)
        test_data = None if test_df is None else cls(test_df, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data) if d is not None)


class LanguageModelData():
    def __init__(self, path, field, train, validation, test=None, bs=64, bptt=70, **kwargs):
        self.path,self.bs = path,bs
        import dill as pickle
        trn_text = pickle.load(open('tokenized_words.pkl', 'rb'))
        val_text = pickle.load(open('tokenized_words.pkl', 'rb'))
        test_text = pickle.load(open('tokenized_words.pkl', 'rb'))

        self.trn_ds = ConcatTextDataset(path,text_field=field,  text=trn_text)
        self.val_ds = ConcatTextDataset(path,text_field=field,  text=val_text)
        self.test_ds = ConcatTextDataset(path,text_field=field, text=test_text)
        field.build_vocab(self.trn_ds, **kwargs)
        self.pad_idx = field.vocab.stoi[field.pad_token]
        self.nt = len(field.vocab)
        self.trn_dl,self.val_dl,self.test_dl = [LanguageModelLoader(ds, bs, bptt) for ds in
                                               (self.trn_ds,self.val_ds,self.test_ds)]

    def get_model(self, opt_fn, emb_sz, n_hid, n_layers, **kwargs):
        """ Method returns a RNN_Learner object, that wraps an instance of the RNN_Encoder module.

        Args:
            opt_fn (Optimizer): the torch optimizer function to use
            emb_sz (int): embedding size
            n_hid (int): number of hidden inputs
            n_layers (int): number of hidden layers
            kwargs: other arguments

        Returns:
            An instance of the RNN_Learner class.

        """
        m = get_language_model(self.bs, self.nt, emb_sz, n_hid, n_layers, self.pad_idx, **kwargs)
        model = SingleModel(to_gpu(m))
        return RNN_Learner(self, model, opt_fn=opt_fn)

    @classmethod
    def from_dataframes(cls, path, field, col, train_df, val_df, test_df=None, bs=64, bptt=70, **kwargs):
        trn_ds, val_ds, test_ds = ConcatTextDatasetFromDataFrames.splits(text_field=field, col=col,
                                    train_df=train_df, val_df=val_df, test_df=test_df)

        return cls(path, field, trn_ds, val_ds, test_ds, bs, bptt, **kwargs)

    @classmethod
    def from_text_files(cls, path, field, train, validation, test=None, bs=64, bptt=70, **kwargs):
        """ Method used to instantiate a LanguageModelData object that can be used for a
            supported nlp task.

        Args:
            path (str): the absolute path in which temporary model data will be saved
            field (Field): torchtext field
            train (str): file location of the training data
            validation (str): file location of the validation data
            test (str): file location of the testing data
            bs (int): batch size to use
            bptt (int): back propagation through time hyper-parameter
            kwargs: other arguments

        Returns:
            a LanguageModelData instance, which most importantly, provides us the datasets for training,
                validation, and testing

        Note:
            The train, validation, and test path can be pointed to any file (or folder) that contains a valid
                text corpus.

        """
        trn_ds, val_ds, test_ds = ConcatTextDataset.splits(
                                    path, text_field=field, train=train, validation=validation, test=test)

        return cls(path, field, trn_ds, val_ds, test_ds, bs, bptt, **kwargs)


class TextDataLoader():
    def __init__(self, src, x_fld, y_fld):
        self.src,self.x_fld,self.y_fld = src,x_fld,y_fld

    def __len__(self): return len(self.src)-1

    def __iter__(self):
        it = iter(self.src)
        for i in range(len(self)):
            b = next(it)
            yield getattr(b, self.x_fld), getattr(b, self.y_fld)


class TextModel(BasicModel):
    def get_layer_groups(self):
        m = self.model[0]
        return [m.encoder, *m.rnns, [m.dropouti, m.dropouth], self.model[1]]


class TextData(ModelData):
    def create_td(self, it): return TextDataLoader(it, self.text_fld, self.label_fld)

    @classmethod
    def from_splits(cls, path, splits, bs, text_name='text', label_name='label'):
        text_fld = splits[0].fields[text_name]
        label_fld = splits[0].fields[label_name]
        if hasattr(label_fld, 'build_vocab'):
            label_fld.build_vocab(splits[0])
        iters = torchtext.data.BucketIterator.splits(splits, batch_size=bs)
        trn_iter,val_iter,test_iter = iters[0],iters[1],None
        test_dl = None
        if len(iters) == 3:
            test_iter = iters[2]
            test_dl = TextDataLoader(test_iter, text_name, label_name)
        trn_dl = TextDataLoader(trn_iter, text_name, label_name)
        val_dl = TextDataLoader(val_iter, text_name, label_name)
        obj = cls.from_dls(path, trn_dl, val_dl, test_dl)
        obj.bs = bs
        obj.pad_idx = text_fld.vocab.stoi[text_fld.pad_token]
        obj.nt = len(text_fld.vocab)
        obj.c = (len(label_fld.vocab) if hasattr(label_fld, 'vocab')
                 else len(getattr(splits[0][0], label_name)))
        return obj

    def to_model(self, m, opt_fn):
        model = TextModel(to_gpu(m))
        return RNN_Learner(self, model, opt_fn=opt_fn)

    def get_model(self, opt_fn, max_sl, bptt, emb_sz, n_hid, n_layers, **kwargs):
        m = get_rnn_classifer(max_sl, bptt, self.bs, self.c, self.nt,
              emb_sz=emb_sz, n_hid=n_hid, n_layers=n_layers, pad_token=self.pad_idx, **kwargs)
        return self.to_model(m, opt_fn)

