## Rap Lyrics Generator (Lyrics somewhat NSFW)

A lyric generator that is generated from top [billboard top lyrics](https://www.kaggle.com/rakannimer/billboard-top-lyrics-analysis/data)

Model was trained on an aws-p2 instance

### Data cleaning and collection

Retried lyrics categorized as 'Hip-Hop' from 2007 - 2017 from lyrics generator
- Had to remove some songs form other languages manually by searching for common strings in other languages
- Randomly split dataset to VAL,TEST,TRN

See `model_generation/lyrics_generator-v1.ipynb` for more details - first attempt

### Deep Learning Model

```
SequentialRNN(
  (0): RNN_Encoder(
    (encoder): Embedding(16709, 500, padding_idx=1)
    (encoder_with_dropout): EmbeddingDropout(
      (embed): Embedding(16709, 500, padding_idx=1)
    )
    (rnns): ModuleList(
      (0): WeightDrop(
        (module): LSTM(500, 500, dropout=0.3)
      )
      (1): WeightDrop(
        (module): LSTM(500, 500, dropout=0.3)
      )
      (2): WeightDrop(
        (module): LSTM(500, 500, dropout=0.3)
      )
    )
    (dropouti): LockedDropout(
    )
    (dropouth): LockedDropout(
    )
  )
  (1): LinearDecoder(
    (decoder): Linear(in_features=500, out_features=16709)
    (dropout): LockedDropout(
    )
  )
)
```

Specs of original model:
Word Embedding Size = 200
number of hidden activations per layer = 500
number of layers = 3
Dropout=10%

See `model_generation/lyrics_generator-v1.ipynb` for latest model training

## Experiments


### Using pretrained embeddings

Model had issues of words repeating and getting stuck in a loop. Increased 
dropout which reduce the issue. Also lyrics are are typically repetitive especially with
with choruses. 

ATM experimenting with loading pretrained word embeddings from [fastText](https://github.com/facebookresearch/fastText)
Was able to load the pretrained embeddings. 

Next steps:

* Freeze embeddings and train other layers for a few iterations first.
* After few iterations unfreeze embeddings and train with small learning rate for embeddings
* Increase learning rate of embeddings model and train for few iterations.

#### Hack to get around looping model

See the method `sample_model()`. Randomly sampling from the highest few probable next words.

## Background Training

Training in Jupyter notebook was limiting since it had to be opened for it to train.
See `model_generation/generate.py` for details of running script over night
Use tmux + generate.py for overnight training

## Running the model

The weights are saved to pickle files along with the vocab of training data 
the latest trained model is `kam-test10`. This is not trained with fastText embeddings

### TODOS

- Improve response time with model
- Export model to ONNX with torch.onnx
- Update paths to be relative so can be ran on any machine
- pull out commonly used methods like `get_model` and `sample_model`
- Try Seq2Seq architecture 

### COMING SOON!!

`Shittyrap.com` - Once I optimize performance on this Django app

### Special Thanks

Some of the code was reused from with some modifications `github.com/fastai/fastai`
