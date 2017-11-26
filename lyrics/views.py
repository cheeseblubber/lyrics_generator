from django.shortcuts import render
from django.http import HttpResponse
from lyrics.models import *
from random import randint
from .forms import LyricForm

import dill as pickle

PATH='fastai/'
# TRN = "/home/ubuntu/course-shortcut/competitions/lyrics_generator/all/trn/"
# VAL = "/home/ubuntu/course-shortcut/competitions/lyrics_generator/all/val/"
from spacy.symbols import ORTH
# import spacy.lang
from random import randint

generator = None

def index(request):
    global generator
    if generator is None:
        generator = Lyrics_Generator()
        generator.get_model()
    song = ""
    if request.method == 'POST':
        form = LyricForm(request.POST)
        if form.is_valid():
            input_lyrics = form.cleaned_data['get_lyric']
            lyrics = generator.sample_model(input_lyrics, num_words=200)
            song = "\n".join( lyrics.split(","))

    form = LyricForm()

    return render(request, 'lyric.html', {'form': form, 'song': song})


