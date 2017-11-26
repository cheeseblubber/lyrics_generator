from django import forms

class LyricForm(forms.Form):
    get_lyric = forms.CharField(label='Enter Lyric', max_length=100)
