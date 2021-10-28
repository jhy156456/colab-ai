import sys
from pathlib import Path

import pandas as pd

import spacy
from spacy import displacy
from textacy.extract import ngrams, entities
def get_attributes(f):
    print([a for a in dir(f) if not a.startswith('_')], end=' ')
nlp = spacy.load('en_core_web_sm')
sample_text = 'Apple is looking at buying U.K. startup for $1 billion'
doc = nlp(sample_text)
get_attributes(doc)