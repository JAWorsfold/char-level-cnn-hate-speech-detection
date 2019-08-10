import numpy as np
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split


#I know I'm going to use a TfIdf Vectorizer so I'll build it here and write the functions after
vectorizer = TfidfVectorizer(
    preprocessor=preprocessor,
    tokenizer=tokenizer,
    stop_words=stopwords,
    ngram_range=(1, 3),
    decode_error='replace'
    )
