import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Conv1D, Activation, MaxPooling1D
from keras.layers import Input, Embedding, Flatten, Dense
from keras.models import Model

# cnn parameters:
VOCAB_SIZE = 70
MAX_LEN = 140

