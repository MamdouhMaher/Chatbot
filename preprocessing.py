import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import re
import os

lemmatizer = WordNetLemmatizer()

# Add your custom nltk_data path if you uploaded it
nltk.data.path.append(os.path.join(os.path.dirname(__file__), "nltk_data"))

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=os.path.join(os.path.dirname(__file__), "nltk_data"))

def clean_text(text, language="en"):
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text

def tokenize(sentence, language="en"):
    sentence = clean_text(sentence, language)
    if language == "ar":
        return sentence.split()
    else:
        return nltk.word_tokenize(sentence)

def bag_of_words(tokens, words, language="en"):
    if language == "en":
        tokens = [lemmatizer.lemmatize(w) for w in tokens]
    bag = [0] * len(words)
    for w in tokens:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

