from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import numpy as np

lemmatizer = WordNetLemmatizer()
stop_words = [
    '.', '?', '!', ',', ':', ';', '"', '\'', '-', '_', '*', '#', '@', '$', '%', '^', '&', '+', '=', '\\', '/', '<', '>',
] # stopwords.words('english')

def tokenize(sentence):
    return word_tokenize(sentence)

def stem(word):
    return lemmatizer.lemmatize(word)

def get_words_in_sentence(sentence):
    return [stem(word).lower() for word in tokenize(sentence) if word not in stop_words]

def get_bag_of_words(unique_words, words):
        bag = np.zeros(len(unique_words), dtype=np.float32)
        for idx, word in enumerate(unique_words):
            if word in words:
                bag[idx] = 1
        return bag

