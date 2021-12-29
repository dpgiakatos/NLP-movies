import numpy as np
import pandas as pd
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class Dataset:
    def __init__(self, punctuation=False, stop_words=False, stem=False, embedding=False):
        self.embeddings_index = None
        self.length_long_sentence = -1
        self.data = pd.read_csv('data/tmdb_5000_movies.csv')[['genres', 'overview']]
        self.data.dropna(inplace=True)
        self.__clean()
        self.__preprocess(punctuation, stop_words, stem, embedding)

    def __clean(self):
        data = {
            'Overview': [],
            'Western': [],
            'Drama': [],
            'Thriller': [],
            'Mystery': [],
            'Music': [],
            'Romance': [],
            'Action': [],
            'Adventure': [],
            'Foreign': [],
            'Crime': [],
            'Documentary': [],
            'Horror': [],
            'Fantasy': [],
            'History': [],
            'Science Fiction': [],
            'Family': [],
            'TV Movie': [],
            'Comedy': [],
            'Animation': [],
            'War': []
        }
        for index, value in self.data.iterrows():
            obj = json.loads(value['genres'])
            if len(obj) == 0:
                continue
            data['Overview'].append(value['overview'])
            genres = []
            for genre in obj:
                genres.append(genre['name'])
            for key in data:
                if key == 'Overview':
                    continue
                elif key in genres:
                    data[key].append(1)
                else:
                    data[key].append(0)
        self.data = pd.DataFrame(data)

    @staticmethod
    def __punctuation(text):
        punc = '''!()-[]{};:'"\,<>’./?@#$%^&*_~'''
        for word in text:
            if word in punc:
                text = text.replace(word, '')
        return text

    @staticmethod
    def __stop_words(text):
        stop_words = set(stopwords.words('english'))
        new_text = []
        for word in word_tokenize(text):
            if word.lower() not in stop_words:
                new_text.append(word)
        return ' '.join(new_text) if len(new_text) else np.nan

    @staticmethod
    def __stem(text):
        ps = PorterStemmer()
        new_test = []
        for word in word_tokenize(text):
            new_test.append(ps.stem(word))
        return ' '.join(new_test)

    def __glove(self):
        # load the whole embedding into memory
        self.embeddings_index = dict()
        f = open('glove.6B.100d.txt', encoding="utf8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
        f.close()
        print(f'Loaded {len(self.embeddings_index)} word vectors.')

    def __embedding(self, text):
        doc = []
        if len(text) == 0:
            print('empty')
            exit(0)
        for word in word_tokenize(text):
            if word.lower() in self.embeddings_index:
                doc.append(self.embeddings_index[word.lower()])
        if len(doc) > self.length_long_sentence:
            self.length_long_sentence = len(doc)
        return doc

    def __pad_sequences(self, embedded):
        zeros = np.zeros(100)
        for _ in range(len(embedded), self.length_long_sentence):
            embedded.append(zeros)
        return np.array(embedded)

    def __preprocess(self, punctuation, stop_words, stem, embedding):
        if punctuation:
            self.data['Overview'] = self.data['Overview'].apply(lambda row: Dataset.__punctuation(row))
        if stop_words:
            self.data['Overview'] = self.data['Overview'].apply(lambda row: Dataset.__stop_words(row))
            self.data.dropna(inplace=True)
        if stem:
            self.data['Overview'] = self.data['Overview'].apply(lambda row: Dataset.__stem(row))
        if embedding:
            self.__glove()
            self.data['Overview'] = self.data['Overview'].apply(lambda row: self.__embedding(row))
            self.data['Overview'] = self.data['Overview'].apply(lambda row: self.__pad_sequences(row))

    def get_dataset(self):
        return self.data

    def train_test_split(self, test_size=0.2):
        x_train, x_test, y_train, y_test = train_test_split(self.data[['Overview']], self.data[
            ['Western', 'Drama', 'Thriller', 'Mystery', 'Music', 'Romance', 'Action', 'Adventure', 'Foreign', 'Crime',
             'Documentary', 'Horror', 'Fantasy', 'History', 'Science Fiction', 'Family', 'TV Movie', 'Comedy',
             'Animation', 'War']], test_size=test_size)
        return x_train, x_test, y_train, y_test
