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
        self.vocabulary_size = None
        self.embedding_dict = None
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
        tokenizer = Tokenizer()  # initialize tokenizer (embedding look up table technique)
        tokenizer.fit_on_texts(self.data['Overview'].tolist())  # transform each word using the lookup table technique (each word transforms into a unique number for each unique word)
        self.vocabulary_size = len(tokenizer.word_index) + 1  # setting in the vocabulary size hyperparameter
        # load the whole embedding into memory
        embeddings_index = dict()
        f = open('glove.6B.100d.txt', encoding="utf8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print(f'Loaded {len(embeddings_index)} word vectors.')
        # create a weight matrix for words in training docs
        # embedding_matrix = np.zeros((self.vocabulary_size, 100))
        self.embedding_dict = dict()
        for word, i in tokenizer.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # embedding_matrix[i] = embedding_vector
                self.embedding_dict[word] = embedding_vector

    def __embedding(self, text):
        doc = []
        if len(text) == 0:
            print('empty')
            exit(0)
        for word in word_tokenize(text):
            if word.lower() in self.embedding_dict:
                doc.append(self.embedding_dict[word.lower()])
        # print(np.array(doc).mean(axis=0).shape)
        return np.array(doc).mean(axis=0)  # return the mean of text glove. maybe we should concat instead of mean!

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

    def get_dataset(self):
        return self.data

    def train_test_split(self, test_size=0.2):
        x_train, x_test, y_train, y_test = train_test_split(self.data[['Overview']], self.data[
            ['Western', 'Drama', 'Thriller', 'Mystery', 'Music', 'Romance', 'Action', 'Adventure', 'Foreign', 'Crime',
             'Documentary', 'Horror', 'Fantasy', 'History', 'Science Fiction', 'Family', 'TV Movie', 'Comedy',
             'Animation', 'War']], test_size=test_size)
        return x_train, x_test, y_train, y_test
