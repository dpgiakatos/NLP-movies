import numpy as np
import pandas as pd
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
import fasttext
import os
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer


class Dataset:
    def __init__(self, punctuation=False, stop_words=False, stem=False, embedding='glove'):
        # Dataset class used for the pre-processing phase
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
        #  Removing all the punctuation from the overviews
        punc = '''!(+)-[|]{};:'"\,<=>’./?@#$%^&*_~'''
        for word in text:
            if word in punc:
                text = text.replace(word, '')
        return text

    @staticmethod
    def __stop_words(text):
        #  Removing all the stopwords from the overviews
        stop_words = set(stopwords.words('english'))
        new_text = []
        for word in word_tokenize(text):
            if word.lower() not in stop_words:
                new_text.append(word)
        return ' '.join(new_text) if len(new_text) else np.nan

    @staticmethod
    def __stem(text):
        # Using PorterStemmer to apply stemming to the overviews
        ps = PorterStemmer()
        new_test = []
        for word in word_tokenize(text):
            new_test.append(ps.stem(word))
        return ' '.join(new_test)

    def __glove(self):
        # Loading the whole embedding into memory and creating a dictionary where the keys are the words of the
        # overviews and the values are the embeddings
        self.embeddings_index = dict()
        f = open('glove.6B.100d.txt', encoding="utf8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
        f.close()
        print(f'Loaded {len(self.embeddings_index)} word vectors.')

    def __fasttext(self):
        # Get word embedding using fasttext
        self.data['Overview'].to_csv('temp_text.txt', header=None, index=None, mode='a')
        self.embeddings_model = fasttext.train_unsupervised('temp_text.txt', model='skipgram')
        os.remove('temp_text.txt')

    def __word2vec(self):
        # Get word embedding using word2vec
        sentences = []
        for i in self.data['Overview'].to_list():
            sentences.append(i.split(' '))
        self.embeddings_model = Word2Vec(sentences)

    def __tfidf(self):
        # Get word embedding using tfidf
        self.embeddings_model = TfidfVectorizer()
        self.embeddings_model.fit(self.data['Overview'].to_list())

    def __embedding(self, text, method='glove'):
        # Replacing each overview with a list of word embeddings
        doc = []
        if len(text) == 0:
            print('empty')
            exit(0)
        for word in word_tokenize(text):
            if method == 'glove':
                if word.lower() in self.embeddings_index:
                    doc.append(self.embeddings_index[word.lower()])
            elif method == 'fasttext':
                doc.append(self.embeddings_model.get_word_vector(word.lower()))
            elif method == 'word2vec':
                try:
                    doc.append(self.embeddings_model.wv[word.lower()])
                except:
                    pass
        if len(doc) > self.length_long_sentence:
            self.length_long_sentence = len(doc)
        return doc

    def __pad_sequences(self, embedded):
        # Filling the sentences with zero embeddings in order to have same sentences length
        zeros = np.zeros(100)
        for _ in range(len(embedded), self.length_long_sentence):
            embedded.append(zeros)
        return np.array(embedded)

    def __vectorize(self, embedding):
        # Create a 3D array from vectorize methods, like tfidf, to have the same input format as dense vectorize methods
        if embedding == 'tfidf':
            vectors = self.embeddings_model.transform(self.data['Overview'].to_list()).toarray().tolist()
        doc = []
        for vector in vectors:
            temp = []
            for ell in vector:
                temp.append([ell])
            doc.append(temp)
        return doc

    def __preprocess(self, punctuation, stop_words, stem, embedding):
        # Secret function to apply the pre-processing functions based on the initialization of the dataset class
        if punctuation:
            self.data['Overview'] = self.data['Overview'].apply(lambda row: Dataset.__punctuation(row))
        if stop_words:
            self.data['Overview'] = self.data['Overview'].apply(lambda row: Dataset.__stop_words(row))
            self.data.dropna(inplace=True)
        if stem:
            self.data['Overview'] = self.data['Overview'].apply(lambda row: Dataset.__stem(row))
        if embedding == 'glove':
            self.__glove()
            self.data['Overview'] = self.data['Overview'].apply(lambda row: self.__embedding(row, embedding))
            self.data['Overview'] = self.data['Overview'].apply(lambda row: self.__pad_sequences(row))
        elif embedding == 'fasttext':
            self.__fasttext()
            self.data['Overview'] = self.data['Overview'].apply(lambda row: self.__embedding(row, embedding))
            self.data['Overview'] = self.data['Overview'].apply(lambda row: self.__pad_sequences(row))
        elif embedding == 'word2vec':
            self.__word2vec()
            self.data['Overview'] = self.data['Overview'].apply(lambda row: self.__embedding(row, embedding))
            self.data['Overview'] = self.data['Overview'].apply(lambda row: self.__pad_sequences(row))
        elif embedding == 'tfidf':
            self.__tfidf()
            self.data['Overview'] = self.__vectorize(embedding)

    def get_dataset(self):
        # Returns the data of the dataset class after the manipulation
        return self.data

    def train_test_split(self, test_size=0.2):
        # Splits the pre-processed data of the dataset into train set and test set.
        x_train, x_test, y_train, y_test = train_test_split(self.data[['Overview']], self.data[
            ['Western', 'Drama', 'Thriller', 'Mystery', 'Music', 'Romance', 'Action', 'Adventure', 'Foreign', 'Crime',
             'Documentary', 'Horror', 'Fantasy', 'History', 'Science Fiction', 'Family', 'TV Movie', 'Comedy',
             'Animation', 'War']], test_size=test_size)
        return x_train, x_test, y_train, y_test
