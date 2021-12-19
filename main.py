import pandas as pd
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self, punctuation=False, stop_words=False, stem=False, tokenization=False):
        self.data = pd.read_csv('data/tmdb_5000_movies.csv')[['genres', 'overview']]
        self.data.dropna(subset=['overview'], inplace=True)
        self.__clean()
        self.__preprocess(punctuation, stop_words, stem, tokenization)

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
        return ' '.join(new_text)

    @staticmethod
    def __stem(text):
        ps = PorterStemmer()
        new_test = []
        for word in word_tokenize(text):
            new_test.append(ps.stem(word))
        return ' '.join(new_test)

    @staticmethod
    def __tokenization(text):
        return word_tokenize(text)

    def __preprocess(self, punctuation, stop_words, stem, tokenization):
        if punctuation:
            self.data['Overview'] = self.data['Overview'].apply(lambda row: Dataset.__punctuation(row))
        if stop_words:
            self.data['Overview'] = self.data['Overview'].apply(lambda row: Dataset.__stop_words(row))
        if stem:
            self.data['Overview'] = self.data['Overview'].apply(lambda row: Dataset.__stem(row))
        if tokenization:
            self.data['Overview'] = self.data['Overview'].apply(lambda row: Dataset.__tokenization(row))

    def get_dataset(self):
        return self.data

    def train_test_split(self, test_size=0.2):
        x_train, x_test, y_train, y_test = train_test_split(self.data[['Overview']], self.data[
            ['Western', 'Drama', 'Thriller', 'Mystery', 'Music', 'Romance', 'Action', 'Adventure', 'Foreign', 'Crime',
             'Documentary', 'Horror', 'Fantasy', 'History', 'Science Fiction', 'Family', 'TV Movie', 'Comedy',
             'Animation', 'War']], test_size=test_size)
        return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    dataset = Dataset(punctuation=True, stop_words=True, tokenization=True)
    x_train, x_test, y_train, y_test = dataset.train_test_split()
