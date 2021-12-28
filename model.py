import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Flatten, MaxPooling1D, Dropout
from keras.layers.embeddings import Embedding
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, accuracy_score


class Model:
    def __init__(self, model_type, feature_dim, vocabulary_size=None, embedding_matrix=None):
        self.models = {
            'Western': None,
            'Drama': None,
            'Thriller': None,
            'Mystery': None,
            'Music': None,
            'Romance': None,
            'Action': None,
            'Adventure': None,
            'Foreign': None,
            'Crime': None,
            'Documentary': None,
            'Horror': None,
            'Fantasy': None,
            'History': None,
            'Science Fiction': None,
            'Family': None,
            'TV Movie': None,
            'Comedy': None,
            'Animation': None,
            'War': None
        }
        self.feature_dim = feature_dim
        self.model_type = model_type
        self.vocabulary_size = vocabulary_size
        self.embedding_matrix = embedding_matrix
        self.__init_model()

    def __init_model(self):
        if self.model_type == 'svm':
            model = SVC()
        elif self.model_type == 'linear':
            model = SGDClassifier()
        elif self.model_type == 'rnn':
            model = self.__create_rnn()
        elif self.model_type == 'model':
            if self.vocabulary_size is None or self.embedding_matrix is None:
                raise 'vocabulary_size and embedding_matrix params must have a value'
            model = self.__create_model()
        else:
            raise f'{self.model_type} does not exist'
        for key in self.models:
            self.models[key] = model

    def __create_rnn(self):
        model = Sequential()
        model.add(SimpleRNN(1000, input_shape=(self.feature_dim, 1)))
        model.add(Dense(1, activation='sigmoid'))
        opt = tf.keras.optimizers.Adam(learning_rate=0.03)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.summary()
        return model

    def __create_model(self):
        model = Sequential()  # initialize the model
        # embedding = Embedding(self.vocabulary_size, 100, weights=[self.embedding_matrix], input_length=1000,
        #                       trainable=False)
        # model.add(embedding)  # add an embedding layer
        # # model.add(GlobalAveragePooling1D())  # globalAveragePooling
        # # model.add(MaxPooling1D(pool_size=1)) # max pooling
        # model.add(Flatten())  # flattening into a 1d to pass it into dense layers
        # model.add(Dropout(0.4))
        # # model.add(Dense(128, activation='tanh'))
        # # model.add(Dropout(0.4))
        # # model.add(Dense(200, activation='tanh'))
        # # model.add(Dense(100, activation='tanh'))
        # model.add(Dense(50, activation='tanh'))
        # model.add(Dropout(0.4))
        # model.add(Dense(2, activation='softmax'))
        # adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.03)
        # model.compile(loss='sparse_categorical_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
        return model

    def train(self, train_x, train_y, test_x=None, test_y=None, epochs=30, batch_size=30):
        train_x = np.array(train_x.iloc[:, 0].values.tolist())
        if self.model_type == 'model':
            for key in self.models:
                self.models[key].fit(train_x, train_y[key].to_numpy(), epochs=epochs, batch_size=batch_size,
                    validation_data=(test_x, test_y))  # fit method to train our neural network with the passed in data
        elif self.model_type == 'rnn':
            for key in self.models:
                self.models[key].fit(train_x, train_y[key].to_numpy(), epochs=epochs, batch_size=batch_size)
        else:
            for key in self.models:
                self.models[key].fit(train_x, train_y[key].to_numpy())

    def get_model(self):  # helper function to return the model instance if needed
        return self.models

    def test(self, test_x, test_y):  # evaluation function
        res = {
            'model': self.model_type,
            'score': dict()
        }
        test_x = np.array(test_x.iloc[:, 0].values.tolist())
        if self.model_type == 'model':
            for key in self.models:
                score = self.models[key].evaluate(test_x, test_y[key].to_numpy())
                if key not in res['score']:
                    res['score'][key] = dict()
                res['score'][key]['loss'] = score[0]
                res['score'][key]['accuracy'] = score[1]
        else:
            for key in self.models:
                pred_y = self.models[key].predict(test_x)
                if key not in res['score']:
                    res['score'][key] = dict()
                res['score'][key]['loss'] = log_loss(test_y[key].to_numpy(), pred_y)
                res['score'][key]['accuracy'] = accuracy_score(test_y[key].to_numpy(), pred_y)
        return res
