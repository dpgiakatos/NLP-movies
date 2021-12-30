import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, Flatten, Input, Reshape, Dropout, TimeDistributed
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score


class Model:
    def __init__(self, model_type, input_shape):
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
        self.input_shape = input_shape
        self.model_type = model_type
        self.__init_model()

    def __init_model(self):
        # Choosing the model and creating a model for each Genre
        if self.model_type == 'svm-linear':
            model = LinearSVC(random_state=42)
        elif self.model_type == 'svm-rbf':
            model = SVC(kernel='rbf', random_state=42)
        elif self.model_type == 'linear':
            model = SGDClassifier(random_state=42)
        elif self.model_type == 'tree':
            model = DecisionTreeClassifier(random_state=42)
        elif self.model_type == 'forest':
            model = RandomForestClassifier(n_estimators=10, random_state=42)
        elif self.model_type == 'mlp':
            model = self.__create_mlp()
        elif self.model_type == 'rnn':
            model = self.__create_model(self.model_type)
        elif self.model_type == 'lstm':
            model = self.__create_model(self.model_type)
        elif self.model_type == 'bilstm':
            model = self.__create_model(self.model_type)
        else:
            raise f'{self.model_type} does not exist'
        for key in self.models:
            self.models[key] = model

    def __create_mlp(self):
        # Secret function to create the MLP with 1 input, 3 hidden and 1 output layers
        model = Sequential()
        model.add(Input(shape=self.input_shape))
        model.add(Flatten())
        model.add(Dense(100, activation='tanh'))
        model.add(Dropout(0.4))
        model.add(Dense(100, activation='tanh'))
        model.add(Dropout(0.4))
        model.add(Dense(100, activation='tanh'))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        opt = tf.keras.optimizers.Adam(learning_rate=0.03)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.summary()
        return model

    def __create_model(self, model_type):
        # Secret function to create the RNN and LSTM networks
        model = Sequential()
        model.add(Input(shape=self.input_shape))
        model.add(Flatten())
        # The RNN and LSTM keras layers take as input a 3D tensor with shape [batch, timesteps, feature]. Reshaping
        # the 2D tensor produced by the "Flatten()" function in order to add timesteps dim so the output will match the
        # input of RNN and LSTM layers
        model.add(Reshape((1, self.input_shape[0] * self.input_shape[1]),
                          input_shape=(self.input_shape[0] * self.input_shape[1],)))
        if model_type == 'rnn':
            model.add(SimpleRNN(1000))
            model.add(Dense(1, activation='sigmoid'))
        elif model_type == 'lstm':
            model.add(LSTM(1000))
            model.add(Dense(1, activation='sigmoid'))
        elif model_type == 'bilstm':
            model.add(LSTM(1000, return_sequences=True))
            model.add(TimeDistributed(Dense(1, activation='sigmoid')))
        opt = tf.keras.optimizers.Adam(learning_rate=0.03)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.summary()
        return model

    def train(self, train_x, train_y, epochs=30, batch_size=30):
        # Training the models on the train set
        train_x = np.array(train_x.iloc[:, 0].values.tolist())
        if self.model_type in 'mlp|rnn|lstm|bilstm':
            for key in self.models:
                self.models[key].fit(train_x, train_y[key].to_numpy(), epochs=epochs, batch_size=batch_size)
        else:
            # Reshaping manually the input into 2D because there is no Flatten() function
            train_x = train_x.reshape(train_x.shape[0], train_x.shape[1] * train_x.shape[2])
            for key in self.models:
                self.models[key].fit(train_x, train_y[key].to_numpy())

    def get_model(self):
        # Helper function to return the model instance if needed
        return self.models

    def test(self, test_x, test_y):
        # Evaluation function, which returns a dictionary as shown bellow:
        # {
        #   (genre): {
        #       'loss': (value),
        #       'accuracy': (value)
        #   }
        # }
        res = {
            'model': self.model_type,
            'score': dict()
        }
        test_x = np.array(test_x.iloc[:, 0].values.tolist())
        if self.model_type in 'mlp|rnn|lstm|bilstm':
            for key in self.models:
                score = self.models[key].evaluate(test_x, test_y[key].to_numpy())
                if key not in res['score']:
                    res['score'][key] = dict()
                res['score'][key]['loss'] = score[0]
                res['score'][key]['accuracy'] = score[1]
        else:
            test_x = test_x.reshape(test_x.shape[0], test_x.shape[1] * test_x.shape[2])
            for key in self.models:
                pred_y = self.models[key].predict(test_x)
                if key not in res['score']:
                    res['score'][key] = dict()
                res['score'][key]['loss'] = log_loss(test_y[key].to_numpy(), pred_y)
                res['score'][key]['accuracy'] = accuracy_score(test_y[key].to_numpy(), pred_y)
        return res
