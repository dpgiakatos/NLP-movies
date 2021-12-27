import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, MaxPooling1D, Dropout
from keras.layers.embeddings import Embedding


class Model:
    def __init__(self):
        nn = Sequential()  # initialize the model
        self.model = nn

    def create_model(self, vocabulary_size, embedding_matrix):
        self.model = Sequential()
        embedding = Embedding(vocabulary_size, 100, weights=[embedding_matrix], input_length=1000, trainable=False)
        self.model.add(embedding)  # add an embedding layer
        # self.model.add(GlobalAveragePooling1D())  # globalAveragePooling
        # self.model.add(MaxPooling1D(pool_size=1)) # max pooling
        self.model.add(Flatten())  # flattening into a 1d to pass it into dense layers
        self.model.add(Dropout(0.4))
        # self.model.add(Dense(128, activation='tanh'))
        # self.model.add(Dropout(0.4))
        # self.model.add(Dense(200, activation='tanh'))
        # self.model.add(Dense(100, activation='tanh'))
        self.model.add(Dense(50, activation='tanh'))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(2, activation='softmax'))
        adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.03)
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=adam_optimizer,metrics=['accuracy'])

    def train(self, train_x, train_y, test_x, test_y, epochs=30, batch_size=30):
        # self.model_conv.summary()
        self.model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_data=(
        test_x, test_y))  # fit method to train our neural network with the passed in data

    def get_model(self):  # helper function to return the model instance if needed
        return self.model

    def test(self, test_x, test_y):  # evaluation function
        score = self.model.evaluate(test_x, test_y)
        print("Loss of the Model: ", score[0])
        print("Accuracy of the Model: ", score[1])
        return score[1]

