from model import Model
from data_prepro import Dataset
import numpy as np
from pprint import pprint

dataset = Dataset(punctuation=True, stop_words=True, embedding=True)  # Initialization the Dataset class
x_train, x_test, y_train, y_test = dataset.train_test_split(0.3)  # Splitting the dataset into train, test set

data_list = np.array(x_train.iloc[:, 0].values.tolist())  # Getting all the overviews as lists of word embeddings
# into a numpy array in order to have access to the dimensions and reshape them
print(data_list.shape)  # 3339: overviews, 103: overviews words length, 100: word embeddings dimensions

# 'svm-linear': Support Vector Machine (linear kernel)
# 'svm-rbf'   : Support Vector Machine (rbf kernel)
# 'linear': Stochastic Gradient Descent Classifier
# 'tree'  : Decision Tree Classifier
# 'forest': Random Forest
# 'mlp'   : Multi Layer Perceptron (used activation tanh)
# 'rnn'   : Recurrent Neural Network
# 'lstm'  : Long Short Term Memory
# 'bilstm': Bidirectional Long Short Term Memory
model = Model('lstm', input_shape=(data_list.shape[1], data_list.shape[2]))  # Initialization of the Model, choose one
# from the above models. The input_shape is a mandatory parameter, and actually it represents the length of words in
# the overviews and the length of the word embeddings
model.train(x_train, y_train, epochs=2)  # Training the model with our train data, epochs is the same in the training
# of each model fo each genre
res = model.test(x_test, y_test)  # Evaluate the trained model using the test set
print(res)  # Printing the results
