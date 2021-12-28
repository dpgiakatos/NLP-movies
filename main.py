from model import Model
from data_prepro import Dataset
import numpy as np

dataset = Dataset(punctuation=True, stop_words=True, embedding=True)
data = dataset.get_dataset()
# print(data.columns)
# print(data['Overview'])
# print(data.isnull().any())
# labels = array([1,1,1,1,1,0,0,0,0,0]) # define class labels
# print(labels)

x_train, x_test, y_train, y_test = dataset.train_test_split(0.3)

# data_list = np.array(x_train.iloc[:, 0].values.tolist())

model = Model('rnn', feature_dim=100)  # initialize the model
model.train(x_train, y_train, x_test, y_test, epochs=2)  # train the model with our data
res = model.test(x_test, y_test)  # evaluate the trained model using the test set
print(res)
