# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

class MLPClassifier():
    def __init__(self, mlp_shape=[2,5,3,1], iteractions=30000, alpha=0.1):
        self.iteractions = iteractions
        self.learning_rate = alpha
        self.mlp_shape = mlp_shape
        # Initializing the weights
        self.__layer1_weights = np.random.rand(self.mlp_shape[0],self.mlp_shape[1])
        self.__layer2_weights = np.random.rand(self.mlp_shape[1],self.mlp_shape[2])
        self.__layer3_weights = np.random.rand(self.mlp_shape[2],self.mlp_shape[3])




    def help(self):
        print('\n')
        print('*'*50)
        print("Code to implement MLP Classifier Algorithm")
        print('*'*50)
        print('\n')

    def __sigmoid(self, x):
        return np.tanh(x)

    def __deltaSigmoid(self, x):
        return 1.0-x**2

    def predict(self, inputs):
        # w1 = np.full((self.mlp_shape[1],1), self.__sigmoid(np.dot(inputs,self.__input_weights.T)))
        w1 = self.__sigmoid(np.dot(inputs, self.__layer1_weights))
        w2 = self.__sigmoid(np.dot(w1,self.__layer2_weights))
        w3 = self.__sigmoid(np.dot(w2,self.__layer3_weights))
        return w1,w2,w3

    def fit(self, traning_data, output_data , learning_rate=0.1):
        for step in range(self.iteractions):
            for data_x, data_y in zip(traning_data, output_data):

                # data_x = np.append(data_x, [[1]], axis=1)

                out_from_layer1, out_from_layer2, out_from_layer3 = self.predict(data_x)

                error_layer3 = (data_y - out_from_layer3)
                delta_error3 = error_layer3 * self.__deltaSigmoid(out_from_layer3)


                error_layer2 = np.dot(delta_error3, self.__layer3_weights.T)
                delta_error2 = error_layer2 * self.__deltaSigmoid(out_from_layer2)


                error_layer1 = np.dot(delta_error2, self.__layer2_weights.T)
                delta_error1 = error_layer1 * self.__deltaSigmoid(out_from_layer1)

                self.__layer3_weights += learning_rate * delta_error3 * data_y
                self.__layer2_weights += learning_rate * delta_error2 * data_y
                self.__layer1_weights += learning_rate * delta_error1 * data_y
