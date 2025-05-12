# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

class MLPClassifier():
    def __init__(self, mlp_shape=[2,5,3,1], iteractions=30000, alpha=0.1):
        self.iteractions = iteractions
        self.learning_rate = alpha
        # Initializing the weights
        # self.__layer1_weights = np.random.rand(self.mlp_shape[0],self.mlp_shape[1])
        # self.__layer2_weights = np.random.rand(self.mlp_shape[1],self.mlp_shape[2])
        # self.__layer3_weights = np.random.rand(self.mlp_shape[2],self.mlp_shape[3])

        self.weights = []

        for i in range(len(mlp_shape)-1):
            self.weights.append(np.random.rand(mlp_shape[i], mlp_shape[i+1]))

    def __sigmoid(self, x):
        return np.tanh(x)

    def __deltaSigmoid(self, x):
        return 1.0-x**2

    def predict(self, inputs):
        # w1 = np.full((self.mlp_shape[1],1), self.__sigmoid(np.dot(inputs,self.__input_weights.T)))
        outputs = []
        outputs.append(self.__sigmoid(np.dot(inputs, self.weights[0])))
        for x in range(1, len(self.weights)):
            outputs.append(self.__sigmoid(np.dot(outputs[x-1], self.weights[x])))
        return outputs

    def fit(self, traning_data, output_data , learning_rate=0.1):
        for step in range(self.iteractions):
            for data_x, data_y in zip(traning_data, output_data):

                out_from_layers = self.predict(data_x)

                # print("out", out_from_layers)

                error_layer = []
                delta_error = []

                error_layer.append(data_y - out_from_layers[len(out_from_layers)-1])
                print("ERROR", error_layer, error_layer*0)
                # delta_error.append(error_layer[0] * self.__deltaSigmoid(out_from_layers[len(out_from_layers)-1]))
                # #Do final pro inicial
                # print(error_layer, "delta", delta_error)
                #
                # for i in range(len(out_from_layers)-1):
                #     error_layer.append(np.dot(delta_error[i], self.weights[len(out_from_layers)-1-i].T))
                #     delta_error.append(error_layer[i+1] * self.__deltaSigmoid(out_from_layers[len(out_from_layers)-1-i]))
                #
                #
                # for i in list(reversed(range(len(self.weights)))):
                #     # print(i, len(self.weights), len(self.weights)-1-i)
                #     # print(delta_error[len(self.weights)-1-i], "d", data_y)
                #     self.weights[i] += learning_rate * delta_error[len(self.weights)-1-i] * data_y

                # error_layer3 = (data_y - out_from_layer3)
                # delta_error3 = error_layer3 * self.__deltaSigmoid(out_from_layer3)
                #
                # self.__layer3_weights += learning_rate * delta_error3 * data_y
                #
                # error_layer2 = np.dot(delta_error3, self.__layer3_weights.T)
                # delta_error2 = error_layer2 * self.__deltaSigmoid(out_from_layer2)
                #
                # self.__layer2_weights += learning_rate * delta_error2 * data_y
                #
                # error_layer1 = np.dot(delta_error2, self.__layer2_weights.T)
                # delta_error1 = error_layer1 * self.__deltaSigmoid(out_from_layer1)
                #
                # self.__layer1_weights += learning_rate * delta_error1 * data_y
