# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
import random

class Perceptron():
    def __init__(self, inputs_number, iteractions=100, alpha=0.2):
        self.weights = np.zeros(inputs_number + 1)
        self.iteractions = iteractions
        self.learning_rate = alpha

    def help(self):
        print('\n')
        print('*'*50)
        print("Code to implement Perceptron Algorithm")
        print('*'*50)
        print('\n')

    def predict(self, inputs):
        # Multiplicação das entradas pelos pesos + o bias
        sum = np.dot(inputs, self.weights[1:]) + self.weights[0]
        # Comparação para ativação
        if sum > 0:
          return 1
        else:
          return 0

    def fit(self, X, Y):
        for i in range(self.iteractions):
            # Começa uma nova iteração com contagem de erro = zero
            count_error = 0
            for inputs, label in zip(X, Y):
                #Calcula valor predito
                prediction = self.predict(inputs)
                # Verifica erro, se erro é diferente de zero, atualiza os pesos
                error = (label - prediction)
                if error != 0:
                    count_error += 1
                    self.weights[1:] = self.weights[1:] + (self.learning_rate * error * inputs)
                    self.weights[0] = self.weights[0] + (self.learning_rate * error)
            print("Iteraction: {}".format(i))
            # Finaliza as iterações se todos os dados não possuem erro
            if count_error == 0:
                print("FINESHED")
                break
