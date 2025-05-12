# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math

class NaiveBayes():
    def __init__(self):
        self.probabilities = []

    def help(self):
        print('\n')
        print('*'*50)
        print("Code to implement Naive Bayes Algorithm")
        print('*'*50)
        print('\n')

    def predict(self, data, value, targets_name='target'):

        self.mean_classes = []
        self.variance_classes = []
        self.classes = max(data[targets_name]) + 1
        dim = len(data.keys()) - 1

        for x in range(self.classes):
            self.mean_classes.append(np.mean(data[data[targets_name] == x].iloc[:,0:dim].values))
            self.variance_classes.append(np.std(data[data[targets_name] == x].iloc[:,0:dim].values))

        probabilities = []
        self.probabilities = []
        for x in range(self.classes):
            exponent = (-(np.power(value-self.mean_classes[x],2)/(2*self.variance_classes[x]*self.variance_classes[x])))
            probabilities.append((np.power((1 / (np.sqrt(2*math.pi)*self.variance_classes[x])),exponent)[0]))

        c_aux = []
        for x in probabilities:
            prob = x/sum(probabilities)
            self.probabilities.append(prob)
            a = 1
            for i in prob:
                a *= i
            c_aux.append(a)

        return c_aux.index(min(c_aux)), self.probabilities[c_aux.index(min(c_aux))]
