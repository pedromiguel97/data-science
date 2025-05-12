# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class LinearDiscriminantAnalysis():
    def __init__(self):
        self.lda_data = []
        self.eig_vecs = []

    def help(self):
        print('\n')
        print('*'*50)
        print("Code to implement Linear Discriminant Analysis Algorithm")
        print('*'*50)
        print('\n')

    def fit(self, data, n_components, plot=False):

        #Computing the d-dimensional mean vectors
        ld = int(data['target'].max()) + 1  #getting the number of the classes
        dim = len(data.keys())-1 #getting the length of features
        mean_vectors = []
        for i in range(ld):
            mean_vectors.append(np.mean(data[data['target']==i].iloc[:,:dim]).values)
        print(mean_vectors)

        #Computing the Scatter Matrices
        scatter_matrices = np.zeros((dim,dim))
        for i,j in zip(range(dim-1), mean_vectors):
            class_sc_mat = np.zeros((dim,dim))
            for row in data[data['target']==i].iloc[:,:dim].values:
                row, j = row.reshape(dim,1), j.reshape(dim,1)
                class_sc_mat += (row-j).dot((row-j).T)
            scatter_matrices += class_sc_mat
        print(scatter_matrices)

        # Between-class scatter matrix
        overall_mean = np.mean(data.iloc[:,:dim].values, axis=0)
        S_B = np.zeros((dim,dim))
        for i,mean_vec in enumerate(mean_vectors):
            n = data[data['target']==i].values.shape[0]
            mean_vec = mean_vec.reshape(dim,1)
            overall_mean = overall_mean.reshape(dim,1)
            S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
        print(S_B)

        #Getting the eigvectors and eigvalues
        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(scatter_matrices).dot(S_B))

        # Make a list of eigenvalue and eigenvector
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

        # Sort the (eigenvalue, eigenvector) from high to low
        eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)

        W = []
        for x in range(n_components):
            W.append(eig_pairs[x][1].reshape(dim,1))

        W = np.array(W).T[0].real

        # Getting data back
        X_lda = data.iloc[:,:dim].values.dot(W.real)

        self.lda_data = X_lda
        self.eig_vecs = W

        # Plotting for 1,2 or 3 dimensions
        if plot:
            colors = ['red','blue','yellow','black','white']
            if n_components == 1:
                for x in range(ld):
                    min_i = min(data[data['target']==x].index)
                    max_i = max(data[data['target']==x].index)
                    plt.scatter(X_lda[min_i:max_i+1],np.zeros((50)),c=colors[x])

            elif (n_components == 2):
                print(data[0:2].values)
                print(X_lda[0:2])
                for x in range(ld):
                    min_i = min(data[data['target']==x].index)
                    max_i = max(data[data['target']==x].index)
                    plt.scatter(X_lda[min_i:max_i+1,0],X_lda[min_i:max_i+1,1],c=colors[x])
                    # print(data[min_i:max_i+1].iloc[:,0].values)
                    # plt.scatter(data[min_i:max_i+1].iloc[:,2].values, data[min_i:max_i+1].iloc[:,3].values,c=colors[x])
                    # plt.xlabel('Petalas Comprimento')
                    # plt.ylabel('Petalas Largura')

            elif n_components == 3:
                # print(X_lda)
                fig = plt.figure()
                ax = Axes3D(fig)
                for x in range(ld):
                    min_i = min(data[data['target']==x].index)
                    max_i = max(data[data['target']==x].index)
                    ax.scatter(X_lda[min_i:max_i+1,0],X_lda[min_i:max_i+1,1],X_lda[min_i:max_i+1,2],c=colors[x])

            else:
                print('Impossible to plot this dimension')
                return []

            plt.show()
    def plot(self):


        plt.show()
