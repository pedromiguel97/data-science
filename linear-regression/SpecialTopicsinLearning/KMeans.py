# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import random
import math
import numpy as np

class KMeans():
    def __init__(self):
        self.centroids = []

    def help(self):
        print('\n')
        print('*'*50)
        print("Code to implement K-Means Algorithm")
        print('*'*50)
        print('\n')

    def fit(self, data, k_number=0, centroids=[]):
        # Checking the k_number param
        if k_number == 0:
            print("Please, choose the k number!")
            return []
        k_centroids = []
        # Getting k initial centroids
        if centroids == []:
            k_centroids = random.sample(data, k=k_number)

        data_c = data
        dim = len(data[0])

        k_centroids_old = np.zeros([k_number,len(data[0])])
        iter = 0


        while not np.allclose(k_centroids, k_centroids_old):
            # Getting means of each centroid
            k_centroids_means = [0] * len(k_centroids)
            k_centroids_count_lines = [0] * len(k_centroids)
            cluster_names = []
            k_centroids_old = k_centroids

            for x in data_c:
                for centroid,i in zip(k_centroids,range(len(k_centroids))):
                    # k_centroids_means[i] = k_centroids_means[i] + self.euclidianDistance(centroid, x)
                    k_centroids_means[i] = self.euclidianDistance(centroid, x)

                cluster_names.append((k_centroids_means.index(min(k_centroids_means))))
                k_centroids_count_lines[(k_centroids_means.index(min(k_centroids_means)))] += 1

            k_centroids_aux = [[0] * dim for x in range(len(k_centroids))]

            for i,j in zip(data_c,cluster_names):
                for x in range(dim):
                    k_centroids_aux[j][x] = k_centroids_aux[j][x] + i[x]

            for x in range(len(k_centroids_aux)):
                for y in range(dim):
                    k_centroids_aux[x][y] = round(k_centroids_aux[x][y]/k_centroids_count_lines[x],2)

            print('Iteration: ' + str(iter))

            if len(data[0]) == 2:
                plt.scatter([x[0] for x in data_c],[x[1] for x in data_c])
                for x in range(len(k_centroids)):
                    plt.scatter(k_centroids[x][0],k_centroids[x][1],c='r')
                plt.show()

            k_centroids = k_centroids_aux
            iter+=1

            targets = []
            for x in data_c:
                aux = []
                for y in k_centroids:
                    aux.append(self.euclidianDistance(x,y))
                targets.append((aux.index(min(aux))))

        self.targets = targets
        self.centroids = k_centroids


    def euclidianDistance(self, d0, d1):
        return round(math.sqrt(sum([(a - b) ** 2 for a, b in zip(d0, d1)])),2)
