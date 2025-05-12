import numpy as np
import matplotlib.pyplot as plt
import SpecialTopicsinLearning.Matrix as mt

class LeastSquares():
    def __init__(self):
        super(LeastSquares, self).__init__()

    def help(self):
        print('\n')
        print('*'*50)
        print("Code to implement Least Squares Algorithm")
        print('*'*50)
        print('\n')
        print("\t - Enter X,Y (array) for data, W for weight, bias and use fit(X,Y,W,bias) function")
        print("\t - Enter X (array) data and use predict(X) function\n")

    def fit(self, X, Y, W=[], bias=0):

        #Checking weight
        if len(W) == 0:
            W = np.ones([len(np.array(X).T),1])
        else:
            W = np.array(W).T

        if bias != 0:
            self.X = np.append(np.ones([len(np.array(X).T),1])*bias, np.array(X).T, axis=1)
        else:
            self.X = np.array(X).T

        self.Y = np.array(Y).T
        #transposing X ...
        self.Xt = (self.X).T
        #doing Xt * W * X
        XtX = (self.Xt).dot(W*self.X)

        #doing Xt * W * y
        XtY = (self.Xt).dot(W*self.Y)

        #Inverting (Xt * W * X)
        invM = np.array(mt.getMatrixInverse(XtX.tolist()))

        #doing (XtX)-1 * XtY
        self.beta = invM.dot(XtY)

    def predict(self, X):
        #Predicting values
        Xt = np.array(X)
        #doing Xt * beta
        y = Xt.dot(self.beta)
        print(y)
        return y

    def plot(self):

        if self.beta.size != 0:
            fig = plt.figure()
            X = np.array([self.X[:,1]]).T
            #Plot of dataset
            plt.plot(X,self.Y,'ro')
            # Plot of regression
            X = np.array([list(range(int(min(X.T[0])) ,int(max(X.T[0])), 1))])
            Xmod = np.append(np.ones([len(np.array(X).T),1])*1, np.array(X).T, axis=1)
            y = list()
            for x in Xmod:
                y.append(x[0]*self.beta[0][0] + x[1]*self.beta[1][0])
            # plt.plot(X[0],y)

            plt.show()
