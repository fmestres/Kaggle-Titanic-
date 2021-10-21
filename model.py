import functions as func

import numpy as np
import matplotlib.pyplot as plt

class Logistic:    
    #convergence_crit is the minimum cost difference between two iters in a sequence. =0 means it will iterate until it reaches max_iter
    def __init__(self, learning_rate=.1, max_iter=10e+5, convergence_crit=0):
        self.theta = None
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.convergence_crit = convergence_crit
        
        self.iter_history = None
        self.cost_history = None

    def train(self, x, y):
        #Gets dimensions of x matrix
        n = np.size(x, axis=1)
        m = np.size(x, axis=0)

        #Initializes theta to match x dimension
        theta = np.random.rand(n, 1)

        learning_rate = self.learning_rate
        max_iter = self.max_iter
        convergence_crit = self.convergence_crit

        #init values
        cost = 0
        iterations = 0
        iter_history = []
        cost_history = []
        while(iterations < max_iter):
            theta -= learning_rate * 1/m * x.T.dot(func.gradient(theta, x, y))
            iterations += 1
            prev_cost = cost
            cost = func.cost(theta, x, y)
            iter_history.append(iterations)
            cost_history.append(cost)
            if(abs(cost - prev_cost) < convergence_crit):
                break
        self.theta = theta
        self.iter_history = np.array(iter_history)
        self.cost_history = np.array(cost_history)
        
        func.decorate('Training')
        print(f'Iterations: {iterations}\nCost: {cost}')
    
    def test(self, x_test, y_test):
        predictions = self.predict(x_test)
        m_test = np.size(x_test, axis=0)
        error = predictions - y_test
        accuracy = (error == 0).sum() / m_test * 100

        func.decorate('Test')
        print(f'Accuracy: {accuracy}%')

    def predict(self, x_pred):
        return func.predict(self.theta, x_pred)

    def plot(self):
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.iter_history, self.cost_history)
        ax.set_title('Training')
        ax.set_xlabel('# of Iterations')
        ax.set_ylabel('Cost')
        plt.show()