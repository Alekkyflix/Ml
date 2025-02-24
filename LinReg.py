import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from matplotlib.animation import FuncAnimation

dataset = 'data_for_lr.csv'
data = pd.read_csv(dataset)
data

# Drop the missing values
data = data.dropna()

# training dataset and labels
train_input = np.array(data.x[0:500]).reshape(500, 1)
train_output = np.array(data.y[0:500]).reshape(500, 1)

# valid dataset and labels
test_input = np.array(data.x[500:700]).reshape(199, 1)
test_output = np.array(data.y[500:700]).reshape(199, 1)

class LinearRegression:
    def __init__(self):
        self.parameters = {}

    def forward_propagation(self, train_input):
        m = self.parameters['m']
        c = self.parameters['c']
        predictions = np.multiply(m, train_input) + c
        return predictions
    
    def cost_function(self, predictions, train_input):
        cost = np.mean((train_input - predictions)**2)
        return cost
    
    def backward_propagation(self, train_input, train_output, predictions):
        derivatives = {}
        df = (predictions - train_output)
        dm = 2 * np.mean(np.multiply(train_input,df)) # dm= 2/n * mean of (predictions-actual) * input
        dc = 2* np.meannp.mean(df) # dc = 2/n * mean of (predictions-actual)
        derivatives['dm'] = dm
        derivatives['dc'] = dc
        return derivatives
    
    def update_parameters(self, derivatives, learning_rate):
        self.parameters['m'] = self.parameters['m'] - learning_rate * derivatives['dm']
        self.parameters['c'] = self.parameters['c'] - learning_rate * derivatives['dc']

    def train(self, train_input, train_output, learning_rate, iterations):
        #init random parameters
        self.parameters['m'] = np.random.uniform(0,1) * - 1
        self.parameters['c'] = np.random.uniform(0,1) * - 1
        
        #init loss
        self.loss = []

        # Initialize figure and axis for animation
        fig, ax = plt.subplots()
        x_vals = np.linspace(min(train_input), max(train_input), 100)
        line, = ax.plot(x_vals, x_vals, self.parameters['m'] * x_vals + self.parameters['c'], color = 'red', label = 'Regression line')
        ax.scatter(train_input, train_output, marker = 'o', color = 'blue', label = 'Training data')

        # Set y-axis limits to exclude negative values 
        ax.set_ylim(0, max(train_output) + 1)

        def update(frame):
            #forward propagation
            predictions = self.forward_propagation(train_input)

            #cost function
            cost = self.cost_function(predictions, train_output)

            #back propagation
            derivatives = self.backward_propagation(train_input, train_output, predictions)

            #update parameters
            self.update_parameters(derivatives, learning_rate)

            #update the regression line
            line.set_ydata(self.parameters['m'] * x_vals + self.parameters['c'])

            #apeend loss and print
            self.loss.append(cost)

            #option 2
            print(f'Iteration: {frame}, Loss: {cost:.4f}')

            #option 1
            print("iteration = {}, loss = {}".format(frame + 1, cost))

            return line
        #create animation
        ani = FuncAnimation(fig, update, frames = iterations, interval = 200, blit = True)

        #saving the animation
        ani.save ('linear_regression_A.gif', writer = 'ffmpeg')

        plt.xlabel('Input')
        plt.ylabel('Output')
        plt.title('Linear Regression')
        plt.legend()
        plt.show()

        return self.parameters, self.loss