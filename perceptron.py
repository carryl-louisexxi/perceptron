import csv
import random
from decimal import Decimal

def adder_function(weights, dataset, bias_term): # adder function 

    activation = 0

    for w, x in zip(weights, dataset):

        activation += w * x 

    activation += bias_term

    return activation


def perceptron(dataset, y_labels, max_iterations):

    weights = [0 for _ in range(len(dataset[0]))]

    bias_term = 1 

    current_dataset = [*dataset]

    current_labels = [*y_labels]

    for iteration in range(max_iterations):

        joined_x_and_y = list(zip(current_dataset, current_labels))

        random.shuffle(joined_x_and_y)

        current_dataset, current_labels = zip(*joined_x_and_y)

        for i, item in enumerate(current_dataset): 

            activation = adder_function(weights, item, bias_term)

            if activation * current_labels[i] <= 0: # activation function
                
                for k, weight in enumerate(weights):

                    weights[k] = weights[k] + current_labels[i] * item[k]

                    bias_term += current_labels[i]

    return weights, bias_term

file_path = './sonar.all-data'

LABELS_CONSTANTS = {'R': -1, 'M': 1}

with open(file_path, 'r') as file:

    dataset = []

    y_labels = []

    csv_reader = csv.reader(file)

    for row in csv_reader:

        if row:

            dataset.append([Decimal(item) for item in row[:-1]])

            y_labels.append(LABELS_CONSTANTS[row[len(row) - 1]])



weights, bias_term = perceptron(dataset, y_labels, 2)

print(weights)