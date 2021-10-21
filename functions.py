import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 


def regularize(variables):
    max_values = np.amax(variables, axis=1)
    return variables / max_values[:, None]


def cost(theta, x, y):
    differences = sigmoid(x.dot(theta)) - y
    return np.sum(np.power(differences, 2))


def gradient(theta, x, y):
    g = sigmoid(x.dot(theta))
    return g * (1 - g) * (g - y)

def predict(theta, x):
    s = sigmoid(x.dot(theta))
    predictions = s
    predictions[s > .5] = 1
    predictions[s < .5] = 0
    return predictions

def cleanse_dataset(dataset, is_training=False):
    dataset['Embarked'] = dataset['Embarked'].replace(['S', 'C', 'Q'], [1, 2, 3])
    dataset['Sex'] = dataset['Sex'].replace(['male', 'female'], [0, 1])

    if(is_training):
        dataset = dataset[dataset['Age'].notna()]
    #dataset['AgeSquared'] = dataset['Age'] ** 2
    SibSpSquared = dataset['SibSp'].pow(2)
    ParchSquared = dataset['Parch'].pow(2)
    dataset['SibSpSquared'] = SibSpSquared
    dataset['ParchSquared'] = ParchSquared


    variables = dataset.drop(['Survived', 'PassengerId', 'Name', 'Cabin', 'Ticket', 'Embarked'], axis=1)
    variables = variables.replace(np.NaN, 0)

    labels = dataset['Survived']

    x = np.array(variables)
    y = np.array(labels)

    m = np.size(x, axis=0)
    
    x = np.hstack((np.ones((m, 1)), x))
    y = y.reshape((m, 1))

    return x, y


def decorate(title):
    print('\n', 3*'#', f' {title} ', 15*'#')