import numpy as np
import pandas as pd

import functions as func
from model import Logistic


dataset = pd.read_csv('data/train.csv')

#Splits dataset
training_set = dataset.sample(frac = .75)
cross_validation_set = dataset.drop(training_set.index)

test_set = pd.read_csv('data/test.csv')
test_set['Survived'] = np.NaN
x, y = func.cleanse_dataset(training_set, is_training=True)
x_cv, y_cv = func.cleanse_dataset(cross_validation_set)
x_test, y_test = func.cleanse_dataset(test_set)

x = func.regularize(x)

learning_rate = 10
max_iter = 500000
convergence_crit = 10e-6
model = Logistic(learning_rate, max_iter, convergence_crit)
model.train(x, y)
model.test(x, y)
model.test(x_cv, y_cv)
model.plot()

predictions = model.predict(x_test)


#Prepares and exports submission
m_test = np.size(x_test, axis=0)
submission = pd.DataFrame({'PassengerId': test_set['PassengerId']})
submission['Survived'] = pd.DataFrame(predictions)
submission.to_csv('my_submission.csv')