# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 13:28:02 2021

@author: User
"""


#pure python

import random
import time

time_start = time.time()

l = [random.randrange(100, 999) for i in range(100000000)]

squared = [x**2 for x in l]
sqrt = [x**0.5 for x in l]
mul = [x * y for x, y in zip(squared, sqrt)]
div = [x / y for x, y in zip(squared, sqrt)]
int_div = [x // y for x, y in zip(squared, sqrt)]

time_end = time.time()

time_passed = time_end - time_start
#print(f'TOTAL TIME = {(time_end - time_start).seconds} seconds')

print(time_passed)




# numpy

import numpy as np
from time import time
from datetime import datetime

start_time = datetime.now()

# Let's take the randomness out of random numbers (for reproducibility)
np.random.seed(0)

size = 4096
A, B = np.random.random((size, size)), np.random.random((size, size))
C, D = np.random.random((size * 128,)), np.random.random((size * 128,))
E = np.random.random((int(size / 2), int(size / 4)))
F = np.random.random((int(size / 2), int(size / 2)))
F = np.dot(F, F.T)
G = np.random.random((int(size / 2), int(size / 2)))

# Matrix multiplication
N = 20
t = time()
for i in range(N):
    np.dot(A, B)
delta = time() - t
print('Dotted two %dx%d matrices in %0.2f s.' % (size, size, delta / N))
del A, B

# Vector multiplication
N = 5000
t = time()
for i in range(N):
    np.dot(C, D)
delta = time() - t
print('Dotted two vectors of length %d in %0.2f ms.' % (size * 128, 1e3 * delta / N))
del C, D

# Singular Value Decomposition (SVD)
N = 3
t = time()
for i in range(N):
    np.linalg.svd(E, full_matrices = False)
delta = time() - t
print("SVD of a %dx%d matrix in %0.2f s." % (size / 2, size / 4, delta / N))
del E

# Cholesky Decomposition
N = 3
t = time()
for i in range(N):
    np.linalg.cholesky(F)
delta = time() - t
print("Cholesky decomposition of a %dx%d matrix in %0.2f s." % (size / 2, size / 2, delta / N))

# Eigendecomposition
t = time()
for i in range(N):
    np.linalg.eig(G)
delta = time() - t
print("Eigendecomposition of a %dx%d matrix in %0.2f s." % (size / 2, size / 2, delta / N))

print('')
end_time = datetime.now()
print(f'TOTAL TIME = {(end_time - start_time).seconds} seconds')




# pandas
import numpy as np
import pandas as pd
from datetime import datetime

time_start = datetime.now()

df = pd.DataFrame()
df['X'] = np.random.randint(low=100, high=999, size=100000000)
df['X_squared'] = df['X'].apply(lambda x: x**2)
df['X_sqrt'] = df['X'].apply(lambda x: x**0.5)
df['Mul'] = df['X_squared'] * df['X_sqrt']
df['Div'] = df['X_squared'] / df['X_sqrt']
df['Int_div'] = df['X_squared'] // df['X_sqrt']

time_end = datetime.now()
print(f'Total time = {(time_end - time_start).seconds} seconds')



# sk learn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

time_start = datetime.now()

# Dataset
iris = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')
time_load = datetime.now()
print(f'Dataset loaded, runtime = {(time_load - time_start).seconds} seconds')

# Train/Test split
X = iris.drop('species', axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
time_split = datetime.now()
print(f'Train/test split, runtime = {(time_split - time_start).seconds} seconds')

# Hyperparameter tuning
model = DecisionTreeClassifier()
params = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [1, 5, 10, 50, 100, 250, 500, 1000],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 3, 4, 5],
    'max_features': ['auto', 'sqrt', 'log2']
}
clf = GridSearchCV(
    estimator=model, 
    param_grid=params, 
    cv=5
)
clf.fit(X_train, y_train)
time_optim = datetime.now()
print(f'Hyperparameter optimization, runtime = {(time_optim - time_start).seconds} seconds')

best_model = DecisionTreeClassifier(**clf.best_params_)
best_model.fit(X_train, y_train)

time_end = datetime.now()
print()
print(f'TOTAL RUNTIME = {(time_end - time_start).seconds} seconds')