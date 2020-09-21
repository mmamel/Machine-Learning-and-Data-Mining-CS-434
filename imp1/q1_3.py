import numpy as np 
from math import pow
import sys

np.set_printoptions(suppress = True)

#get X 
train = np.genfromtxt(sys.argv[1], delimiter=',');
train=np.delete(train, 13, axis=1)

#get dims
train_row = train.shape[0]
train_col = train.shape[1]

#get Y
target = np.genfromtxt(sys.argv[1], delimiter=',');
target=target[:, 13]

#calculate weight vector
xtx = np.dot(train.T, train)
inv = np.linalg.inv(xtx)
invxt = np.dot(inv, train.T)
w = np.dot(invxt, target)
# w = np.dot(np.dot(np.linalg.inv(np.dot(train.T, train)), train.T), target)
# print(w.reshape(train_col, 1))

print("Weight Vector = ")
print(w)

#find ASE for training
train_sum=0
for x in range(train_row):
    train_sum+= pow(target[x] - np.dot(w.T, train[x]),2)
train_sum/=train_row

print("training ASE = ", train_sum)

#get data from test
test = np.genfromtxt(sys.argv[2], delimiter=',')
test=np.delete(test, 13, axis=1)

#get Y
test_target = np.genfromtxt(sys.argv[2], delimiter=',');
test_target=test_target[:, 13]

#get dim
test_row = test.shape[0]

test_sum=0
for x in range(test_row):
    test_sum+= pow(test_target[x] - np.dot(w.T, test[x]),2)
test_sum/=test_row
print("testing ASE = ", test_sum)