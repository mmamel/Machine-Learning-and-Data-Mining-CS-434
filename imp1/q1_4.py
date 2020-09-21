import numpy as np 
from math import pow
import sys
import matplotlib.pyplot as plt

np.set_printoptions(suppress = True)

#get X 
train = np.genfromtxt(sys.argv[1], delimiter=',');
train=np.delete(train, 13, axis=1)
train=np.insert(train, 0, 1, axis=1)
train_row = train.shape[0]

#get Y
target = np.genfromtxt(sys.argv[1], delimiter=',');
target=target[:, 13]

#get data from test
test = np.genfromtxt(sys.argv[2], delimiter=',')
test=np.delete(test, 13, axis=1)
test=np.insert(test, 0,1,axis=1)
test_row = test.shape[0]

#get Y
test_target = np.genfromtxt(sys.argv[2], delimiter=',');
test_target=test_target[:, 13]
train_ASE =[]
test_ASE=[]

#append random features
for i in range(1,11):

    #append random reatures
    for j in range(2):
        random_train = np.random.standard_normal(train_row)
        random_train = random_train.reshape(train_row, 1)
        random_test = np.random.standard_normal(test_row)
        random_test = random_test.reshape(test_row, 1)
        train = np.append(train, random_train, axis=1)
        test = np.append(test, random_test, axis=1)
    #calculate weight vector
    xtx = np.dot(train.T, train)
    inv = np.linalg.inv(xtx)
    invxt = np.dot(inv, train.T)
    w = np.dot(invxt, target)
    # w = np.dot(np.dot(np.linalg.inv(np.dot(train.T, train)), train.T), target)
    # print(w.reshape(train_col, 1))
    
    # #find ASE for training
    train_sum=0
    print("ASE for dummy: ", (2*i))
    for x in range(train_row):
        train_sum+= pow(target[x] - np.dot(w.T, train[x]),2)
    train_sum/=train_row

    print("training ASE = ", train_sum)
    train_ASE.append(train_sum)

    test_sum=0
    for x in range(test_row):
        test_sum+= pow(test_target[x] - np.dot(w.T, test[x]),2)
    test_sum/=test_row
    print("testing ASE = ", test_sum)
    test_ASE.append(test_sum)

plt.plot([2,4,6,8,10,12,14,16,18,20],train_ASE, label = "train ASE")
plt.plot([2,4,6,8,10,12,14,16,18,20],test_ASE, label = "test ASE")
plt.ylabel("ASE")
plt.xlabel("number of random features")
plt.savefig('q1_4.png')
plt.show()