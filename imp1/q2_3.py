import numpy as np 
from math import pow
import sys
import matplotlib.pyplot as plt
import time

def sigmoid(x):
    return 1/(1 + np.exp(x))

#get lambda
learn = 0.001
length = len(sys.argv[3])
lambdas = sys.argv[3][1:length-1]
lambdas = lambdas.replace("[", "")
lambdas = lambdas.replace("]", "")
lambdas = lambdas.split(',')
length = len(lambdas)
for i in range(length):
    lambdas[i] = float(lambdas[i])

#get x
train = np.genfromtxt(sys.argv[1], delimiter = ',')
train = np.delete(train, 256,axis=1)
train = np.insert(train, 0, 1, axis=1)
train = train/256
train_row = train.shape[0]

#get Y
target = np.genfromtxt(sys.argv[1], delimiter=',');
target=target[:, 256]

#get data from test
test = np.genfromtxt(sys.argv[2], delimiter=',')
test=np.delete(test, 256, axis=1)
test=np.insert(test, 0,1,axis=1)
test = test/256
test_row = test.shape[0]

#get Y
test_target = np.genfromtxt(sys.argv[2], delimiter=',');
test_target=test_target[:, 256]

#generate w
# w = np.zeros(257)
# w=np.full(257,-1)
#define epoch
epoch = 80

cont = 1
test_accuracy = []
train_accuracy = []
# x_axis = np.arange(1,epoch+1)

for z in range(length):
    w = np.zeros(257)

    #calculate batch on training data
    for i in range(epoch):
        #initalize gradient
        grad = np.zeros(257)
        #accumulate gradient
        for j in range(train_row):
            wtx = np.dot(-1*w.T, train[j]) 
            yhat = sigmoid(wtx)
            loss = (yhat-target[j]) * train[j]
            grad += (yhat - target[j])*train[j] 

        #update weight vector
        w -= learn*(grad + lambdas[z]*w)
        

        correct=0
        answer =-1   
        #collect data for train
    for k in range(train_row):
        predic = np.dot(w.T, train[k])
        if(predic >= 0):
            answer =1;
        else:
            answer =0
        if(answer == target[k]):
            correct+=1
    train_accuracy.append(correct/train_row)
    correct=0

    #collect dat for test
    for k in range(test_row):
        predic = np.dot(w.T, test[k])
        if(predic>=0):
            answer = 1;
        else:
            answer = 0
        if(answer == test_target[k]):
            correct+=1
    test_accuracy.append(correct/test_row)
    
#plot data
plt.plot(lambdas, test_accuracy, label="test accuracy")
plt.plot(lambdas, train_accuracy, label = "train accuracy")
plt.ylabel("Accuracy")
plt.xlabel("lambda")
plt.title("Accuracy vs Lambda value [0.001,0.01,0.1,1,10,100,1000]")
plt.legend()
plt.savefig("q2_3.png")
plt.show()