import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import re
import math
import matplotlib.pyplot as plt
import time

vocab = 2000
start = time.time()

# Importing the dataset
imdb_data = pd.read_csv('IMDB.csv', delimiter=',')


def clean_text(text):

    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # remove the characters [\], ['] and ["]
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)
    #pattern = r'[^a-zA-z0-9\s]'
    #text = re.sub(pattern, '', text)

    # convert text to lowercase
    text = text.strip().lower()

    # replace punctuation characters with spaces
    filters = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    return text


# this vectorizer will skip stop words
vectorizer = CountVectorizer(
    stop_words="english",
    preprocessor=clean_text,
    max_features = 2000
)

# fit the vectorizer on the text
vectorizer.fit(imdb_data['review'])

# get the vocabulary
inv_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
vocabulary = [inv_vocab[i] for i in range(len(inv_vocab))]
n = 30000
training1=0
training0=0
a=1

#grab data from csv files
label = pd.read_csv("IMDB_labels.csv")
# training_data = pd.read_csv("IMDB.csv")

#initialize array for wi
y1 = np.zeros(vocab)
y0 = np.zeros(vocab)

vect_training_data = vectorizer.transform(imdb_data['review'])

# #convert testing data into numpy array
arr_training_label = label.to_numpy()
arr_training_data = vect_training_data.toarray()
#calculate p(y=1) and p(y=0) and p(xi)
for x in range(n):
    if np.equal(arr_training_label[x], "positive") == True:
        training1 += 1
        y1 = np.add(y1, arr_training_data[x])
#         y1 += arr_training_data[x]
    else:
        training0 +=1
        y0= np.add(y0, arr_training_data[x])
#         y0 += arr_training_data[x]
py1 = training1/n
py0 = training0/n
px1 = (y1+a) / (vocab + n*a)
px0 = (y0+a) / (vocab + n*a)

correct = 0
for x in range(n, n+10000):
    positive = np.log(py1)
    negative = np.log(py0)
    positive = np.multiply(arr_training_data[x], np.log(px1)).sum()
    negative = np.multiply(arr_training_data[x], np.log(px0)).sum()
    if positive > negative:
        if arr_training_label[x] == "positive":
            correct+=1
    else:
        if arr_training_label[x] == "negative":
            correct += 1
print("accuracy = ", correct/10000)

test_prediction1 = open("test-prediction1.csv", "w")

for x in range(n+10000, n+20000):
    positive = np.log(py1)
    negative = np.log(py0)
    positive = np.sum(np.multiply(arr_training_data[x], np.log(px1)))
    negative = np.sum(np.multiply(arr_training_data[x], np.log(px0)))
    if positive > negative:
        test_prediction1.write("1\n")
    else:
        test_prediction1.write("0\n")

test_prediction1.close()
best_a = 0
test_accuracy = []
alpha = []
a=0
curr_acc = 0
for x in range(11):
    correct = 0
    px1 = (y1+a) / (training1 + n*a)
    px0 = (y0+a) / (training0 + n*a)
    for x in range(n, n+10000):
        positive = np.log(py1)
        negative = np.log(py0)
        positive = np.sum(np.multiply(arr_training_data[x], np.log(px1)))
        negative = np.sum(np.multiply(arr_training_data[x], np.log(px0)))
        if positive > negative:
            if arr_training_label[x] == "positive":
                correct+=1
        else:
            if arr_training_label[x] == "negative":
                correct += 1
    if (correct/10000 >= curr_acc):
        best_a = a
        curr_acc = correct/10000
    test_accuracy.append(correct/10000)
    alpha.append(a)
    a+=0.2
print(best_a)
print(curr_acc)
plt.plot(alpha, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("alpha")
plt.title("Accuracy vs alpha value [0,0.2...2]")
plt.legend()
plt.savefig("q4.png")
plt.show()
px1 = (y1+best_a) / (vocab + n*best_a)
px0 = (y0+best_a) / (vocab + n*best_a)

test_prediction2 = open("test-prediction2.csv", "w")

for x in range(n+10000, n+20000):
    positive = np.log(py1)
    negative = np.log(py0)
    positive = np.sum(np.multiply(arr_training_data[x], np.log(px1)))
    negative = np.sum(np.multiply(arr_training_data[x], np.log(px0)))
    if positive > negative:
        test_prediction2.write("1\n")
    else:
        test_prediction2.write("0\n")

test_prediction2.close()
best_feature=0
best_max_df = 0
best_min_df=0
curr_acc = 0
a=best_a
for q in range(0,3):
    for w in range(1,3):
        for z in range(1,3):    
        # this vectorizer will skip stop words
            vocab = 2000+(q*1000)
            vectorizer2 = CountVectorizer(
                stop_words="english",
                preprocessor=clean_text,
                max_features = vocab,
                max_df = w*3000,
                min_df = z*100
            )

            # fit the vectorizer on the text
            vectorizer2.fit(imdb_data['review'])

            # get the vocabulary
            inv_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
            vocabulary = [inv_vocab[i] for i in range(len(inv_vocab))]
            training1=0
            training0=0

            # training_data = pd.read_csv("IMDB.csv")

            vect_training_data = vectorizer2.transform(imdb_data['review'])
            vocab = vect_training_data.shape[1]
            #initialize array for wi
            y1 = np.zeros(vocab)
            y0 = np.zeros(vocab)
            # #convert testing data into numpy array
            arr_training_data = vect_training_data.toarray()
    #calculate p(y=1) and p(y=0) and p(xi)
            for x in range(n):
                if np.equal(arr_training_label[x], "positive") == True:
                    y1 = np.add(y1, arr_training_data[x])
                else:
                    y0 = np.add(y0, arr_training_data[x])

            px1 = (y1+a) / (vocab + n*a)
            px0 = (y0+a) / (vocab + n*a)
            correct = 0
            for x in range(n, n+10000):
                positive = np.log(py1)
                negative = np.log(py0)
                positive = np.multiply(arr_training_data[x], np.log(px1)).sum()
                negative = np.multiply(arr_training_data[x], np.log(px0)).sum()
                if positive > negative:
                    if arr_training_label[x] == "positive":
                        correct+=1
                else:
                    if arr_training_label[x] == "negative":
                        correct += 1
            print("accuracy = ", correct/10000)
            print("max_df = ", w*3000)
            print("min_df", z*100)
            print("feature ", 2000 + (q*1000))
            print("-------------")
            if(correct/10000 > curr_acc):
                curr_acc = correct/10000
                best_feature = vocab
                best_max_df = w*3000
                best_min_df = z*100
test_prediction1 = open("test-prediction3.csv", "w")
vectorizer3 = CountVectorizer(
            stop_words="english",
            preprocessor=clean_text,
            max_features = best_feature,
            max_df = best_max_df,
            min_df = best_min_df
        )
    # fit the vectorizer on the text
vectorizer3.fit(imdb_data['review'])

        # get the vocabulary
inv_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
vocabulary = [inv_vocab[i] for i in range(len(inv_vocab))]
        
training1=0
training0=0
a=best_a

        # training_data = pd.read_csv("IMDB.csv")

        #initialize array for wi
y1 = np.zeros(best_feature)
y0 = np.zeros(best_feature)

vect_training_data = vectorizer3.transform(imdb_data['review'])

arr_training_data = vect_training_data.toarray()
#calculate p(y=1) and p(y=0) and p(xi)
for x in range(n):
    if np.equal(arr_training_label[x], "positive") == True:
        y1 = np.add(y1, arr_training_data[x])
    else:
        y0 = np.add(y0, arr_training_data[x])
px1 = (y1+a) / (best_feature + n*a)
px0 = (y0+a) / (best_feature + n*a)

#try here
correct = 0
for x in range(n, n+10000):
    positive = math.log(py1)
    negative = math.log(py0)
    positive = np.multiply(arr_training_data[x], np.log(px1)).sum()
    negative = np.multiply(arr_training_data[x], np.log(px0)).sum()
    if positive > negative:
        if arr_training_label[x] == "positive":
            correct+=1
    else:
        if arr_training_label[x] == "negative":
            correct += 1
print("accuracy = ", correct/10000)
print(best_feature, best_max_df, best_min_df)
for x in range(n+10000, n+20000):
    positive = np.log(py1)
    negative = np.log(py0)
    positive = np.multiply(arr_training_data[x], np.log(px1)).sum()
    negative = np.multiply(arr_training_data[x], np.log(px0)).sum()
    if positive > negative:
        test_prediction1.write("1\n")
    else:
        test_prediction1.write("0\n")

test_prediction1.close()