import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import argparse
import time
from utils import load_data, f1, accuracy_score, load_dictionary, dictionary_info, load_ada_data
from tree import DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier

def load_args():

	parser = argparse.ArgumentParser(description='arguments')
	parser.add_argument('--county_dict', default=1, type=int)
	parser.add_argument('--decision_tree', default=1, type=int)
	parser.add_argument('--random_forest', default=1, type=int)
	parser.add_argument('--ada_boost', default=1, type=int)
	parser.add_argument('--root_dir', default='./data/', type=str)
	args = parser.parse_args()

	return args


def county_info(args):
	county_dict = load_dictionary(args.root_dir)
	dictionary_info(county_dict)

def decision_tree_testing(x_train, y_train, x_test, y_test, depth):
	print('Decision Tree\n\n')
	clf = DecisionTreeClassifier(max_depth=depth)
	clf.fit(x_train, y_train)
	preds_train = clf.predict(x_train)
	preds_test = clf.predict(x_test)
	train_accuracy = accuracy_score(preds_train, y_train)
	test_accuracy = accuracy_score(preds_test, y_test)
	print('Train {}'.format(train_accuracy))
	print('Test {}'.format(test_accuracy))
	preds = clf.predict(x_test)
	print('F1 Test {}'.format(f1(y_test, preds)))
	return train_accuracy, test_accuracy, f1(y_train,preds_train), f1(y_test,preds)

def random_forest_testing(x_train, y_train, x_test, y_test, feat, tree):
	print('Random Forest\n\n')
	rclf = RandomForestClassifier(max_depth=7, max_features=feat, n_trees=tree)
	rclf.fit(x_train, y_train)
	preds_train = rclf.predict(x_train)
	preds_test = rclf.predict(x_test)
	train_accuracy = accuracy_score(preds_train, y_train)
	test_accuracy = accuracy_score(preds_test, y_test)
	print('Train {}'.format(train_accuracy))
	print('Test {}'.format(test_accuracy))
	preds = rclf.predict(x_test)
	print('F1 Test {}'.format(f1(y_test, preds)))
	preds_train = rclf.predict(x_train)
	return train_accuracy, test_accuracy, f1(y_train, preds_train), f1(y_test, preds)

def adaboost_testing(x_train, y_train, x_test, y_test, M):
	print("Adaboost Tree\n\n")
	aclf = AdaBoostClassifier(max_depth = 1)
	aclf.fit(x_train, y_train, M)
	preds_train = aclf.predict(x_train)
	preds_test = aclf.predict(x_test)
	train_accuracy = accuracy_score(preds_train, y_train)
	test_accuracy = accuracy_score(preds_test, y_test)
	print('Train {}'.format(train_accuracy))
	print('Test {}'.format(test_accuracy))
	preds = aclf.predict(x_test)
	print('F1 Test {}'.format(f1(y_test, preds)))
	preds_train = aclf.predict(x_train)
	return train_accuracy, test_accuracy, f1(y_train, preds_train), f1(y_test, preds)
###################################################
# Modify for running your experiments accordingly #
###################################################
if __name__ == '__main__':
	args = load_args()
	accuracy_train = []
	f1_tscore = []
	accuracy = []
	f1_score = []
	x_axis = np.arange(1, 26)
	max_features = [1,2,5,8,10,20,25,35,50]
	# x_axis = np.arange(10,220,10)
	random = np.arange(1,11)
	x_train, y_train, x_test, y_test = load_data(args.root_dir)
	L = np.arange(10,210,10)
	if args.county_dict == 1:
		county_info(args)
	if args.decision_tree == 1:
		for x in range(1,126):
			train_acc, test_acc, f1_train, f1_test = decision_tree_testing(x_train, y_train, x_test, y_test,x)
			accuracy_train.append(train_acc)
			accuracy.append(test_acc)
			f1_tscore.append(f1_train)
			f1_score.append(f1_test)
		plt.plot(x_axis, accuracy, label="Testing Accuracy")
		plt.plot(x_axis, f1_score, label = "Training F1 Score")
		plt.plot(x_axis, accuracy_train, label = "Training Accuracy")
		plt.plot(x_axis, f1_tscore, label = "Testing F1 score")
		plt.ylabel("Accuracy")
		plt.xlabel("Depth of Tree")
		plt.title("Accuracy vs Depth of tree")
		plt.legend()
		plt.savefig("q1_d.png")
		plt.show()
	x_axis = []
	if args.random_forest == 1:
		accuracy_train = []
		f1_tscore = []
		accuracy = []
		f1_score = []
		x_axis = np.arange(10,220,10)
		for x in range(10, 220, 10):
			train_acc, test_acc, f1_train, f1_test = random_forest_testing(x_train, y_train, x_test, y_test,11,x)
			accuracy_train.append(train_acc)
			accuracy.append(test_acc)
			f1_tscore.append(f1_train)
			f1_score.append(f1_test)
			# np.append(f1_train,f1_train)
			# np.append(f1_score, f1_test)
		plt.plot(x_axis, accuracy, label="Testing Accuracy")
		plt.plot(x_axis, f1_score, label = "Training F1 Score")
		plt.plot(x_axis, accuracy_train, label = "Training Accuracy")
		plt.plot(x_axis, f1_tscore, label = "Testing F1 score")
		plt.ylabel("Accuracy")
		plt.xlabel("N Trees")
		plt.title("Accuracy vs N Trees")
		plt.legend()
		plt.savefig("q2_b.png")
		plt.show()
		accuracy_train = []
		f1_tscore = []
		accuracy = []
		f1_score = []
		for x in max_features:
			train_acc, test_acc, f1_train, f1_test = random_forest_testing(x_train, y_train, x_test, y_test,x, 50)
			accuracy_train.append(train_acc)
			accuracy.append(test_acc)
			f1_tscore.append(f1_train)
			f1_score.append(f1_test)
		plt.plot(max_features, accuracy, label="Testing Accuracy")
		plt.plot(max_features, f1_score, label = "Training F1 Score")
		plt.plot(max_features, accuracy_train, label = "Training Accuracy")
		plt.plot(max_features, f1_tscore, label = "Testing F1 score")
		plt.ylabel("Accuracy")
		plt.xlabel("Max Features")
		plt.title("Accuracy vs Max Features")
		plt.legend()
		plt.savefig("q2_d.png")
		plt.show()

		#RAndom forest
		accuracy_train = []
		f1_tscore = []
		accuracy = []
		f1_score = []
		for x in range(10):
			np.random.seed(int(time.time()))
			train_acc, test_acc, f1_train, f1_test = random_forest_testing(x_train, y_train, x_test, y_test,25, 75)
			accuracy_train.append(train_acc)
			accuracy.append(test_acc)
			f1_tscore.append(f1_train)
			f1_score.append(f1_test)
		plt.plot(random, accuracy, label="Testing Accuracy")
		plt.plot(random, f1_score, label = "Training F1 Score")
		plt.plot(random, accuracy_train, label = "Training Accuracy")
		plt.plot(random, f1_tscore, label = "Testing F1 score")
		plt.ylabel("Accuracy")
		plt.xlabel("Trial")
		plt.title("Accuracy over random seeds max_feat = 25 max_tree = 80")
		plt.legend()
		plt.savefig("q2_e.png")
		plt.show()

		#ADABOOST
	accuracy_train = []
	f1_tscore = []
	accuracy = []
	f1_score = []
	x_train, y_train, x_test, y_test = load_ada_data(args.root_dir)
	for z in (L):
		train_acc, test_acc, f1_train, f1_test = adaboost_testing(x_train, y_train, x_test, y_test,z)
		accuracy_train.append(train_acc)
		accuracy.append(test_acc)
		f1_tscore.append(f1_train)
		f1_score.append(f1_test)
	plt.plot(L, accuracy, label="Testing Accuracy")
	plt.plot(L, f1_score, label = "Training F1 Score")
	plt.plot(L, accuracy_train, label = "Training Accuracy")
	plt.plot(L, f1_tscore, label = "Testing F1 score")
	plt.ylabel("Accuracy")
	plt.xlabel("Parameter L (base classfier/number of trees")
	plt.title("Accuracy vs L parameter")
	plt.legend()
	plt.savefig("q3_f.png")
	plt.show()
	print('Done')
	
	





