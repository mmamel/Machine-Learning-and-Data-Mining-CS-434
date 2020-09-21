import numpy as np

class Node():
	"""
	Node of decision tree

	Parameters:
	-----------
	prediction: int
		Class prediction at this node
	feature: int
		Index of feature used for splitting on
	split: int
		Categorical value for the threshold to split on for the feature
	left_tree: Node
		Left subtree
	right_tree: Node
		Right subtree
	"""
	def __init__(self, prediction, feature, split, left_tree, right_tree):
		self.prediction = prediction
		self.feature = feature
		self.split = split
		self.left_tree = left_tree
		self.right_tree = right_tree


class DecisionTreeClassifier():
	"""
	Decision Tree Classifier. Class for building the decision tree and making predictions

	Parameters:
	------------
	max_depth: int
		The maximum depth to build the tree. Root is at depth 0, a single split makes depth 1 (decision stump)
	"""

	def __init__(self, max_depth=None):
		self.max_depth = max_depth

	# take in features X and labels y
	# build a tree
	def fit(self, X, y):
		self.num_classes = len(set(y))
		self.root = self.build_tree(X, y, depth=1)

	# make prediction for each example of features X
	def predict(self, X):
		preds = [self._predict(example) for example in X]

		return preds

	# prediction for a given example
	# traverse tree by following splits at nodes
	def _predict(self, example):
		node = self.root
		while node.left_tree:
			if example[node.feature] < node.split:
				node = node.left_tree
			else:
				node = node.right_tree
		return node.prediction

	# accuracy
	def accuracy_score(self, X, y):
		preds = self.predict(X)
		accuracy = (preds == y).sum()/len(y)
		return accuracy

	# function to build a decision tree
	def build_tree(self, X, y, depth):
		num_samples, num_features = X.shape
		# which features we are considering for splitting on
		self.features_idx = np.arange(0, X.shape[1])

		# store data and information about best split
		# used when building subtrees recursively
		best_feature = None
		best_split = None
		best_gain = 0.0
		best_left_X = None
		best_left_y = None
		best_right_X = None
		best_right_y = None

		# what we would predict at this node if we had to
		# majority class
		num_samples_per_class = [np.sum(y == i) for i in range(self.num_classes)]
		prediction = np.argmax(num_samples_per_class)

		# if we haven't hit the maximum depth, keep building
		if depth <= self.max_depth:
			# consider each feature
			for feature in self.features_idx:
				# consider the set of all values for that feature to split on
				possible_splits = np.unique(X[:, feature])
				for split in possible_splits:
					# get the gain and the data on each side of the split
					# >= split goes on right, < goes on left
					gain, left_X, right_X, left_y, right_y = self.check_split(X, y, feature, split)
					# if we have a better gain, use this split and keep track of data
					if gain > best_gain:
						best_gain = gain
						best_feature = feature
						best_split = split
						best_left_X = left_X
						best_right_X = right_X
						best_left_y = left_y
						best_right_y = right_y
		
		# if we haven't hit a leaf node
		# add subtrees recursively
		if best_gain > 0.0:
			left_tree = self.build_tree(best_left_X, best_left_y, depth=depth+1)
			right_tree = self.build_tree(best_right_X, best_right_y, depth=depth+1)
			return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=left_tree, right_tree=right_tree)

		# if we did hit a leaf node
		return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=None, right_tree=None)


	# gets data corresponding to a split by using numpy indexing
	def check_split(self, X, y, feature, split):
		left_idx = np.where(X[:, feature] < split)
		right_idx = np.where(X[:, feature] >= split)
		left_X = X[left_idx]
		right_X = X[right_idx]
		left_y = y[left_idx]
		right_y = y[right_idx]
		# calculate gini impurity and gain for y, left_y, right_y
		gain = self.calculate_gini_gain(y, left_y, right_y)
		return gain, left_X, right_X, left_y, right_y

	def calculate_uncertainty(self, arr,arr_2, p,y):
		Cp = np.count_nonzero(arr)
		Cn = len(arr)-Cp
		uncertainty = 1 - ((Cp)/(Cp + Cn))**2 - ((Cn)/(Cp+Cn))**2
		if(p!= 0):
			C = np.count_nonzero(y) 
			#+ np.count_nonzero(arr_2)
			C_ = len(y)-C
			uncertainty *= (Cp+Cn)/(C+C_)	 
		return uncertainty

	def calculate_gini_gain(self, y, left_y, right_y):
		# not a leaf node
		# calculate gini impurity and gain
		gain = 0
		if len(left_y) > 0 and len(right_y) > 0:
			gain += self.calculate_uncertainty(y,y,0,y)
			gain -= self.calculate_uncertainty(left_y, right_y, 1,y)
			gain -= self.calculate_uncertainty(right_y, left_y,1,y)

			return gain
		# we hit leaf node
		# don't have any gain, and don't want to divide by 0
		else:
			return 0

class RandomForestClassifier():
	"""
	Random Forest Classifier. Build a forest of decision trees.
	Use this forest for ensemble predictions

	YOU WILL NEED TO MODIFY THE DECISION TREE VERY SLIGHTLY TO HANDLE FEATURE BAGGING

	Parameters:
	-----------
	n_trees: int
		Number of trees in forest/ensemble
	max_features: int
		Maximum number of features to consider for a split when feature bagging
	max_depth: int
		Maximum depth of any decision tree in forest/ensemble
	"""
	def __init__(self, n_trees, max_features, max_depth):
		self.n_trees = n_trees
		self.max_features = max_features
		self.max_depth = max_depth

		##################
		# YOUR CODE HERE #
		##################
	def predict(self, X):
		preds = [self._predict(example) for example in X]

		return preds

	# prediction for a given example
	# traverse tree by following splits at nodes
	def _predict(self, example):
		node = self.root
		while node.left_tree:
			if example[node.feature] < node.split:
				node = node.left_tree
			else:
				node = node.right_tree
		return node.prediction

	# accuracy
	def accuracy_score(self, X, y):
		preds = self.predict(X)
		accuracy = (preds == y).sum()/len(y)
		return accuracy
	def build_tree(self, X, y, depth):
		num_samples, num_features = X.shape
		# which features we are considering for splitting on
		self.features_idx = np.arange(0, X.shape[1])

		# store data and information about best split
		# used when building subtrees recursively
		best_feature = None
		best_split = None
		best_gain = 0.0
		best_left_X = None
		best_left_y = None
		best_right_X = None
		best_right_y = None

		# what we would predict at this node if we had to
		# majority class
		num_samples_per_class = [np.sum(y == i) for i in range(self.num_classes)]
		prediction = np.argmax(num_samples_per_class)

		# if we haven't hit the maximum depth, keep building
		if depth <= self.max_depth:
			# consider each feature
			for feature in self.features_idx:
				# consider the set of all values for that feature to split on
				possible_splits = np.unique(X[:, feature])
				for split in possible_splits:
					# get the gain and the data on each side of the split
					# >= split goes on right, < goes on left
					gain, left_X, right_X, left_y, right_y = self.check_split(X, y, feature, split)
					# if we have a better gain, use this split and keep track of data
					if gain > best_gain:
						best_gain = gain
						best_feature = feature
						best_split = split
						best_left_X = left_X
						best_right_X = right_X
						best_left_y = left_y
						best_right_y = right_y
		
		# if we haven't hit a leaf node
		# add subtrees recursively
		if best_gain > 0.0:
			left_tree = self.build_tree(best_left_X, best_left_y, depth=depth+1)
			right_tree = self.build_tree(best_right_X, best_right_y, depth=depth+1)
			return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=left_tree, right_tree=right_tree)

		# if we did hit a leaf node
		return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=None, right_tree=None)
	# gets data corresponding to a split by using numpy indexing
	def check_split(self, X, y, feature, split):
		left_idx = np.where(X[:, feature] < split)
		right_idx = np.where(X[:, feature] >= split)
		left_X = X[left_idx]
		right_X = X[right_idx]
		left_y = y[left_idx]
		right_y = y[right_idx]
		# calculate gini impurity and gain for y, left_y, right_y
		gain = self.calculate_gini_gain(y, left_y, right_y)
		return gain, left_X, right_X, left_y, right_y

	def calculate_uncertainty(self, arr,arr_2, p,y):
		Cp = np.count_nonzero(arr)
		Cn = len(arr)-Cp
		uncertainty = 1 - ((Cp)/(Cp + Cn))**2 - ((Cn)/(Cp+Cn))**2
		if(p!= 0):
			C = np.count_nonzero(arr) + np.count_nonzero(arr_2)
			C_ = len(arr)+len(arr_2)-C
			uncertainty *= (Cp+Cn)/(C+C_)	 
		return uncertainty

	def calculate_gini_gain(self, y, left_y, right_y):
		# not a leaf node
		# calculate gini impurity and gain
		gain = 0
		if len(left_y) > 0 and len(right_y) > 0:
			gain += self.calculate_uncertainty(y,y,0,y)
			gain -= self.calculate_uncertainty(left_y, right_y, 1,y)
			gain -= self.calculate_uncertainty(right_y, left_y,1,y)

			return gain
		# we hit leaf node
		# don't have any gain, and don't want to divide by 0
		else:
			return 0
	# fit all trees
	def fit(self, X, y):
		bagged_X, bagged_y = self.bag_data(X, y)
		print('Fitting Random Forest...\n')
		for i in range(self.n_trees):
			print(i+1, end='\t\r')
			self.num_classes = len(set(y))
			self.root = self.build_tree(bagged_X[i], bagged_y[i], depth=1)
			##################
			# YOUR CODE HERE #
			##################
		print()

	def bag_data(self, X, y, proportion=1.0):
		bagged_X = []
		bagged_y = []
		for i in range(self.n_trees):
			temp_x=[]
			temp_y=[]
			for j in range(2098):
				index = np.random.random_integers(0,2097)
				temp_x.append(X[index])
				temp_y.append(y[index])
			bagged_X.append(temp_x)
			bagged_y.append(temp_y)
			##################
			# YOUR CODE HERE #
			##################

		# ensure data is still numpy arrays
		return np.array(bagged_X), np.array(bagged_y)


	# def predict(self, X):
	# 	preds = []

	# 	# remove this one \/
	# 	preds = np.ones(len(X)).astype(int)
	# 	# ^that line is only here so the code runs

	# 	##################
	# 	# YOUR CODE HERE #
	# 	##################
	# 	return preds


################################################
# YOUR CODE GOES IN ADABOOSTCLASSIFIER         #
# MUST MODIFY THIS EXISTING DECISION TREE CODE #
################################################
class AdaBoostClassifier():
	def __init__(self, max_depth):
		self.max_depth = max_depth
		self.D = [np.full((2098), 1/2098)]
		self.alpha=[]
		self.feature=[]
		self.split = []
	def fit(self, X, y, M):
		self.num_classes = len(set(y))
		self.M=M
		self.root = self.build_tree(X, y)	
	def build_tree(self, X,y):
		for x in range(self.M):
			#learn
			# which features we are considering for splitting on
			self.features_idx = np.arange(0, X.shape[1])

			# store data and information about best split
			# used when building subtrees recursively
			best_feature = None
			best_split = None
			min_error = 9999999
			best_left_X = None
			best_left_y = None
			best_right_X = None
			best_right_y = None

			# what we would predict at this node if we had to
			# majority class
			num_samples_per_class = [np.sum(y == i) for i in range(self.num_classes)]
			prediction = np.argmax(num_samples_per_class)
			for feature in self.features_idx:
				# consider the set of all values for that feature to split on
				possible_splits = np.unique(X[:, feature])
				for split in possible_splits:
					# get the gain and the data on each side of the split
					# >= split goes on right, < goes on left
					error, left_X, right_X, left_y, right_y = self.check_split(X, y, feature, split, x)
					# if we have a better gain, use this split and keep track of data
					if error < min_error:
						min_error = error
						best_feature = feature
						best_split = split
						best_left_X = left_X
						best_right_X = right_X
						best_left_y = left_y
						best_right_y = right_y
			# print("min_error", min_error)
			# print("feature", best_feature)
			# print("D:", self.D[x])
			alpha_temp = 0.5 * np.log((1-min_error)/min_error)
			self.alpha.append(alpha_temp)
			# print("alpha", alpha_temp)
			correct = np.exp(-alpha_temp)
			wrong = np.exp(alpha_temp)
			D_temp=[]
			left = 0
			right = 0
			left_idx = np.where(X[:, best_feature] < best_split)
			right_idx = np.where(X[:, best_feature] >= best_split)

			for j in range(2098):
				if(left <  len(left_idx[0])):
					if(left_idx[0][left] == j):
						if (best_left_y[left] == 1):
							D_temp.append(self.D[x][j] * correct) 
						else:
							D_temp.append(self.D[x][j] * wrong)
						left+=1
				if(right < len(right_idx[0])):
					if (right_idx[0][right]==j):
						if(best_right_y[right] == -1):
							D_temp.append(self.D[x][j] * correct)
						else:
							D_temp.append(self.D[x][j] * wrong)
						right +=1
			D_temp = np.divide(D_temp, np.sum(D_temp))
			self.D.append(D_temp)
			self.feature.append(best_feature)
			self.split.append(best_split)


			

			#error
			#alpha
			#Update
			#normalize
	def check_split(self, X, y, feature, split, M):
		left_idx = np.where(X[:, feature] < split)
		right_idx = np.where(X[:, feature] >= split)
		# print("this is left_idx", left_idx)
		# print("this is right_idx", right_idx)
		left_X = X[left_idx]
		right_X = X[right_idx]
		left_y = y[left_idx]
		right_y = y[right_idx]
		# print("left y", left_y)
		# print("right y", right_y)
		error = 0
		# print(self.D[M])
		for x in range(len(left_y)):
			if (left_y[x] == -1):
				error += self.D[M][left_idx[0][x]]
				# print("wrong")
			# else:
			# 	print("right")
		for x in range(len(right_y)):
			# print(right_idx[0][x])
			# print(y[right_idx[0][x]])
			if (right_y[x] == 1):
				error+= self.D[M][right_idx[0][x]]
			# else:
				# print("right")
		return error, left_X, right_X, left_y, right_y

	# make prediction for each example of features X
	def predict(self, X):
		preds = [self._predict(example) for example in X]

		return preds

	# prediction for a given example
	# traverse tree by following splits at nodes
	def _predict(self, example):
		sign = 0
		for x in range(self.M):
			if(example[self.feature[x]]<self.split[x]):
				sign+=self.alpha[x] * 1
			else:
				sign+=self.alpha[x] * -1
		if(sign >= 0):
			sign = 1
		else:
			sign = -1
		return sign

	# accuracy
	def accuracy_score(self, X, y):
		preds = self.predict(X)
		accuracy = (preds == y).sum()/len(y)
		return accuracy

	


