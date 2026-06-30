import numpy as np
import pandas as pd

df = pd.read_csv("heart.csv")

#print(df.head())
#print(df.shape)
#print(df.info())
#print(df.columns)
#print(df.loc[df["age"]>np.mean(df["age"])])
#print(np.mean(df["age"]))

X = df.iloc[:,:-1]
y = df.iloc[:,-1]
#print(y)

feature_names = list(X.columns)

#np.random.seed(420)

indices = np.arange(len(X))
np.random.shuffle(indices)
split_1 = int(0.7 * len(X))
split_2 = int(0.85 * len(X))

X_train = X.iloc[indices[:split_1],:]
y_train = y.iloc[indices[:split_1]]
X_val = X.iloc[indices[split_1:split_2],:]
y_val = y.iloc[indices[split_1:split_2]].values
X_test = X.iloc[indices[split_2:],:]
y_test = y.iloc[indices[split_2:]].values


#=========================================================
#DECISION TREE CLASS
#=========================================================


class Node:

    def __init__(self, feature=None, threshold=None, left=None, right=None, label=None):

        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label

    def is_leaf(self):

        return self.label is not None
    

class DecisionTreeClassifier:

    def __init__(self, bagging = False, n_col = 5):
        self.max_depth = 9
        self.min_sample_split = 30
        self.min_sample_leaf = 10
        self.bagging = bagging
        self.n_col = n_col

    def gini_impurity(self, x):
        n = len(x)

        if n == 0:
            return 0
        
        else:
            gini = 1

            for cls in np.unique(x):
                p = np.sum(x == cls) / n
                gini -= p ** 2

            return gini
    
    def weighted_gini(self, x_l, x_r):
        return ((len(x_l) * self.gini_impurity(x_l)) + (len(x_r) * self.gini_impurity(x_r))) / (len(x_l) + len(x_r))

    def best_split(self, X, y):
        best_gini = float("inf")
        best_feature = None
        best_threshold = None

        if self.bagging:
            features = np.random.choice(range(X.shape[1]), size = self.n_col , replace = False)
        else:
            features = [x for x in range(X.shape[1])]

        for feature in features:
            
            unq_val = np.unique(X.iloc[:,feature])
            thrshlds = (unq_val[1:] + unq_val[:-1]) / 2

            for threshold in thrshlds:

                logical = X.iloc[:, feature] <= threshold

                y_left = y[logical]
                y_right = y[~logical]

                gini = self.weighted_gini(y_left, y_right)

                if gini < best_gini:
                    if len(y_left) > self.min_sample_leaf and len(y_right) > self.min_sample_leaf:
                        best_gini = gini
                        best_feature = feature
                        best_threshold = threshold
        
        return best_feature, best_threshold

    def build_tree(self, X, y, depth = 0):
        majority_class = np.bincount(y.astype(int)).argmax()

        if len(np.unique(y)) == 1:
            return Node(label = y.iloc[0])
        
        if depth >= self.max_depth:
            return Node(label = majority_class)
        
        if len(y) < self.min_sample_split:
            return Node(label = majority_class)
        
        ifeature, ithreshold = self.best_split(X, y)

        if ifeature is None:
            return Node(label = majority_class)

        log = X.iloc[:, ifeature] <= ithreshold

        left_child = self.build_tree(X.loc[log], y[log], depth + 1)
        right_child = self.build_tree(X.loc[~log], y[~log], depth + 1)

        return Node(feature = ifeature, threshold = ithreshold, left = left_child, right = right_child)
    
    def train(self, X, y):
        self.tree = self.build_tree(X, y)


    def predict_one(self, node, x):
        if node.is_leaf():
            return node.label
        
        if x.iloc[node.feature] <= node.threshold:
            return self.predict_one(node.left, x)
        
        return self.predict_one(node.right, x)

    def predict(self, X):
        prediction = []

        for i in range(X.shape[0]):
            pred = self.predict_one(self.tree, X.iloc[i,:])
            prediction.append(pred)

        return np.array(prediction)

    def print_tree(self, tree = None, depth=0):
        if tree is None:
            tree = self.tree

        indent = "    " * depth

        if tree.is_leaf():
            print(f"{indent}Predict -> {tree.label}")
            return

        feature_name = feature_names[tree.feature]

        print(f"{indent}{feature_name} <= {tree.threshold:.3f}")

        print(f"{indent}├── True")
        self.print_tree(tree.left, depth + 1)

        print(f"{indent}└── False")
        self.print_tree(tree.right, depth + 1)

    def accuracy_test(self, X_train, y_train, X_test, y_test):
        self.train(X_train, y_train)
        prediction = self.predict(X_test)

        return np.sum(prediction == y_test) / len(prediction)


#tree = DecisionTree()
#tree.train(X_train, y_train)
#print(tree.accuracy_test(X_train, y_train, X_val, y_val))













