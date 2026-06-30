import numpy as np
import pandas as pd

class Node:

    def __init__(self, feature=None, threshold=None, left=None, right=None, label=None):

        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label

    def is_leaf(self):

        return self.label is not None
    

class DecisionTreeRegressor:

    def __init__(self, bagging = False, n_col = 5):
        self.max_depth = 9
        self.min_sample_split = 30
        self.min_sample_leaf = 10
        self.bagging = bagging
        self.n_col = n_col

    def variance(self, x):
        return np.var(x)
    
    def weighted_variance(self, x_l, x_r):
        return ((len(x_l) * self.variance(x_l)) + (len(x_r) * self.variance(x_r))) / (len(x_l) + len(x_r))

    def best_split(self, X, y):
        max_gain = -float("inf")
        best_feature = None
        best_threshold = None
        parent_var = np.var(y)

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

                child_var = self.weighted_variance(y_left, y_right)
                gain = parent_var - child_var

                if gain > max_gain:
                    if len(y_left) > self.min_sample_leaf and len(y_right) > self.min_sample_leaf:
                        max_gain = gain
                        best_feature = feature
                        best_threshold = threshold
        
        return best_feature, best_threshold, max_gain

    def build_tree(self, X, y, depth = 0):
        leaf_mean = np.mean(np.array(y))

        if np.var(y) < 1e-8:
            return Node(label = leaf_mean)
        
        if depth >= self.max_depth:
            return Node(label = leaf_mean)
        
        if len(y) < self.min_sample_split:
            return Node(label = leaf_mean)
        
        ifeature, ithreshold, gain = self.best_split(X, y)

        if ifeature is None or gain < 1e-8:
            return Node(label = leaf_mean)

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

    def evaluate(self, X_train, y_train, X_test, y_test):
        self.train(X_train, y_train)
        prediction = self.predict(X_test)

        return np.sqrt(np.mean((prediction - y_test) ** 2))




df = pd.read_csv("insurance.csv")

df['sex'] = df['sex'].map({'female': 0, 'male': 1})
df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
df = pd.get_dummies(df, columns=['region'], dtype=int)

#print(df.head())

X = df.drop(columns=["charges"])
y = df.loc[:,"charges"]

feature_names = list(X.columns)

#np.random.seed(420)

""" indices = np.arange(len(X))
np.random.shuffle(indices)
split_1 = int(0.7 * len(X))
split_2 = int(0.85 * len(X))

X_train = X.iloc[indices[:split_1],:]
y_train = y.iloc[indices[:split_1]]
X_val = X.iloc[indices[split_1:split_2],:]
y_val = y.iloc[indices[split_1:split_2]].values
X_test = X.iloc[indices[split_2:],:]
y_test = y.iloc[indices[split_2:]].values

tree = DecisionTreeRegressor()
tree.train(X_train, y_train)
print(tree.evaluate(X_train, y_train, X_train, y_train))
print(tree.evaluate(X_train, y_train, X_val, y_val))
print(tree.evaluate(X_train, y_train, X_test, y_test)) """










