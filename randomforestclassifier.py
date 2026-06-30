import numpy as np
import pandas as pd
from decisiontreeclassifier import Node, DecisionTreeClassifier

class RandomForestClassifier:

    def __init__(self, n_trees = 99, n_col = None):
        self.n_trees = n_trees
        self.n_col = n_col
        

    def bootstrap(self, X, y):
        index = np.random.choice(len(X), size = len(X), replace = True)

        X_boot = X.iloc[index,:]
        y_boot = y.iloc[index]

        return X_boot, y_boot, index
    
    def train(self, X, y):
        self.forest = []
        self.X = X
        self.y = y
        self.bootstrap_indices = []
        
        if self.n_col == None:
            self.n_col = int(np.sqrt(X.shape[1]))

        for _ in range(self.n_trees):
            tree = DecisionTreeClassifier(bagging = True, n_col = self.n_col)

            X_boot, y_boot, index = self.bootstrap(X, y)
            self.bootstrap_indices.append(index)
            tree.train(X_boot, y_boot)

            self.forest.append(tree)

    def predict_each(self, X):
        predictions = []

        for tree in self.forest:
            prediction = tree.predict(X)
            predictions.append(prediction)

        return np.array(predictions)
    
    def predict(self, X):
        final_prediction = []
        predictions = self.predict_each(X)

        for each in predictions.T:
            final = np.bincount(each).argmax()
            final_prediction.append(final)

        return np.array(final_prediction)
    
    def oob_predict(self):
        self.y_oob_indices = []
        final_predictions = []

        for i in range(self.X.shape[0]):
            votes = []

            for j, tree in enumerate(self.forest):

                if i not in self.bootstrap_indices[j]:
                    predi = tree.predict(self.X.iloc[[i]])[0]
                    votes.append(predi)

            if len(votes) > 0:
                final = np.bincount(votes).argmax()
                final_predictions.append(final)
                self.y_oob_indices.append(i)

        return np.array(final_predictions)

    def oob_accuracy(self):
        predic = self.oob_predict()
        return np.mean(predic == self.y.iloc[self.y_oob_indices].values)

    def accuaracy_test(self, X_test, y_test):
        prediction = self.predict(X_test)
        return np.sum(prediction == y_test) / len(prediction)
    



df = pd.read_csv("heart.csv")

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

np.random.seed(420)

indices = np.arange(len(X))
np.random.shuffle(indices)
split_1 = int(0.75 * len(X))

X_train = X.iloc[indices[:split_1],:]
y_train = y.iloc[indices[:split_1]]
X_test = X.iloc[indices[split_1:],:]
y_test = y.iloc[indices[split_1:]].values




forest = RandomForestClassifier(n_trees = 51)
forest.train(X_train, y_train)
print(forest.accuaracy_test(X_test, y_test))
print(forest.oob_predict())
print(forest.oob_accuracy())

print("===================")







    



