import numpy as np
import pandas as pd
from decisiontreeregression import Node, DecisionTreeRegressor

class RandomForestRegressor:

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
            tree = DecisionTreeRegressor(bagging = True, n_col = self.n_col)

            X_boot, y_boot, index = self.bootstrap(X, y)
            self.bootstrap_indices.append(set(index))
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
            final = np.mean(each)
            final_prediction.append(final)

        return np.array(final_prediction)
    
    def oob_predict(self):
        self.y_oob_indices = []
        final_predictions = []

        for i in range(self.X.shape[0]):
            pred_val = []

            for j, tree in enumerate(self.forest):

                if i not in self.bootstrap_indices[j]:
                    predi = tree.predict(self.X.iloc[[i]])[0]
                    pred_val.append(predi)

            if len(pred_val) > 0:
                final = np.mean(pred_val)
                final_predictions.append(final)
                self.y_oob_indices.append(i)

        return np.array(final_predictions)
    
    def oob_rmse(self):
        predic = self.oob_predict()
        return np.sqrt(np.mean((predic - self.y.iloc[self.y_oob_indices].values) ** 2))

    def test_rmse(self, X_test, y_test):
        prediction = self.predict(X_test)
        return np.sqrt(np.mean((prediction - y_test) ** 2))











