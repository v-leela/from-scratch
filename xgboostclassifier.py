import numpy as np
import pandas as pd
from decisiontreeclassifier import Node


class Logliklihood:
    def gradient_and_hessian(self, residual, pre_prob):
        gradient = -np.array(residual)
        hessian = np.array(pre_prob * (1 - pre_prob))

        return gradient, hessian


def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))


class XGBoostClassifier:
    def __init__(self, n_esti=599, learning_rate=0.33, lamb=300, gamma=0.1):
        self.n_esti = n_esti
        self.learning_rate = learning_rate
        self.trees = []
        self.loss = Logliklihood()
        self.lamb = lamb
        self.gamma = gamma
        self.max_depth = 10
        self.min_sample_leaf = 10
        self.min_sample_split = 30
        self.row_subsample = 0.8
        self.col_subsample = 0.8
        self.early_stopping_rounds = 20
        self.best_iteration = 0
        self.n_bins = 10

    def similarity(self, r, p):
        g, h = self.loss.gradient_and_hessian(r, p)

        return (np.sum(g) ** 2) / (np.sum(h) + self.lamb)

    def best_split(self, X, residuals, pre_prob):
        best_gain = -float("inf")
        best_threshold = None
        best_feature = None
        parent = self.similarity(residuals, pre_prob)

        for feature in list(X.columns):
            """ unq_val = np.unique(X.loc[:, feature])
            thresholds = (unq_val[1:] + unq_val[:-1]) / 2 """
            quantiles = np.linspace(0, 1, self.n_bins + 1)[1:-1]
            thresholds = np.quantile(np.array(X.loc[:, feature]), quantiles)

            for threshold in thresholds:
                logical = X.loc[:, feature] <= threshold

                r_left = residuals[logical]
                r_right = residuals[~logical]
                p_left = pre_prob[logical]
                p_right = pre_prob[~logical]

                left = self.similarity(r_left, p_left)
                right = self.similarity(r_right, p_right)

                gain = left + right - parent

                if gain > best_gain:
                    if (
                        gain > self.gamma
                        and len(r_left) > self.min_sample_leaf
                        and len(r_right) > self.min_sample_leaf
                    ):
                        best_gain = gain
                        best_feature = feature
                        best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def build_tree(self, X, residuals, pre_prob, depth=0):
        leaf_val = np.sum(residuals) / (np.sum(pre_prob * (1 - pre_prob)) + self.lamb)

        if depth > self.max_depth:
            return Node(label=leaf_val)

        if len(residuals) < self.min_sample_split:
            return Node(label=leaf_val)

        ifeature, ithreshold, igain = self.best_split(X, residuals, pre_prob)

        if ifeature is None:
            return Node(label=leaf_val)

        self.features_gain[ifeature] += float(igain)

        log = X.loc[:, ifeature] <= ithreshold

        left_child = self.build_tree(
            X.loc[log], residuals[log], pre_prob[log], depth + 1
        )
        right_child = self.build_tree(
            X.loc[~log], residuals[~log], pre_prob[~log], depth + 1
        )

        return Node(ifeature, ithreshold, left_child, right_child)

    def predict_onetree(self, node, x):
        if node.is_leaf():
            return node.label

        if x[node.feature] <= node.threshold:
            return self.predict_onetree(node.left, x)

        return self.predict_onetree(node.right, x)

    def predict_tree(self, X):
        prediction = []

        for i in range(X.shape[0]):
            pred = self.predict_onetree(self.btree, X.iloc[i, :])
            prediction.append(pred)

        return np.array(prediction)

    def boosting(self, X, y):
        self.initial_prediction = np.log(np.sum(y == 1) / np.sum(y == 0))
        self.iprediction = np.full(y.shape, self.initial_prediction)
        pre_prob = sigmoid(self.iprediction)
        residuals = np.array(y - pre_prob)

        self.val_rmse = []
        best_rmse = float("inf")
        rnds_noimporve = 0

        for i in range(self.n_esti):
            row_idx = np.random.choice(
                len(residuals), int(self.row_subsample * len(residuals)), replace=False
            )
            col_features = np.random.choice(
                X.shape[1], int(self.col_subsample * X.shape[1]), replace=False
            )
            self.btree = self.build_tree(
                X.iloc[row_idx][self.features[col_features]],
                residuals[row_idx],
                pre_prob[row_idx],
            )
            self.trees.append(self.btree)
            self.iprediction += self.learning_rate * self.predict_tree(X)
            pre_prob = sigmoid(self.iprediction)
            residuals = np.array(y - pre_prob)

            vali_rmse = self.evaluate(self.X_val, self.y_val)
            self.val_rmse.append(vali_rmse)

            if vali_rmse < best_rmse:
                best_rmse = vali_rmse
                self.best_iteration = i
                rnds_noimporve = 0
            else:
                rnds_noimporve += 1

            if rnds_noimporve >= self.early_stopping_rounds:
                print(f"Early stopping at tree {i + 1}")
                self.trees = self.trees[: -1 * self.early_stopping_rounds]
                break

    def fit(self, X, y, X_val, y_val):
        self.X = X
        self.y = y
        self.X_val = X_val
        self.y_val = y_val
        self.features = np.array(self.X.columns)
        self.features_gain = dict(zip(self.X.columns, [0] * X.shape[1]))

        self.boosting(self.X, self.y)

    def predict_each(self, x, n_trees=None):
        final_pred = self.initial_prediction
        if n_trees is None:
            n_trees = len(self.trees)

        for tree in self.trees[:n_trees]:
            final_pred += self.learning_rate * self.predict_onetree(tree, x)

        return sigmoid(final_pred)

    def predict(self, X):
        final_predictions = []

        for i in range(X.shape[0]):
            final_predictions.append(self.predict_each(X.iloc[i]))

        return np.array(final_predictions)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)

        return np.sqrt(np.mean((np.array(y_test) - predictions) ** 2))
        return np.mean(np.abs(y_test - predictions))

    def accuracy(self, X_test, y_test):
        predictions = self.predict(X_test)
        predictions[predictions >= 0.5] = 1
        predictions[predictions < 0.5] = 0

        return np.mean(y_test == predictions)

    def val_rmse_each_tree(self):
        return np.array(self.val_rmse)

    def feature_imp(self):
        total = sum(self.features_gain.values())

        self.ftrpercent = {
            key: (val / total) * 100 for key, val in self.features_gain.items()
        }

        for key, val in self.ftrpercent.items():
            print(f"{key} : {val}%")


df = pd.read_csv("heart.csv")

X = df.drop(columns=["target"])
y = df.loc[:, "target"]

np.random.seed(42)

indices = np.arange(len(X))
np.random.shuffle(indices)
split_1 = int(0.7 * len(X))
split_2 = int(0.85 * len(X))

X_train = X.iloc[indices[:split_1], :]
y_train = y.iloc[indices[:split_1]].values
X_val = X.iloc[indices[split_1:split_2], :]
y_val = y.iloc[indices[split_1:split_2]].values
X_test = X.iloc[indices[split_2:], :]
y_test = y.iloc[indices[split_2:]].values

xtree = XGBoostClassifier()
xtree.fit(X_train, y_train, X_val, y_val)
print(xtree.val_rmse_each_tree()[:-20], xtree.best_iteration)
# print(np.min(xtree.val_rmse_each_tree()))
print(f"validation set accuracy --> {xtree.accuracy(X_val, y_val)}")
print(f"test set accuracy -->       {xtree.accuracy(X_test, y_test)}")
xtree.feature_imp()
print("----- xgboost done -----")
