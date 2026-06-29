import numpy as np
from decisiontreeclassifier import Node


class SquaredError:
    def gradient_and_hessian(self, residual):
        gradient = -np.array(residual)
        hessian = np.ones_like(residual)

        return gradient, hessian


class XGBoostRegressor:
    def __init__(self, n_esti=99, learning_rate=0.3, lamb=0.1):
        self.n_esti = n_esti
        self.learning_rate = learning_rate
        self.trees = []
        self.loss = SquaredError()
        self.lamb = lamb
        self.max_depth=10
        self.min_sample_leaf=10
        self.min_sample_split=30

    def similarity(self, r):
        g, h = self.loss.gradient_and_hessian(
            r,
        )

        return (np.sum(g)**2)/np.sum(h)

    def best_split(self, X, residuals):
        best_gain = -float("inf")
        best_threshold = None
        best_feature = None
        parent=self.similarity(residuals)

        for feature in range(X.shape[1]):
            unq_val = np.unique(X.iloc[:, feature])
            thresholds = (unq_val[1:] + unq_val[:-1]) / 2

            for threshold in thresholds:
                logical = X.iloc[:, feature] <= threshold

                r_left = residuals[logical]
                r_right = residuals[~logical]

                left=self.similarity(r_left)
                right=self.similarity(r_right)

                gain=left+right-parent
                
                if gain > best_gain:
                    if len(r_left) > self.min_sample_leaf and len(r_right) > self.min_sample_leaf:
                        best_gain = gain
                        best_feature = feature
                        best_threshold = threshold

        return best_feature,best_threshold,best_gain
    
    def build_tree(self,X,residuals,depth=0):
        leaf_val=np.mean(residuals)

        if depth>self.max_depth:
            return Node(label=leaf_val)
        
        if len(residuals)<self.min_sample_split:
            return Node(label=leaf_val)
        
        ifeature,ithreshold,igain=self.best_split(X,residuals)

        if ifeature is None or igain < 1e-8:
            return Node(label = leaf_val)
        
        log=X.iloc[:,ifeature]<=ithreshold

        left_child=self.build_tree(X.loc[log],residuals[log],depth+1)
        right_child=self.build_tree(X.loc[~log],residuals[~log],depth+1)

        return Node(ifeature,ithreshold,left_child,right_child)
    
    def predict_onetree(self, node, x):
        if node.is_leaf():
            return node.label
        
        if x.iloc[node.feature] <= node.threshold:
            return self.predict_onetree(node.left, x)
        
        return self.predict_onetree(node.right, x)

    def predict_tree(self, X):
        prediction = []

        for i in range(X.shape[0]):
            pred = self.predict_onetree(self.btree, X.iloc[i,:])
            prediction.append(pred)

        return np.array(prediction)

    def boosting(self,X,y):
        self.initial_prediction=np.mean(y)
        self.iprediction=np.full(y.shape,self.initial_prediction)
        residuals=np.array(y-self.iprediction)

        for _ in range(self.n_esti):
            self.btree=self.build_tree(X,residuals)
            self.trees.append(self.btree)
            self.iprediction+=self.learning_rate*self.predict_tree(X)
            residuals=y-self.iprediction

    def fit(self,X,y):
        self.X=X
        self.y=y
        self.boosting(self.X,self.y)


    def predict_each(self,x):
        final_pred=self.initial_prediction
        
        for tree in self.trees:
            final_pred+=self.learning_rate*self.predict_onetree(tree,x)

        return final_pred
    
    def predict(self,X):
        final_predictions=[]

        for i in range(X.shape[0]):
            final_predictions.append(self.predict_each(X.iloc[i]))

        return np.array(final_predictions)
    
    def evaluate(self,X_test,y_test):
        predictions=self.predict(X_test)

        return np.sqrt(np.mean((np.array(y_test)-predictions)**2))
    


